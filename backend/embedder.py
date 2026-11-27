"""CLAP (Contrastive Language-Audio Pretraining) local embedder & zero-shot tagger."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from .config import CLAP_MODEL_ID, CLAP_SAMPLE_RATE, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Curated music production taxonomy for zero-shot CLAP auto-categorization
CATEGORY_PROMPTS: list[tuple[str, str]] = [
    ("Kick", "kick drum bass drum thump one-shot"),
    ("Snare", "snare drum rimshot crack percussive hit"),
    ("Hi-Hat", "hi-hat cymbal closed open metallic tick"),
    ("Clap & Snap", "hand clap finger snap percussive"),
    ("Percussion", "percussion bongo conga shaker tom woodblock"),
    ("Drum Loop", "drum beat breakbeat groove rhythm loop"),
    ("Bass & 808", "sub bass 808 bassline synth bass low end"),
    ("Keys & Piano", "electric piano rhodes acoustic piano keyboard chord"),
    ("Synth & Lead", "synthesizer lead arpeggio analog synth melody"),
    ("Pad & Atmosphere", "ambient pad lush synth drone texture atmosphere"),
    ("Guitar & Plucked", "electric acoustic guitar pluck strum chord"),
    ("Strings & Brass", "orchestral strings violin horn brass section"),
    ("Vocal", "human voice vocal chop singing chant"),
    ("FX & Riser", "sound effect riser sweep impact transition noise"),
]

# Sonic texture / vibe descriptors for automatic descriptive tags
CHARACTER_PROMPTS: list[tuple[str, str]] = [
    ("punchy", "punchy tight transient hard-hitting impact"),
    ("warm", "warm rich analog vintage mellow tone"),
    ("dusty", "dusty lo-fi vinyl crackle tape saturated"),
    ("bright", "bright crisp airy shimmering high frequency"),
    ("dark", "dark moody deep muffled low-pass filtered"),
    ("distorted", "distorted overdriven gritty crunchy fuzz"),
    ("acoustic", "acoustic natural organic live studio recorded"),
    ("electronic", "electronic synthetic digital club sound design"),
    ("subby", "deep sub bass heavy booming low frequency"),
    ("lush", "lush dreamy wide stereo reverb spacious"),
    ("dry", "dry close-mic tight clean studio sound"),
    ("jazzy", "jazzy soulful Rhodes seventh chord smooth"),
    ("acid", "squelchy resonant 303 acid filter synth"),
    ("metallic", "metallic bell fm clangy resonant chime"),
    ("atmospheric", "cinematic atmospheric ethereal ambient texture"),
]


def _extract_feature_tensor(out: object) -> torch.Tensor:
    """Safely unwrap tensor from transformers ClapModel outputs across versions."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("audio_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            if val.ndim == 3:
                return val[:, 0, :]
            return val
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Unexpected CLAP output type: {type(out)}")


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        return (vec / norm).astype(np.float32)
    return vec.astype(np.float32)


class ClapEmbedder:
    """Thread-safe lazy loader and inference engine for LAION CLAP."""

    def __init__(self, model_id: str = CLAP_MODEL_ID) -> None:
        self.model_id = model_id
        self._lock = threading.Lock()
        self._model: Optional[ClapModel] = None
        self._processor: Optional[ClapProcessor] = None
        self._device: str = self._select_device()
        self._loading: bool = False
        self._load_error: Optional[str] = None

        self._cat_labels: list[str] = []
        self._cat_matrix: Optional[np.ndarray] = None
        self._char_labels: list[str] = []
        self._char_matrix: Optional[np.ndarray] = None

    @staticmethod
    def _select_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def device(self) -> str:
        return self._device

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._loading = True
            self._load_error = None
            try:
                logger.info("Loading CLAP model '%s' on %s...", self.model_id, self._device)
                try:
                    # Prefer local Hugging Face cache first so offline restarts are instant
                    processor = ClapProcessor.from_pretrained(self.model_id, local_files_only=True)
                    model = ClapModel.from_pretrained(self.model_id, local_files_only=True)
                except Exception:
                    # Use token=False for public LAION checkpoints so stale local HF tokens don't 401
                    processor = ClapProcessor.from_pretrained(self.model_id, token=False)
                    model = ClapModel.from_pretrained(self.model_id, token=False)

                model.eval()
                try:
                    model = model.to(self._device)
                except Exception as dev_err:
                    logger.warning("Falling back to CPU from %s: %s", self._device, dev_err)
                    self._device = "cpu"
                    model = model.to("cpu")

                self._processor = processor
                self._model = model
                self._precompute_tag_embeddings()
                logger.info("CLAP model ready on %s.", self._device)
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load CLAP model")
                raise
            finally:
                self._loading = False

    def _precompute_tag_embeddings(self) -> None:
        self._cat_labels = [label for label, _ in CATEGORY_PROMPTS]
        cat_texts = [prompt for _, prompt in CATEGORY_PROMPTS]
        self._cat_matrix = self._embed_text_batch_unlocked(cat_texts)

        self._char_labels = [label for label, _ in CHARACTER_PROMPTS]
        char_texts = [prompt for _, prompt in CHARACTER_PROMPTS]
        self._char_matrix = self._embed_text_batch_unlocked(char_texts)

    def _embed_text_batch_unlocked(self, texts: list[str]) -> np.ndarray:
        assert self._processor is not None and self._model is not None
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
            feats = _extract_feature_tensor(out).detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        return feats / norms

    def embed_text(self, query: str) -> np.ndarray:
        """Compute L2-normalized 512-dim CLAP embedding for a plain-English query."""
        self.ensure_loaded()
        with self._lock:
            mat = self._embed_text_batch_unlocked([query.strip()])
            return mat[0]

    def embed_audio(self, audio_48k: np.ndarray, sr: int = CLAP_SAMPLE_RATE) -> np.ndarray:
        """Compute L2-normalized 512-dim CLAP embedding for a mono 48kHz waveform."""
        self.ensure_loaded()
        assert self._processor is not None and self._model is not None

        y = np.asarray(audio_48k, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=-1)
        if len(y) == 0:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # CLAP expects up to 10s chunks at 48kHz; for longer files average up to 3 windows
        max_len = sr * 10
        if len(y) <= max_len:
            windows = [y]
        else:
            mid = max(0, (len(y) - max_len) // 2)
            end = max(0, len(y) - max_len)
            windows = [y[:max_len], y[mid : mid + max_len], y[end : end + max_len]]

        with self._lock:
            inputs = self._processor(
                audio=windows,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.get_audio_features(**inputs)
                feats = _extract_feature_tensor(out).detach().cpu().numpy().astype(np.float32)

        # Average normalized chunk embeddings
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        feats = feats / norms
        avg = feats.mean(axis=0)
        return _l2_normalize(avg)

    def classify_tags(
        self,
        audio_emb: np.ndarray,
        sample_type: str = "one-shot",
        duration_sec: float = 1.0,
    ) -> tuple[str, list[str]]:
        """Zero-shot classify primary instrument category and top sonic character tags."""
        self.ensure_loaded()
        assert self._cat_matrix is not None and self._char_matrix is not None

        cat_scores = (self._cat_matrix @ audio_emb).copy()

        # Prevent very short one-shots from being labeled "Drum Loop"
        if "Drum Loop" in self._cat_labels:
            dl_idx = self._cat_labels.index("Drum Loop")
            if sample_type == "one-shot" and duration_sec < 1.5:
                cat_scores[dl_idx] -= 0.15
            elif sample_type == "loop":
                cat_scores[dl_idx] += 0.03

        best_idx = int(np.argmax(cat_scores))
        category = self._cat_labels[best_idx]

        char_scores = self._char_matrix @ audio_emb
        top_indices = np.argsort(char_scores)[::-1]
        tags: list[str] = []
        best_char_score = float(char_scores[top_indices[0]])
        for idx in top_indices[:4]:
            sc = float(char_scores[idx])
            if len(tags) < 2 or (sc >= best_char_score - 0.08 and sc > 0.12):
                tags.append(self._char_labels[int(idx)])
            if len(tags) >= 3:
                break

        return category, tags


embedder = ClapEmbedder()
