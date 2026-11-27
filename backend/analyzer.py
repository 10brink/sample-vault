"""Audio MIR analysis: loading, BPM, musical key, one-shot vs loop, waveform peaks."""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from .config import ANALYSIS_SAMPLE_RATE, CLAP_SAMPLE_RATE, MAX_ANALYSIS_SECONDS

logger = logging.getLogger(__name__)

# Krumhansl-Kessler / Temperley hybrid pitch-class profiles (C through B)
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)
MAJOR_PROFILE = (MAJOR_PROFILE - MAJOR_PROFILE.mean()) / (MAJOR_PROFILE.std() + 1e-8)
MINOR_PROFILE = (MINOR_PROFILE - MINOR_PROFILE.mean()) / (MINOR_PROFILE.std() + 1e-8)

NOTE_NAMES_MAJOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
NOTE_NAMES_MINOR = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

# Camelot wheel mapping indexed by semitone (0=C .. 11=B)
CAMELOT_MAJOR = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
CAMELOT_MINOR = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]

LOOP_HINT_RE = re.compile(
    r"(?:^|[\s_\-\./])(loops?|breaks?|grooves?|top[\s_-]?loops?|full[\s_-]?drums?|drum[\s_-]?loops?|arp[\s_-]?loops?|\d{2,3}\s*bpm)(?:$|[\s_\-\./])",
    re.IGNORECASE,
)
ONESHOT_HINT_RE = re.compile(
    r"(?:^|[\s_\-\./])(one[\s_-]?shots?|oneshots?|shots?|hits?|stabs?|single|kicks?|snares?|claps?|hats?|hihats?|rims?|toms?|cymbals?|crashes?|rides?|808s?)(?:$|[\s_\-\./])",
    re.IGNORECASE,
)
BPM_HINT_RE = re.compile(r"(?:^|[\s_\-\./])(\d{2,3})(?:\s*bpm)(?:$|[\s_\-\./])", re.IGNORECASE)


@dataclass
class AudioAnalysis:
    duration_sec: float
    sample_rate: int
    channels: int
    bpm: Optional[float]
    key_name: Optional[str]
    camelot: Optional[str]
    sample_type: str  # "one-shot" or "loop"
    type_confidence: float
    energy: float
    brightness: float
    waveform_peaks: list[float]
    audio_48k: np.ndarray


def _decode_with_afconvert(path: Path) -> tuple[np.ndarray, int, int, float]:
    """Fallback decoder using macOS native afconvert for m4a/caf/mp3/aifc without ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEF32", str(path), str(tmp_path)],
            capture_output=True,
            check=True,
            timeout=30,
        )
        _ = proc
        data, sr = sf.read(str(tmp_path), dtype="float32", always_2d=True)
        channels = data.shape[1]
        duration_sec = float(len(data)) / float(sr)
        return data, int(sr), int(channels), duration_sec
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_audio_file(path: Path) -> tuple[np.ndarray, int, int, float]:
    """Load audio as float32 (samples, channels), returning (data, sr, channels, true_duration)."""
    try:
        info = sf.info(str(path))
        true_duration = float(info.duration)
        sr = int(info.samplerate)
        channels = int(info.channels)
        max_frames = int(MAX_ANALYSIS_SECONDS * sr)
        data, sr = sf.read(str(path), frames=max_frames, dtype="float32", always_2d=True)
        if true_duration <= 0 and len(data) > 0:
            true_duration = float(len(data)) / float(sr)
        return data, int(sr), int(channels), true_duration
    except Exception:
        pass

    try:
        y, sr = librosa.load(str(path), sr=None, mono=False, duration=MAX_ANALYSIS_SECONDS)
        if y.ndim == 1:
            data = y[:, np.newaxis].astype(np.float32)
            channels = 1
        else:
            data = y.T.astype(np.float32)
            channels = data.shape[1]
        try:
            true_duration = float(librosa.get_duration(path=str(path)))
        except Exception:
            true_duration = float(len(data)) / float(sr)
        return data, int(sr), int(channels), true_duration
    except Exception as librosa_err:
        if os.path.exists("/usr/bin/afconvert"):
            return _decode_with_afconvert(path)
        raise librosa_err


def compute_waveform_peaks(mono: np.ndarray, num_bins: int = 120) -> list[float]:
    """Generate a compact normalized peak array for instant UI waveform rendering."""
    n = len(mono)
    if n == 0:
        return [0.0] * num_bins
    if n < num_bins:
        vals = np.abs(mono)
        peak = float(vals.max()) or 1.0
        padded = np.interp(
            np.linspace(0, n - 1, num_bins),
            np.arange(n),
            vals / peak,
        )
        return [round(float(v), 3) for v in padded]

    edges = np.linspace(0, n, num_bins + 1, dtype=np.int64)
    peaks = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        seg = mono[edges[i] : edges[i + 1]]
        if len(seg) == 0:
            continue
        abs_seg = np.abs(seg)
        max_v = float(abs_seg.max())
        rms_v = float(np.sqrt(np.mean(seg * seg)))
        peaks[i] = 0.72 * max_v + 0.28 * rms_v

    max_peak = float(peaks.max())
    if max_peak > 1e-6:
        peaks /= max_peak
    return [round(float(v), 3) for v in peaks]


def _onset_periodicity(onset_env: np.ndarray, sr: int) -> float:
    """Measure rhythmic autocorrelation peak strength (0.0 = flat/non-rhythmic, 1.0 = steady beat)."""
    if len(onset_env) < 32 or float(onset_env.std()) < 1e-5:
        return 0.0
    oe = onset_env - onset_env.mean()
    ac = np.correlate(oe, oe, mode="full")[len(oe) - 1 :]
    if ac[0] <= 1e-6:
        return 0.0
    ac = ac / ac[0]
    hop_sec = 512.0 / float(sr)
    min_lag = max(2, int(0.22 / hop_sec))
    max_lag = min(len(ac) - 1, int(2.0 / hop_sec))
    if max_lag <= min_lag:
        return 0.0
    return float(np.max(ac[min_lag:max_lag]))


def _classify_sample_type(
    y: np.ndarray,
    sr: int,
    duration_sec: float,
    onset_env: np.ndarray,
    onset_frames: np.ndarray,
    path_str: str,
) -> tuple[str, float]:
    """Classify audio sample as 'one-shot' or 'loop' with confidence."""
    score = 0.0
    n_onsets = int(len(onset_frames))
    periodicity = _onset_periodicity(onset_env, sr)

    # 1. Duration priors
    if duration_sec < 1.35:
        score -= 2.4
    elif duration_sec < 2.8:
        score -= 0.75
    elif duration_sec < 3.6:
        score += 0.25
    elif duration_sec <= 24.0:
        score += 1.25
    else:
        score += 0.9

    # 2. Envelope decay & sustained energy ratio across 8 windows
    n = len(y)
    if n >= 8:
        chunks = np.array_split(y, 8)
        rms_chunks = np.array([float(np.sqrt(np.mean(c * c))) for c in chunks], dtype=np.float32)
        peak_rms = float(rms_chunks.max()) + 1e-8
        norm_rms = rms_chunks / peak_rms
        active_fraction = float(np.mean(norm_rms > 0.16))
        head_rms = float(norm_rms[:2].mean())
        tail_rms = float(norm_rms[-2:].mean())
        tail_ratio = tail_rms / (head_rms + 1e-6)

        if active_fraction <= 0.375 or (tail_ratio < 0.20 and duration_sec < 3.5):
            score -= 1.35
        elif active_fraction >= 0.75 and tail_ratio >= 0.32:
            score += 0.9

    # 3. Onset density and rhythmic repetition
    if duration_sec >= 1.5:
        if n_onsets >= 5:
            score += 1.6
        elif n_onsets >= 3:
            score += 0.7
        elif n_onsets <= 2:
            score -= 1.4
    else:
        if n_onsets <= 2:
            score -= 0.9

    if periodicity > 0.32 and duration_sec >= 1.5 and n_onsets >= 3:
        score += 1.0
    elif periodicity < 0.14 and duration_sec < 4.5:
        score -= 0.75

    # 4. Path/filename hints (gentle tie-breaker)
    if LOOP_HINT_RE.search(path_str):
        score += 0.8
    if ONESHOT_HINT_RE.search(path_str):
        score -= 0.7

    sample_type = "loop" if score >= 0.2 else "one-shot"
    confidence = 1.0 / (1.0 + math.exp(-abs(score)))
    return sample_type, round(float(confidence), 3)


def _detect_bpm(
    onset_env: np.ndarray,
    sr: int,
    duration_sec: float,
    sample_type: str,
    onset_count: int,
    path_str: str,
) -> Optional[float]:
    """Estimate BPM for loops / rhythmic material; returns None for unpitched/short/flat sounds."""
    if duration_sec < 1.5:
        return None
    if sample_type == "one-shot" and onset_count < 4:
        return None
    if len(onset_env) < 20 or float(onset_env.max()) < 1e-5:
        return None

    # Require actual rhythmic events + periodicity so pads/risers/chords get BPM=None
    periodicity = _onset_periodicity(onset_env, sr)
    if onset_count < 3 or (onset_count < 5 and periodicity < 0.22):
        return None

    try:
        if hasattr(librosa.feature, "tempo"):
            tempos = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
            tempo_val = float(np.median(tempos))
        else:
            tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            tempo_val = float(tempos[0])
    except Exception:
        return None

    if not math.isfinite(tempo_val) or tempo_val <= 10.0:
        return None

    # Check filename BPM hint for octave/polyrhythm disambiguation
    hint_match = BPM_HINT_RE.search(path_str)
    hint_bpm = float(hint_match.group(1)) if hint_match else None

    if hint_bpm and 50.0 <= hint_bpm <= 220.0:
        for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
            cand = tempo_val * mult
            if abs(cand - hint_bpm) / hint_bpm <= 0.05:
                tempo_val = hint_bpm
                break
    else:
        while tempo_val < 68.0:
            tempo_val *= 2.0
        while tempo_val > 176.0:
            tempo_val /= 2.0

    # Snap near-integer tempos cleanly (e.g. 120.04 -> 120.0)
    nearest_int = round(tempo_val)
    if abs(tempo_val - nearest_int) <= 0.35:
        return float(nearest_int)
    return round(float(tempo_val), 1)


def _detect_key(y: np.ndarray, sr: int, duration_sec: float) -> tuple[Optional[str], Optional[str]]:
    """Detect musical key (Krumhansl-Schmuckler on HPSS harmonic CQT chroma + bass weighting)."""
    if len(y) < int(sr * 0.08):
        return None, None

    # Pad very short tonal hits (like 808s/kicks/stabs) so CQT has enough context
    y_work = y
    if len(y_work) < int(sr * 0.6):
        y_work = np.pad(y_work, (0, int(sr * 0.6) - len(y_work)))

    # Harmonic-percussive separation to isolate pitch content
    try:
        y_harm, y_perc = librosa.effects.hpss(y_work, margin=1.5)
    except Exception:
        y_harm, y_perc = y_work, np.zeros_like(y_work)

    harm_rms = float(np.sqrt(np.mean(y_harm * y_harm)))
    perc_rms = float(np.sqrt(np.mean(y_perc * y_perc)))
    harm_ratio = harm_rms / (harm_rms + perc_rms + 1e-8)

    flatness = float(np.median(librosa.feature.spectral_flatness(y=y_work)))

    # Purely noisy/unpitched percussion (snare, hi-hat, shaker, clap, noise riser)
    if harm_ratio < 0.22 and flatness > 0.06:
        return None, None

    # Compute CQT chroma on harmonic component + bass-register chroma for chord root clarity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, n_chroma=12, bins_per_octave=36)
            chroma_bass = librosa.feature.chroma_cqt(
                y=y_harm,
                sr=sr,
                fmin=librosa.note_to_hz("C1"),
                n_octaves=3,
                n_chroma=12,
                bins_per_octave=24,
            )
            min_f = min(chroma.shape[1], chroma_bass.shape[1])
            chroma = 0.72 * chroma[:, :min_f] + 0.28 * chroma_bass[:, :min_f]
        except Exception:
            chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr, n_chroma=12)

    if chroma.size == 0:
        return None, None

    # Weight frames by RMS energy so quiet tails don't corrupt the key
    frame_energy = librosa.feature.rms(y=y_harm)[0]
    min_len = min(chroma.shape[1], len(frame_energy))
    chroma = chroma[:, :min_len]
    weights = frame_energy[:min_len]
    if float(weights.sum()) < 1e-6:
        return None, None

    chroma_vec = (chroma * weights[np.newaxis, :]).sum(axis=1)
    if float(chroma_vec.std()) < 1e-6:
        return None, None

    # Check if pitch-class energy is sharply concentrated on a single pitch (e.g. tuned 808/kick)
    chroma_sum = float(chroma_vec.sum()) + 1e-8
    chroma_norm_dist = chroma_vec / chroma_sum
    sorted_pc = np.sort(chroma_norm_dist)[::-1]
    top_pc = int(np.argmax(chroma_norm_dist))

    chroma_z = (chroma_vec - chroma_vec.mean()) / (chroma_vec.std() + 1e-8)

    best_corr = -2.0
    best_key = None
    best_camelot = None

    for shift in range(12):
        maj_prof = np.roll(MAJOR_PROFILE, shift)
        min_prof = np.roll(MINOR_PROFILE, shift)
        c_maj = float(np.dot(chroma_z, maj_prof) / 12.0)
        c_min = float(np.dot(chroma_z, min_prof) / 12.0)

        if c_maj > best_corr:
            best_corr = c_maj
            best_key = f"{NOTE_NAMES_MAJOR[shift]} maj"
            best_camelot = CAMELOT_MAJOR[shift]
        if c_min > best_corr:
            best_corr = c_min
            best_key = f"{NOTE_NAMES_MINOR[shift]} min"
            best_camelot = CAMELOT_MINOR[shift]

    # Single-note tonal hits (like 808 bass note or tuned kick) -> report root note if very clear
    if duration_sec < 2.2 and sorted_pc[0] > 0.42 and (sorted_pc[0] / (sorted_pc[1] + 1e-6)) > 2.2:
        if harm_ratio >= 0.28:
            return NOTE_NAMES_MINOR[top_pc], None

    if best_corr < 0.46 or (harm_ratio < 0.25 and best_corr < 0.66):
        return None, None

    return best_key, best_camelot


def analyze_audio_file(path: Path) -> AudioAnalysis:
    """Run full MIR analysis on an audio file and prepare 48kHz waveform for CLAP."""
    data, orig_sr, channels, true_duration = load_audio_file(path)

    # Mix to mono float32
    mono_orig = data.mean(axis=1).astype(np.float32)

    # Peak-normalize gentle check (keep raw energy before normalizing)
    rms_raw = float(np.sqrt(np.mean(mono_orig * mono_orig))) if len(mono_orig) else 0.0
    # Map RMS roughly into [0, 1] using dB scale (-42dB -> 0, -4dB -> 1)
    if rms_raw > 1e-6:
        db = 20.0 * math.log10(rms_raw)
        energy = max(0.0, min(1.0, (db + 42.0) / 38.0))
    else:
        energy = 0.0

    # Waveform peaks on original mono
    waveform_peaks = compute_waveform_peaks(mono_orig, num_bins=120)

    # Resample for MIR analysis (22.05kHz) and CLAP (48kHz)
    if orig_sr != ANALYSIS_SAMPLE_RATE:
        y_22k = librosa.resample(mono_orig, orig_sr=orig_sr, target_sr=ANALYSIS_SAMPLE_RATE)
    else:
        y_22k = mono_orig

    if orig_sr != CLAP_SAMPLE_RATE:
        y_48k = librosa.resample(mono_orig, orig_sr=orig_sr, target_sr=CLAP_SAMPLE_RATE)
    else:
        y_48k = mono_orig

    # Spectral centroid -> normalized brightness [0, 1]
    try:
        centroid = librosa.feature.spectral_centroid(y=y_22k, sr=ANALYSIS_SAMPLE_RATE)
        med_centroid = float(np.median(centroid))
        # Log scale from 150 Hz (0.0) to 7000 Hz (1.0)
        brightness = max(
            0.0,
            min(1.0, (math.log2(max(med_centroid, 150.0)) - math.log2(150.0)) / (math.log2(7000.0) - math.log2(150.0))),
        )
    except Exception:
        brightness = 0.5

    # Onset detection (higher delta suppresses tremolo/vinyl crackle false onsets)
    try:
        onset_env = librosa.onset.onset_strength(y=y_22k, sr=ANALYSIS_SAMPLE_RATE)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=ANALYSIS_SAMPLE_RATE,
            delta=0.22,
            wait=4,
            backtrack=False,
        )
    except Exception:
        onset_env = np.zeros(1, dtype=np.float32)
        onset_frames = np.array([], dtype=np.int64)

    path_str = str(path)
    sample_type, type_conf = _classify_sample_type(
        y_22k,
        ANALYSIS_SAMPLE_RATE,
        true_duration,
        onset_env,
        onset_frames,
        path_str,
    )
    bpm = _detect_bpm(
        onset_env,
        ANALYSIS_SAMPLE_RATE,
        true_duration,
        sample_type,
        int(len(onset_frames)),
        path_str,
    )
    key_name, camelot = _detect_key(y_22k, ANALYSIS_SAMPLE_RATE, true_duration)

    return AudioAnalysis(
        duration_sec=round(true_duration, 3),
        sample_rate=orig_sr,
        channels=channels,
        bpm=bpm,
        key_name=key_name,
        camelot=camelot,
        sample_type=sample_type,
        type_confidence=type_conf,
        energy=round(energy, 3),
        brightness=round(brightness, 3),
        waveform_peaks=waveform_peaks,
        audio_48k=y_48k.astype(np.float32),
    )
