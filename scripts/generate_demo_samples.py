#!/usr/bin/env python3
"""Generate a realistic local demo sample pack (one-shots & loops) for testing CLAP + MIR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000


def _norm(y: np.ndarray, peak: float = 0.92) -> np.ndarray:
    m = float(np.max(np.abs(y)))
    if m > 1e-6:
        y = y / m * peak
    return y.astype(np.float32)


def _lowpass(y: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, min(cutoff, SR * 0.45), btype="low", fs=SR, output="sos")
    return signal.sosfilt(sos, y)


def _highpass(y: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, max(cutoff, 20.0), btype="high", fs=SR, output="sos")
    return signal.sosfilt(sos, y)


def _bandpass(y: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [low, min(high, SR * 0.45)], btype="bandpass", fs=SR, output="sos")
    return signal.sosfilt(sos, y)


def synth_kick(f_start: float = 165.0, f_end: float = 48.0, dur: float = 0.48, click: float = 0.35, drive: float = 1.6) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    freq = f_end + (f_start - f_end) * np.exp(-t * 34.0)
    phase = 2.0 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t * 9.5)
    # Beater click transient
    rng = np.random.default_rng(42)
    transient = _bandpass(rng.standard_normal(len(t)), 900, 4800) * np.exp(-t * 140.0) * click
    y = np.tanh((body + transient) * drive)
    return _norm(y)


def synth_808_sub(freq: float = 43.65, dur: float = 1.15) -> np.ndarray:
    """Deep tuned 808 sub bass (default F1 = 43.65 Hz)."""
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    glide = freq + 95.0 * np.exp(-t * 55.0)
    phase = 2.0 * np.pi * np.cumsum(glide) / SR
    y = np.sin(phase) + 0.22 * np.sin(2.0 * phase)
    env = np.minimum(t / 0.005, 1.0) * np.exp(-t * 2.6)
    return _norm(np.tanh(y * env * 1.5))


def synth_snare(dur: float = 0.32, tone_hz: float = 195.0, lofi: bool = False) -> np.ndarray:
    rng = np.random.default_rng(101 if not lofi else 202)
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    body = np.sin(2.0 * np.pi * (tone_hz + 70.0 * np.exp(-t * 45.0)) * t) * np.exp(-t * 22.0)
    wires = _bandpass(rng.standard_normal(len(t)), 1400, 9500 if not lofi else 4200) * np.exp(-t * 14.0)
    y = 0.55 * body + 0.85 * wires
    if lofi:
        # Subtle tape saturation + vinyl crackle
        y = np.tanh(y * 2.2)
        y = _lowpass(y, 5200)
    return _norm(y)


def synth_hihat(dur: float = 0.14, open_hat: bool = False) -> np.ndarray:
    rng = np.random.default_rng(303 if not open_hat else 404)
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    noise = rng.standard_normal(len(t))
    # Metallic square cluster (TR-808 style)
    ratios = [205.3, 304.4, 369.6, 522.7, 540.0, 800.0]
    metal = sum(np.sign(np.sin(2.0 * np.pi * f * 5.2 * t)) for f in ratios) / len(ratios)
    y = _highpass(0.65 * noise + 0.35 * metal, 6800)
    decay = 5.5 if open_hat else 36.0
    env = np.exp(-t * decay)
    return _norm(y * env)


def synth_clap(dur: float = 0.36) -> np.ndarray:
    rng = np.random.default_rng(505)
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    noise = _bandpass(rng.standard_normal(len(t)), 900, 3800)
    env = np.zeros_like(t)
    for offset in (0.0, 0.011, 0.023, 0.036):
        mask = t >= offset
        dt = t[mask] - offset
        env[mask] += np.exp(-dt * (65.0 if offset < 0.03 else 13.0))
    return _norm(noise * env)


def synth_rhodes_chord(freqs: list[float], dur: float = 2.6, dusty: bool = True) -> np.ndarray:
    """Warm electric piano chord with bell tine attack & optional vinyl dust."""
    rng = np.random.default_rng(606)
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    y = np.zeros_like(t)
    for i, f in enumerate(freqs):
        # Slight strum offset
        delay = i * 0.018
        mask = t >= delay
        dt = t[mask] - delay
        # Tremolo
        trem = 1.0 + 0.14 * np.sin(2.0 * np.pi * 4.2 * dt)
        fund = np.sin(2.0 * np.pi * f * dt) * np.exp(-dt * 1.15)
        h2 = 0.25 * np.sin(2.0 * np.pi * 2.0 * f * dt) * np.exp(-dt * 2.1)
        tine = 0.18 * np.sin(2.0 * np.pi * 7.0 * f * dt) * np.exp(-dt * 18.0)
        y[mask] += (fund + h2 + tine) * trem * np.minimum(dt / 0.008, 1.0)
    y = np.tanh(y * 0.75)
    if dusty:
        y = _lowpass(y, 3400)
        crackle = rng.standard_normal(len(t)) * 0.004
        pops = (rng.random(len(t)) > 0.99965).astype(np.float32) * rng.uniform(-0.08, 0.08, len(t))
        y = y + _highpass(crackle + pops, 1800)
    return _norm(y)


def synth_ambient_pad(freqs: list[float], dur: float = 5.5) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    y = np.zeros_like(t)
    for f in freqs:
        for detune in (-0.004, 0.0, 0.0045):
            ff = f * (1.0 + detune)
            # Soft sawtooth approximated with harmonics
            sig = sum((1.0 / k) * np.sin(2.0 * np.pi * ff * k * t) for k in range(1, 6))
            y += sig
    y = _lowpass(y, 1600)
    fade_in = np.minimum(t / 1.1, 1.0)
    fade_out = np.minimum((dur - t) / 1.2, 1.0)
    shimmer = 1.0 + 0.15 * np.sin(2.0 * np.pi * 0.35 * t)
    return _norm(y * fade_in * fade_out * shimmer)


def synth_drum_loop(bpm: float = 122.0, bars: int = 2, style: str = "break") -> np.ndarray:
    beat_sec = 60.0 / bpm
    total_beats = bars * 4
    total_len = int(SR * total_beats * beat_sec)
    y = np.zeros(total_len, dtype=np.float32)

    kick = synth_kick(160, 50, 0.35, click=0.4)
    snare = synth_snare(0.24, tone_hz=200, lofi=(style == "break"))
    ch = synth_hihat(0.08, open_hat=False)
    oh = synth_hihat(0.26, open_hat=True)
    clap = synth_clap(0.25)

    def place(buf: np.ndarray, sound: np.ndarray, beat_pos: float, gain: float = 1.0) -> None:
        idx = int(beat_pos * beat_sec * SR)
        end = min(len(buf), idx + len(sound))
        if idx < len(buf):
            buf[idx:end] += sound[: end - idx] * gain

    for bar in range(bars):
        b0 = bar * 4.0
        if style == "break":
            for kpos in (0.0, 1.5, 2.5):
                place(y, kick, b0 + kpos, 0.95)
            for spos in (1.0, 3.0):
                place(y, snare, b0 + spos, 0.85)
            for step in range(8):
                place(y, ch, b0 + step * 0.5, 0.35 if step % 2 == 0 else 0.22)
        else:  # house top/full groove
            for beat in range(4):
                place(y, kick, b0 + beat, 0.95)
                place(y, oh, b0 + beat + 0.5, 0.38)
            place(y, clap, b0 + 1.0, 0.72)
            place(y, clap, b0 + 3.0, 0.72)
            for step in range(16):
                if step % 4 != 2:
                    place(y, ch, b0 + step * 0.25, 0.18)

    return _norm(y)


def synth_acid_bass_loop(bpm: float = 128.0, bars: int = 2) -> np.ndarray:
    """Resonant 303-style acid bassline loop in A minor."""
    step_sec = (60.0 / bpm) / 4.0
    notes = [
        55.0, 55.0, 65.41, 55.0, 73.42, 55.0, 82.41, 65.41,
        55.0, 55.0, 110.0, 82.41, 73.42, 65.41, 55.0, 49.0,
    ] * bars
    total_len = int(len(notes) * step_sec * SR)
    y = np.zeros(total_len, dtype=np.float32)

    for i, freq in enumerate(notes):
        n_samp = int(step_sec * SR)
        t = np.linspace(0, step_sec, n_samp, endpoint=False)
        saw = 2.0 * ((t * freq) % 1.0) - 1.0
        # Resonant lowpass approximation via formant peak
        cutoff = 420.0 + 1800.0 * np.exp(-t * 22.0)
        filt = _lowpass(saw, float(cutoff.mean()), order=4)
        res = np.sin(2.0 * np.pi * (freq * 6.0) * t) * np.exp(-t * 20.0) * 0.35
        env = np.minimum(t / 0.004, 1.0) * np.exp(-t * 9.0)
        seg = np.tanh((filt + res) * env * 2.4)
        start = i * n_samp
        end = min(total_len, start + n_samp)
        y[start:end] += seg[: end - start]

    return _norm(y)


def synth_arp_loop(bpm: float = 120.0, bars: int = 2) -> np.ndarray:
    """Bright analog synth arpeggio loop in C major."""
    step_sec = (60.0 / bpm) / 4.0
    arp_freqs = [261.63, 329.63, 392.00, 493.88, 523.25, 493.88, 392.00, 329.63] * (bars * 2)
    total_len = int(len(arp_freqs) * step_sec * SR)
    y = np.zeros(total_len, dtype=np.float32)
    for i, freq in enumerate(arp_freqs):
        n_samp = int(step_sec * SR)
        t = np.linspace(0, step_sec, n_samp, endpoint=False)
        sig = np.sin(2.0 * np.pi * freq * t) + 0.45 * np.sin(2.0 * np.pi * 2 * freq * t) + 0.2 * np.sin(2.0 * np.pi * 3 * freq * t)
        env = np.minimum(t / 0.005, 1.0) * np.exp(-t * 11.0)
        start = i * n_samp
        end = min(total_len, start + n_samp)
        y[start:end] += (sig * env)[: end - start]
    return _norm(y)


def synth_riser_fx(dur: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(909)
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    noise = rng.standard_normal(len(t))
    # Sweeping pitch & noise crescendo
    sweep_f = 180.0 * (2.0 ** (t / dur * 4.0))
    phase = 2.0 * np.pi * np.cumsum(sweep_f) / SR
    tone = np.sin(phase) * 0.45
    env = (t / dur) ** 2.2
    y = (_bandpass(noise, 400, 5000) * 0.65 + tone) * env
    return _norm(y)


def synth_guitar_pluck_loop(bpm: float = 100.0, bars: int = 2) -> np.ndarray:
    """Karplus-Strong plucked acoustic guitar chord progression in G major."""
    beat_sec = 60.0 / bpm
    total_len = int(SR * bars * 4 * beat_sec)
    y = np.zeros(total_len, dtype=np.float32)
    chords = [
        (0.0, [196.00, 246.94, 293.66, 392.00]),  # G maj
        (2.0, [164.81, 246.94, 293.66, 329.63]),  # E min
        (4.0, [130.81, 196.00, 261.63, 329.63]),  # C maj
        (6.0, [146.83, 220.00, 293.66, 369.99]),  # D maj
    ]
    rng = np.random.default_rng(707)

    def karplus(freq: float, dur: float) -> np.ndarray:
        n_samp = int(SR * dur)
        delay = max(2, int(SR / freq))
        buf = rng.uniform(-1.0, 1.0, delay).astype(np.float32)
        out = np.zeros(n_samp, dtype=np.float32)
        idx = 0
        for i in range(n_samp):
            out[i] = buf[idx]
            nxt = (idx + 1) % delay
            buf[idx] = 0.996 * 0.5 * (buf[idx] + buf[nxt])
            idx = nxt
        return out

    for beat, freqs in chords:
        for k, f in enumerate(freqs):
            pluck = karplus(f, 1.8)
            start = int((beat * beat_sec + k * 0.025) * SR)
            end = min(total_len, start + len(pluck))
            if start < total_len:
                y[start:end] += pluck[: end - start] * 0.4
    return _norm(y)


def generate_pack(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, np.ndarray]] = [
        ("Drums/One-Shots/Kick_Acoustic_Punchy.wav", synth_kick(175, 52, 0.42, click=0.55, drive=1.5)),
        ("Drums/One-Shots/Kick_Tight_Club.wav", synth_kick(190, 46, 0.36, click=0.30, drive=2.2)),
        ("Drums/One-Shots/Snare_Acoustic_Crisp.wav", synth_snare(0.30, tone_hz=205, lofi=False)),
        ("Drums/One-Shots/Snare_Lofi_Dusty.wav", synth_snare(0.34, tone_hz=175, lofi=True)),
        ("Drums/One-Shots/HiHat_Closed_Tight.wav", synth_hihat(0.11, open_hat=False)),
        ("Drums/One-Shots/HiHat_Open_Shimmer.wav", synth_hihat(0.55, open_hat=True)),
        ("Drums/One-Shots/Clap_Studio_Dry.wav", synth_clap(0.34)),
        ("Drums/Loops/DrumBreak_Dusty_120bpm.wav", synth_drum_loop(120.0, bars=2, style="break")),
        ("Drums/Loops/HouseGroove_Full_124bpm.wav", synth_drum_loop(124.0, bars=2, style="house")),
        ("Bass/Bass_808_Sub_F.wav", synth_808_sub(43.65, 1.2)),
        ("Bass/Acid_303_Loop_128bpm_Amin.wav", synth_acid_bass_loop(128.0, bars=2)),
        ("Keys/Rhodes_Warm_Dusty_Amin7.wav", synth_rhodes_chord([220.0, 261.63, 329.63, 392.0, 493.88], dur=2.6, dusty=True)),
        ("Keys/ElectricPiano_Bright_Cmaj9.wav", synth_rhodes_chord([261.63, 329.63, 392.0, 493.88, 587.33], dur=2.4, dusty=False)),
        ("Synths/Pad_Lush_Ambient_Cmaj.wav", synth_ambient_pad([130.81, 196.0, 246.94, 329.63], dur=5.2)),
        ("Synths/SynthArp_Analog_120bpm_Cmaj.wav", synth_arp_loop(120.0, bars=2)),
        ("Guitar/Guitar_Pluck_Loop_100bpm_Gmaj.wav", synth_guitar_pluck_loop(100.0, bars=2)),
        ("FX/Riser_Noise_Sweep.wav", synth_riser_fx(3.8)),
    ]

    written: list[Path] = []
    for rel_path, audio in specs:
        full_path = out_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(full_path), audio, SR, subtype="PCM_16")
        written.append(full_path)
    return written


if __name__ == "__main__":
    default_out = Path(__file__).resolve().parent.parent / "demo_samples"
    files = generate_pack(default_out)
    print(f"Created {len(files)} demo audio files in {default_out}")
