#!/usr/bin/env python3
"""MCU-oriented log-mel extraction for fixed-duration 16 kHz audio.

Processing order (mirrored by main/logmel_features.c):
  PCM float -> DC removal -> Hamming STFT power -> HTK mel filterbank
  -> 10*log10 -> sample-max reference -> top_db clip.
Per-mel CMVN is applied later from the stored log-mel image (logmel_cnn.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from audio_dataset import DEFAULT_CLASS_NAMES


DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 256
DEFAULT_N_MELS = 32
DEFAULT_F_MIN_HZ = 0.0
DEFAULT_F_MAX_HZ = None
DEFAULT_LOG_FLOOR = 1e-12
DEFAULT_TOP_DB = 80.0
DEFAULT_CMVN_STD_FLOOR_DB = 2.0


# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------

def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    """Convert Hz to the HTK-style mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert HTK-style mel values back to Hz."""
    return 700.0 * (np.power(10.0, np.asarray(mel) / 2595.0) - 1.0)


def mel_filter_bin_edges(
    sr: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Return the ``n_mels + 2`` FFT-bin edges of the triangular mel filters.

    Filter ``m`` rises over bins ``[edges[m], edges[m + 1])`` and falls over
    ``[edges[m + 1], edges[m + 2])``. The firmware embeds these exact integers
    (tools/convert_logmel_tflite_to_c*.py), so this is the single source of truth.
    """
    if sr <= 0 or n_fft <= 0 or n_mels <= 0:
        raise ValueError("sr, n_fft, and n_mels must be greater than zero")
    if f_max is None:
        f_max = sr / 2.0
    if not 0.0 <= f_min < f_max <= sr / 2.0:
        raise ValueError("mel frequency range must satisfy 0 <= f_min < f_max <= Nyquist")

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(np.int32)
    return np.clip(bins, 0, n_fft // 2)


def create_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Create triangular mel filters with deterministic FFT-bin boundaries."""
    bins = mel_filter_bin_edges(sr, n_fft, n_mels, f_min=f_min, f_max=f_max)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for mel_index in range(n_mels):
        left, center, right = (int(value) for value in bins[mel_index : mel_index + 3])
        if center > left:
            filters[mel_index, left:center] = (
                np.arange(left, center, dtype=np.float32) - left
            ) / float(center - left)
        if right > center:
            filters[mel_index, center:right] = 1.0 - (
                np.arange(center, right, dtype=np.float32) - center
            ) / float(right - center)
    return filters


# ---------------------------------------------------------------------------
# STFT and log-mel
# ---------------------------------------------------------------------------

def stft_frame_count(sample_count: int, n_fft: int, hop_length: int) -> int:
    """Return the no-padding STFT frame count shared by Python and C."""
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be greater than zero")
    if sample_count < n_fft:
        return 1
    return 1 + (sample_count - n_fft) // hop_length


def fix_audio_length(audio: np.ndarray, sample_count: int) -> np.ndarray:
    """Crop or zero-pad one waveform to the model's fixed input duration."""
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero")
    fixed = np.asarray(audio, dtype=np.float32).reshape(-1)
    fixed = np.nan_to_num(fixed, copy=False)
    if fixed.size >= sample_count:
        return fixed[:sample_count].copy()
    return np.pad(fixed, (0, sample_count - fixed.size)).astype(np.float32)


def compute_mel_power_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    *,
    f_min: float = DEFAULT_F_MIN_HZ,
    f_max: float | None = DEFAULT_F_MAX_HZ,
) -> np.ndarray:
    """Compute an uncentered Hamming-window mel power spectrogram."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError("audio is empty")
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))

    # DC removal is cheap in C and prevents microphone bias from entering the
    # lowest mel bins. RMS normalization is intentionally not applied: the
    # sample-max dB reference and per-mel CMVN handle recording level.
    samples = (samples - np.mean(samples, dtype=np.float64)).astype(np.float32)
    frame_count = stft_frame_count(samples.size, n_fft, hop_length)
    window = np.hamming(n_fft).astype(np.float32)
    power = np.empty((n_fft // 2 + 1, frame_count), dtype=np.float32)

    for frame_index in range(frame_count):
        start = frame_index * hop_length
        frame = samples[start : start + n_fft]
        spectrum = np.fft.rfft(frame * window, n=n_fft)
        power[:, frame_index] = (
            spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        ).astype(np.float32)

    # Restrict the analysis band in the mel matrix rather than adding another
    # time-domain filter. This keeps Python and the MCU implementation
    # deterministic while preserving the configured number of mel bands.
    mel_filters = create_mel_filterbank(sr, n_fft, n_mels, f_min=f_min, f_max=f_max)
    mel_power = mel_filters @ power
    return np.maximum(mel_power, 0.0).astype(np.float32)


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    *,
    f_min: float = DEFAULT_F_MIN_HZ,
    f_max: float | None = DEFAULT_F_MAX_HZ,
    log_floor: float = DEFAULT_LOG_FLOOR,
    top_db: float = DEFAULT_TOP_DB,
    reference: str = "sample_max",
) -> np.ndarray:
    """Return a float32 log-mel image with shape ``(n_mels, frames)``.

    ``sample_max`` removes one scalar recording-gain offset and bounds the
    result to [-top_db, 0]. It does not normalize individual mel bands, so the
    CNN can still use the spectral envelope that separates sound classes.
    """
    if log_floor <= 0.0:
        raise ValueError("log_floor must be greater than zero")
    if top_db <= 0.0:
        raise ValueError("top_db must be greater than zero")
    if reference not in {"sample_max", "absolute"}:
        raise ValueError("reference must be 'sample_max' or 'absolute'")

    mel_power = compute_mel_power_spectrogram(
        audio,
        sr,
        n_fft,
        hop_length,
        n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    log_mel_db = 10.0 * np.log10(np.maximum(mel_power, log_floor))
    if reference == "sample_max":
        log_mel_db -= float(np.max(log_mel_db))
    log_mel_db = np.maximum(log_mel_db, -top_db).astype(np.float32)
    if not np.all(np.isfinite(log_mel_db)):
        raise ValueError("log-mel extraction produced NaN or infinite values")
    return log_mel_db


def make_flat_log_mel_extractor(args):
    """Bind CLI log-mel options into the (audio, sr, n_fft, hop, n_mels) extractor
    signature used by the dataset builders and augmenters; returns a flat vector."""

    def extract(audio: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int) -> np.ndarray:
        return extract_log_mel_spectrogram(
            audio,
            sr,
            n_fft,
            hop_length,
            n_mels,
            f_min=args.f_min,
            f_max=args.f_max,
            log_floor=args.log_floor,
            top_db=args.top_db,
            reference=args.log_reference,
        ).reshape(-1)

    return extract


# ---------------------------------------------------------------------------
# preprocess_config.json contract
# ---------------------------------------------------------------------------

def log_mel_feature_config(
    *,
    input_dir: Path,
    target_sr: int,
    duration_seconds: float,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float = DEFAULT_F_MIN_HZ,
    f_max: float | None = DEFAULT_F_MAX_HZ,
    log_floor: float,
    top_db: float,
    reference: str,
    input_channels: str,
    num_samples: int,
) -> Dict:
    sample_count = int(round(duration_seconds * target_sr))
    frame_count = stft_frame_count(sample_count, n_fft, hop_length)
    channel_count = {
        "logmel": 1,
        "logmel-delta": 2,
        "logmel-cmn-delta": 3,
        "cmn-delta": 2,
        "cmvn": 1,
        "ltcmn": 1,
        "cmvn-ltcmn": 2,
    }[input_channels]
    return {
        "input_dir": str(input_dir),
        "target_sr": int(target_sr),
        "segment_seconds": float(duration_seconds),
        "segment_samples": sample_count,
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "window": "periodic_false_hamming_numpy",
        "stft_center": False,
        "spectrum_type": "power",
        "n_mels": int(n_mels),
        "mel_scale": "htk_2595_log10",
        "f_min_hz": float(f_min),
        "f_max_hz": float(target_sr / 2.0 if f_max is None else f_max),
        "frequency_selection": "mel_filterbank_analysis_band",
        "dc_removal": True,
        "rms_normalization": False,
        "log_compression": {
            "formula": "10*log10(max(mel_power, log_floor))",
            "log_floor": float(log_floor),
            "reference": reference,
            "top_db": float(top_db),
        },
        "input_channels": input_channels,
        "cmvn": {
            "enabled": input_channels in {"cmvn", "cmvn-ltcmn"},
            "mean_axis": "per_sample_per_mel_over_time",
            "variance_axis": "per_sample_per_mel_over_time",
            "std_floor_db": DEFAULT_CMVN_STD_FLOOR_DB,
        },
        "input_shape": [int(n_mels), int(frame_count), channel_count],
        "stored_feature_shape": [int(n_mels), int(frame_count)],
        "flattened_feature_dim": int(n_mels * frame_count),
        "feature_type": "log_mel_spectrogram",
        "feature_source": "log_mel_features.extract_log_mel_spectrogram",
        "feature_storage": "row_major_flattened_npy",
        "class_names": DEFAULT_CLASS_NAMES,
        "num_samples": int(num_samples),
    }
