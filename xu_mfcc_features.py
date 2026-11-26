#!/usr/bin/env python3
"""NumPy port of Xu's MFCC front end (SENPAI_Code mfcc_function-2.c).

Reference: extract_mfcc_from_array() in mfcc_function-2.c, called from
audio_dnn_drone_detection.ipynb.  The C code is fixed at 44.1 kHz; this port
takes the sample rate as a parameter so the same front end can run on the
16 kHz recordings.

Per 2 s clip (int16-scale float samples):
  x / 32768 -> 512-sample frames, hop 256, symmetric Hamming window
  -> |FFT|^2 -> 26 triangular Slaney-mel filters (bin-centre frequencies,
     each filter normalised to unit sum, edges = librosa.mel_frequencies(28, 0, sr/2))
  -> 10*log10(mel / mean(mel) + 1e-10)   (log relative to the frame's mean mel power)
  -> orthonormal DCT-II, keep 13 coefficients
  -> 13 means + 13 population stds over frames  = 26 features.

The hard-coded 44.1 kHz mel_points table in the C file equals
librosa.mel_frequencies(28, fmin=0, fmax=22050, htk=False) (max |diff| 4e-7 Hz),
so the 16 kHz port uses the same formula with fmax = 8000 Hz.

preprocess_esp32_audio() ports the notebook's ESP-recording preprocessing
(pre-emphasis 0.90, 5th-order Butterworth band-pass, clip). The 13.5 kHz high
cut is above the 8 kHz Nyquist of 16 kHz audio, so the caller chooses a high
cut below Nyquist.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.fft
import scipy.signal

FRAME_SIZE = 512
STEP_SIZE = 256
NUM_FILTERS = 26
NUM_MFCC = 13
FEATURE_DIM = 2 * NUM_MFCC
INT16_SCALE = 32768.0


def slaney_mel_points(sample_rate: int, num_filters: int = NUM_FILTERS) -> np.ndarray:
    """Filter edge frequencies in Hz (matches the C mel_points table at 44.1 kHz)."""
    import librosa

    return librosa.mel_frequencies(
        num_filters + 2, fmin=0.0, fmax=sample_rate / 2.0, htk=False
    ).astype(np.float64)


@lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int) -> np.ndarray:
    """create_mel_filterbank() + normalize_filterbank(); shape (26, 257)."""
    points = slaney_mel_points(sample_rate)
    fft_bins = FRAME_SIZE // 2 + 1
    freq = (np.arange(fft_bins, dtype=np.float64) + 0.5) / FRAME_SIZE * sample_rate
    bank = np.zeros((NUM_FILTERS, fft_bins), dtype=np.float64)
    for i in range(1, NUM_FILTERS + 1):
        left, center, right = points[i - 1], points[i], points[i + 1]
        rising = (freq >= left) & (freq <= center)
        falling = (freq > center) & (freq <= right)
        bank[i - 1, rising] = (freq[rising] - left) / (center - left)
        bank[i - 1, falling] = (right - freq[falling]) / (right - center)
    sums = bank.sum(axis=1, keepdims=True)
    return np.where(sums > 1e-6, bank / np.where(sums > 1e-6, sums, 1.0), bank)


@lru_cache(maxsize=1)
def hamming_symmetric() -> np.ndarray:
    n = np.arange(FRAME_SIZE, dtype=np.float64)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (FRAME_SIZE - 1))


def extract_xu_mfcc(int16_scale_audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return the 26-dim [13 MFCC means, 13 MFCC stds] vector for one clip."""
    data = np.asarray(int16_scale_audio, dtype=np.float64) / INT16_SCALE
    if data.size < FRAME_SIZE:
        raise ValueError(f"clip needs at least {FRAME_SIZE} samples, got {data.size}")
    num_frames = (data.size - FRAME_SIZE) // STEP_SIZE + 1
    frames = np.lib.stride_tricks.sliding_window_view(data, FRAME_SIZE)[::STEP_SIZE][:num_frames]
    spectrum = np.fft.rfft(frames * hamming_symmetric(), axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    mel = power @ mel_filterbank(sample_rate).T
    mean_mel = mel.mean(axis=1, keepdims=True)
    log_mel = 10.0 * np.log10(mel / (mean_mel + 1e-10) + 1e-10)
    mfcc = scipy.fft.dct(log_mel, type=2, norm="ortho", axis=1)[:, :NUM_MFCC]
    mean = mfcc.mean(axis=0)
    std = np.sqrt(np.maximum((mfcc ** 2).mean(axis=0) - mean ** 2, 0.0))
    return np.concatenate([mean, std]).astype(np.float32)


def float32_to_int16(y: np.ndarray) -> np.ndarray:
    """Notebook float32_to_int16(): clip to [-1, 1] then truncate *32767."""
    return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)


def preprocess_esp32_audio(
    signal: np.ndarray,
    sample_rate: int,
    *,
    low_hz: float = 50.0,
    high_hz: float = 13500.0,
    alpha: float = 0.90,
    order: int = 5,
) -> np.ndarray:
    """Notebook preprocess_esp32_audio() on [-1, 1] float audio."""
    signal = np.asarray(signal, dtype=np.float64)
    signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    nyquist = 0.5 * sample_rate
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError(f"band-pass {low_hz}-{high_hz} Hz must lie inside (0, {nyquist})")
    b, a = scipy.signal.butter(N=order, Wn=[low_hz / nyquist, high_hz / nyquist], btype="band")
    signal = scipy.signal.lfilter(b, a, signal)
    return np.clip(signal, -1.0, 1.0)
