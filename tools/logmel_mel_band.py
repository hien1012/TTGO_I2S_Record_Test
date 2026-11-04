#!/usr/bin/env python3
"""Mel analysis-band contract shared by the float32 and int8 log-mel converters.

The firmware does not recompute mel filter edges. The converters embed the
exact FFT-bin edges produced by log_mel_features.mel_filter_bin_edges(), so the
C frontend uses the same integers as training for any f_min/f_max.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from log_mel_features import mel_filter_bin_edges  # noqa: E402


SAMPLE_RATE_HZ = 16000
N_FFT = 512
N_MELS = 32


def validate_mel_band(config: dict) -> tuple[float, float, list[int]]:
    """Return (f_min_hz, f_max_hz, bin_edges) after checking firmware support."""
    if "f_min_hz" not in config or "f_max_hz" not in config:
        raise ValueError("config must record f_min_hz and f_max_hz")
    f_min = float(config["f_min_hz"])
    f_max = float(config["f_max_hz"])
    nyquist = SAMPLE_RATE_HZ / 2.0
    if not (math.isfinite(f_min) and math.isfinite(f_max)) or not 0.0 <= f_min < f_max <= nyquist:
        raise ValueError(f"mel band must satisfy 0 <= f_min < f_max <= {nyquist:g}, got {f_min}-{f_max}")
    mel_scale = config.get("mel_scale", "htk_2595_log10")
    if mel_scale != "htk_2595_log10":
        raise ValueError(f"firmware implements htk_2595_log10 mel scale, got {mel_scale!r}")

    # Training zeroes masked rows after normalization; the firmware does not,
    # so a masked model would silently see different inputs on device.
    mask = config.get("mel_band_mask") or {}
    if mask.get("enabled"):
        raise ValueError(
            f"firmware does not apply mel_band_mask (indices {mask.get('indices')}); "
            "retrain without --mask-mel-bands or add mask support to the firmware"
        )

    edges = [
        int(value)
        for value in mel_filter_bin_edges(SAMPLE_RATE_HZ, N_FFT, N_MELS, f_min=f_min, f_max=f_max)
    ]
    # A filter is all-zero when its rising edge has no non-zero weight and it
    # has no falling edge. Python behaves identically, so this is only a notice.
    empty = [
        mel
        for mel in range(N_MELS)
        if edges[mel + 1] - edges[mel] < 2 and edges[mel + 2] <= edges[mel + 1]
    ]
    if empty:
        print(f"Warning: mel filters {empty} cover no FFT bins for {f_min:g}-{f_max:g} Hz")
    return f_min, f_max, edges


def format_hz(value: float) -> str:
    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def mel_band_c_block(f_min: float, f_max: float, edges: list[int]) -> str:
    """Render the preprocess_config*.h section consumed by the C frontend."""
    rows = ",\n".join(
        "    " + ", ".join(str(value) for value in edges[start : start + 12])
        for start in range(0, len(edges), 12)
    )
    return f"""/* Mel analysis band. Filter m spans FFT bins g_audio_mel_bin_edges[m..m+2]. */
#define AUDIO_MODEL_MEL_F_MIN_HZ {format_hz(f_min)}
#define AUDIO_MODEL_MEL_F_MAX_HZ {format_hz(f_max)}
static const uint16_t g_audio_mel_bin_edges[AUDIO_MODEL_INPUT_HEIGHT + 2] = {{
{rows}
}};
"""
