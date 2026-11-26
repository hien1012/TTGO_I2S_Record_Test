#!/usr/bin/env python3
"""Compare xu_mfcc_features.extract_xu_mfcc() with Xu's C code (mfcc_function-2.c).

44.1 kHz: the C source is compiled unmodified.
16 kHz  : SAMPLE_RATE and the hard-coded mel_points table are patched to the
          16 kHz Slaney points, which is exactly the change the NumPy port makes.

Example:
  python tools/check_xu_mfcc_parity.py --wav-dir raw_audio/new_16kHz/ics
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audio_dataset import collect_wavs, read_wav_mono  # noqa: E402
from xu_mfcc_features import extract_xu_mfcc, slaney_mel_points  # noqa: E402


def patched_source(source: str, sample_rate: int) -> str:
    if sample_rate == 44100:
        return source
    points = ", ".join(f"{value:.6f}" for value in slaney_mel_points(sample_rate))
    source, rate_hits = re.subn(r"#define SAMPLE_RATE 44100", f"#define SAMPLE_RATE {sample_rate}", source)
    source, table_hits = re.subn(
        r"float mel_points\[28\] = \{[^}]*\};",
        f"float mel_points[28] = {{ {points} }};",
        source,
    )
    if rate_hits != 1 or table_hits != 1:
        raise RuntimeError("Could not patch SAMPLE_RATE / mel_points in the C source")
    return source


def run_c(source_path: Path, sample_rate: int, clips: np.ndarray, workdir: Path) -> np.ndarray:
    patched = workdir / f"mfcc_function_{sample_rate}.c"
    patched.write_text(patched_source(source_path.read_text(encoding="utf-8"), sample_rate), encoding="utf-8")
    exe = workdir / f"xu_mfcc_{sample_rate}.exe"
    subprocess.run(
        [
            "gcc", "-O2", "-std=gnu99",
            f'-DXU_MFCC_SOURCE="{patched.as_posix()}"',
            str(REPO_ROOT / "tools" / "xu_mfcc_c_cli.c"),
            "-o", str(exe), "-lm",
        ],
        check=True,
    )
    in_path, out_path = workdir / f"in_{sample_rate}.bin", workdir / f"out_{sample_rate}.bin"
    clips.astype("<f4").tofile(in_path)
    subprocess.run([str(exe), str(in_path), str(out_path), str(len(clips)), str(clips.shape[1])], check=True)
    return np.fromfile(out_path, dtype="<f4").reshape(len(clips), 26)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--c-source", default=str(REPO_ROOT / "mfcc_function-2.c"))
    parser.add_argument("--wav-dir", default=str(REPO_ROOT / "raw_audio" / "new_16kHz" / "ics"))
    parser.add_argument("--clips", type=int, default=12)
    args = parser.parse_args()

    wavs = collect_wavs(Path(args.wav_dir))[:: max(1, len(collect_wavs(Path(args.wav_dir))) // args.clips)][: args.clips]
    worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for sample_rate in (44100, 16000):
            clips = []
            for path in wavs:
                _, audio = read_wav_mono(path, sample_rate)
                clips.append(np.round(audio[: 2 * sample_rate] * 32767.0))
            clips = np.stack(clips).astype(np.float32)
            c_features = run_c(Path(args.c_source), sample_rate, clips, Path(tmp))
            py_features = np.stack([extract_xu_mfcc(clip, sample_rate) for clip in clips])
            abs_diff = np.abs(c_features - py_features)
            rel = abs_diff.max() / (np.abs(c_features).max() + 1e-12)
            worst = max(worst, rel)
            print(
                f"{sample_rate} Hz: {len(clips)} clips, max|C-py|={abs_diff.max():.3e}, "
                f"mean|C-py|={abs_diff.mean():.3e}, max feature |value|={np.abs(c_features).max():.2f}, "
                f"relative={rel:.3e}"
            )
    print("PASS" if worst < 1e-3 else "FAIL")
    sys.exit(0 if worst < 1e-3 else 1)


if __name__ == "__main__":
    main()
