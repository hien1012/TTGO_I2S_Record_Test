#!/usr/bin/env python3
"""Apply a fixed gain to every WAV under a folder tree.

Output files retain the input-relative path, filename, sample rate, channel
count, WAV format, and PCM subtype. A CSV report records requested and actual
gain so peak limiting cannot silently change the processing contract.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - dependency failure path
    raise SystemExit("Missing dependency: install it with 'python -m pip install soundfile'") from exc


REPORT_NAME = "gain_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply gain to all WAV files below an input folder while preserving "
            "filenames and relative subdirectories."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Input WAV folder tree.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination folder tree.")
    gain_group = parser.add_mutually_exclusive_group(required=True)
    gain_group.add_argument(
        "--gain-db",
        type=float,
        help="Amplitude gain in dB; for example 6 dB is approximately 2x amplitude.",
    )
    gain_group.add_argument(
        "--gain-linear",
        type=float,
        help="Linear amplitude multiplier; for example 2.0 is approximately +6.02 dB.",
    )
    parser.add_argument(
        "--clip-policy",
        choices=("limit", "clip", "error"),
        default="limit",
        help=(
            "Overflow handling: 'limit' uniformly reduces per-file gain (default), "
            "'clip' hard-clips samples, and 'error' rejects overflowing files."
        ),
    )
    parser.add_argument(
        "--peak-limit-dbfs",
        type=float,
        default=-0.1,
        help="Peak ceiling used by --clip-policy limit (default: -0.1 dBFS).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output WAV files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed WAV.")
    return parser


def rms_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(rms) if rms > 0.0 else float("-inf")


def peak_dbfs(audio: np.ndarray) -> float:
    if audio.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(audio)))
    return 20.0 * math.log10(peak) if peak > 0.0 else float("-inf")


def format_metric(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else "-inf"


def process_audio(
    audio: np.ndarray,
    requested_gain: float,
    clip_policy: str,
    peak_limit: float,
) -> tuple[np.ndarray, float, int, bool]:
    """Return processed audio, actual gain, overflow count, and limit status."""
    if audio.size == 0:
        return audio.copy(), requested_gain, 0, False

    input_peak = float(np.max(np.abs(audio)))
    projected_peak = input_peak * requested_gain
    overflow_count = int(np.count_nonzero(np.abs(audio.astype(np.float64) * requested_gain) > 1.0))
    actual_gain = requested_gain
    gain_limited = False

    if projected_peak > peak_limit and clip_policy == "limit" and input_peak > 0.0:
        actual_gain = peak_limit / input_peak
        gain_limited = True
    elif projected_peak > 1.0 and clip_policy == "error":
        raise ValueError(
            f"requested gain would clip {overflow_count} samples "
            f"(projected peak={projected_peak:.6f})"
        )

    processed = audio.astype(np.float64) * actual_gain
    if clip_policy == "clip":
        processed = np.clip(processed, -1.0, 1.0)
    else:
        processed = np.clip(processed, -peak_limit, peak_limit)
    return processed.astype(np.float32), actual_gain, overflow_count, gain_limited


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "status",
        "relative_path",
        "sample_rate_hz",
        "channels",
        "wav_subtype",
        "requested_gain_db",
        "actual_gain_db",
        "gain_limited",
        "input_peak_dbfs",
        "output_peak_dbfs",
        "input_rms_dbfs",
        "output_rms_dbfs",
        "would_clip_sample_count",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("--output-dir must not be the input folder or a child of it")
    if args.peak_limit_dbfs > 0.0:
        raise ValueError("--peak-limit-dbfs must be <= 0")

    requested_gain = (
        10.0 ** (float(args.gain_db) / 20.0)
        if args.gain_db is not None
        else float(args.gain_linear)
    )
    if not math.isfinite(requested_gain) or requested_gain <= 0.0:
        raise ValueError("gain must resolve to a finite value greater than zero")
    requested_gain_db = 20.0 * math.log10(requested_gain)
    peak_limit = 10.0 ** (float(args.peak_limit_dbfs) / 20.0)

    wav_files = sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    if not wav_files:
        raise ValueError(f"No WAV files found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    written = skipped = failed = limited = 0

    for index, source in enumerate(wav_files, start=1):
        relative = source.relative_to(input_dir)
        destination = output_dir / relative
        row: dict[str, Any] = {
            "status": "",
            "relative_path": relative.as_posix(),
            "sample_rate_hz": "",
            "channels": "",
            "wav_subtype": "",
            "requested_gain_db": f"{requested_gain_db:.6f}",
            "actual_gain_db": "",
            "gain_limited": "",
            "input_peak_dbfs": "",
            "output_peak_dbfs": "",
            "input_rms_dbfs": "",
            "output_rms_dbfs": "",
            "would_clip_sample_count": "",
            "message": "",
        }
        try:
            if destination.exists() and not args.overwrite:
                row.update(status="skipped", message="output exists; use --overwrite")
                skipped += 1
                rows.append(row)
                print(f"[{index}/{len(wav_files)}] SKIP {relative}")
                continue

            info = sf.info(source)
            audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
            processed, actual_gain, overflow_count, gain_limited = process_audio(
                audio,
                requested_gain,
                args.clip_policy,
                peak_limit,
            )

            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            sf.write(
                temporary,
                processed,
                sample_rate,
                format=info.format,
                subtype=info.subtype,
                endian=info.endian,
            )
            os.replace(temporary, destination)

            actual_gain_db = 20.0 * math.log10(actual_gain)
            row.update(
                status="written",
                sample_rate_hz=sample_rate,
                channels=info.channels,
                wav_subtype=info.subtype,
                actual_gain_db=f"{actual_gain_db:.6f}",
                gain_limited=int(gain_limited),
                input_peak_dbfs=format_metric(peak_dbfs(audio)),
                output_peak_dbfs=format_metric(peak_dbfs(processed)),
                input_rms_dbfs=format_metric(rms_dbfs(audio)),
                output_rms_dbfs=format_metric(rms_dbfs(processed)),
                would_clip_sample_count=overflow_count,
            )
            written += 1
            limited += int(gain_limited)
            suffix = f" limited to {actual_gain_db:.2f} dB" if gain_limited else ""
            print(f"[{index}/{len(wav_files)}] OK   {relative} | gain={requested_gain_db:.2f} dB{suffix}")
        except Exception as exc:
            failed += 1
            row.update(status="failed", message=str(exc))
            print(f"[{index}/{len(wav_files)}] FAIL {relative}: {exc}", file=sys.stderr)
            if args.fail_fast:
                rows.append(row)
                break
        rows.append(row)

    report_path = output_dir / REPORT_NAME
    write_report(report_path, rows)
    print(
        f"Done: written={written}, limited={limited}, skipped={skipped}, "
        f"failed={failed}, report={report_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
