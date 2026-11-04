#!/usr/bin/env python3
"""Measure WAV DC offsets and make negative-offset recordings positive.

For each input WAV, the script measures the mean sample value (DC offset). If
the overall mean is below ``--negative-threshold``, every sample in every
channel is multiplied by -1. This is a polarity inversion; it is intentionally
not DC removal and does not add a bias to the waveform.

Audio is decoded and encoded by libsndfile through ``soundfile``. The output
keeps the source sample rate, channel count, WAV subtype/bit depth, and file
name. Non-audio RIFF metadata chunks are not copied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class WavJob:
    source: Path
    destination: Path


@dataclass(frozen=True)
class DcResult:
    source: Path
    destination: Path
    sample_rate_hz: int
    frames: int
    channels: int
    subtype: str
    dc_before: tuple[float, ...]
    dc_after: tuple[float, ...]
    overall_dc_before: float
    overall_dc_after: float
    inverted: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure WAV DC offsets and polarity-invert files whose overall "
            "DC offset is negative."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more WAV files or directories containing WAV files.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Destination root. Source files are never overwritten.",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=0.0,
        help=(
            "Invert only when overall normalized DC offset is below this value "
            "(default: 0.0). Example: -0.001 ignores tiny negative offsets."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="For directory inputs, inspect only the directory itself.",
    )
    parser.add_argument(
        "--plot-name",
        default="first_wav_dc_polarity_comparison.png",
        help="File name for the first WAV before/after comparison plot.",
    )
    return parser.parse_args()


def find_wavs(directory: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path]
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: str(path).lower(),
    )


def build_jobs(inputs: Sequence[Path], output_dir: Path, recursive: bool) -> list[WavJob]:
    resolved_output = output_dir.resolve()
    jobs: list[WavJob] = []
    seen_sources: set[Path] = set()
    seen_destinations: dict[Path, Path] = {}
    multiple_inputs = len(inputs) > 1

    for raw_input in inputs:
        input_path = raw_input.resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input does not exist: {raw_input}")

        if input_path.is_file():
            if input_path.suffix.lower() != ".wav":
                raise ValueError(f"Input file is not WAV: {raw_input}")
            source_files = [input_path]
            destination_root = resolved_output
            relative_root = input_path.parent
        elif input_path.is_dir():
            source_files = find_wavs(input_path, recursive)
            destination_root = resolved_output / input_path.name if multiple_inputs else resolved_output
            relative_root = input_path
        else:
            raise ValueError(f"Unsupported input path: {raw_input}")

        for source in source_files:
            source = source.resolve()
            if source in seen_sources:
                continue
            relative = source.relative_to(relative_root)
            destination = (destination_root / relative).resolve()
            if destination == source:
                raise ValueError(
                    f"Output would overwrite source WAV: {source}. Choose another --output-dir."
                )
            if destination in seen_destinations:
                other = seen_destinations[destination]
                raise ValueError(
                    f"Output collision: {source} and {other} both map to {destination}"
                )
            seen_sources.add(source)
            seen_destinations[destination] = source
            jobs.append(WavJob(source=source, destination=destination))

    jobs.sort(key=lambda job: str(job.source).lower())
    if not jobs:
        raise ValueError("No WAV files were found in the supplied inputs")
    return jobs


def measure_dc(audio: np.ndarray) -> tuple[tuple[float, ...], float]:
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise ValueError(f"Expected non-empty audio shaped (frames, channels), got {audio.shape}")
    per_channel = np.mean(audio, axis=0, dtype=np.float64)
    overall = float(np.mean(audio, dtype=np.float64))
    return tuple(float(value) for value in per_channel), overall


def process_wav(job: WavJob, negative_threshold: float) -> tuple[DcResult, np.ndarray, np.ndarray]:
    info = sf.info(str(job.source))
    if info.format != "WAV":
        raise ValueError(f"File extension is WAV but container is {info.format}: {job.source}")

    # float64 prevents the signed-integer minimum from overflowing during -1x.
    before, sample_rate = sf.read(str(job.source), dtype="float64", always_2d=True)
    dc_before, overall_before = measure_dc(before)
    inverted = overall_before < negative_threshold
    after = -before if inverted else before.copy()
    dc_after, overall_after = measure_dc(after)

    job.destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(job.destination),
        after,
        sample_rate,
        format=info.format,
        subtype=info.subtype,
        endian=info.endian,
    )

    result = DcResult(
        source=job.source,
        destination=job.destination,
        sample_rate_hz=int(sample_rate),
        frames=int(before.shape[0]),
        channels=int(before.shape[1]),
        subtype=info.subtype,
        dc_before=dc_before,
        dc_after=dc_after,
        overall_dc_before=overall_before,
        overall_dc_after=overall_after,
        inverted=inverted,
    )
    return result, before, after


def plot_first_wav(
    result: DcResult,
    before: np.ndarray,
    after: np.ndarray,
    output_path: Path,
) -> None:
    # Limit plotted points for large WAVs without changing the processed audio.
    max_plot_points = 200_000
    stride = max(1, int(np.ceil(before.shape[0] / max_plot_points)))
    indices = np.arange(0, before.shape[0], stride)
    time_seconds = indices / float(result.sample_rate_hz)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
    for channel in range(before.shape[1]):
        label = f"Channel {channel + 1}"
        axes[0].plot(time_seconds, before[indices, channel], linewidth=0.65, label=label)
        axes[1].plot(time_seconds, after[indices, channel], linewidth=0.65, label=label)

    axes[0].axhline(result.overall_dc_before, color="crimson", linestyle="--", linewidth=1.1)
    axes[1].axhline(result.overall_dc_after, color="crimson", linestyle="--", linewidth=1.1)
    axes[0].set_title(
        f"Before: overall DC offset = {result.overall_dc_before:+.8f} normalized FS"
    )
    axes[1].set_title(
        f"After: overall DC offset = {result.overall_dc_after:+.8f} normalized FS; "
        f"polarity inverted = {result.inverted}"
    )
    for axis in axes:
        axis.set_ylabel("Amplitude (normalized FS)")
        axis.grid(True, alpha=0.25)
        if before.shape[1] > 1:
            axis.legend(loc="upper right")
    axes[1].set_xlabel("Time (s)")
    fig.suptitle(f"DC polarity comparison: {result.source.name}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_results_csv(results: Sequence[DcResult], output_path: Path) -> None:
    max_channels = max(result.channels for result in results)
    fieldnames = [
        "source_path",
        "output_path",
        "sample_rate_hz",
        "frames",
        "duration_seconds",
        "channels",
        "subtype",
        "overall_dc_before_normalized_fs",
        "overall_dc_after_normalized_fs",
        "polarity_inverted",
    ]
    for channel in range(max_channels):
        fieldnames.extend(
            [
                f"channel_{channel + 1}_dc_before_normalized_fs",
                f"channel_{channel + 1}_dc_after_normalized_fs",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: dict[str, object] = {
                "source_path": str(result.source),
                "output_path": str(result.destination),
                "sample_rate_hz": result.sample_rate_hz,
                "frames": result.frames,
                "duration_seconds": result.frames / float(result.sample_rate_hz),
                "channels": result.channels,
                "subtype": result.subtype,
                "overall_dc_before_normalized_fs": result.overall_dc_before,
                "overall_dc_after_normalized_fs": result.overall_dc_after,
                "polarity_inverted": result.inverted,
            }
            for channel in range(max_channels):
                if channel < result.channels:
                    row[f"channel_{channel + 1}_dc_before_normalized_fs"] = result.dc_before[channel]
                    row[f"channel_{channel + 1}_dc_after_normalized_fs"] = result.dc_after[channel]
                else:
                    row[f"channel_{channel + 1}_dc_before_normalized_fs"] = ""
                    row[f"channel_{channel + 1}_dc_after_normalized_fs"] = ""
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.negative_threshold > 0.0:
        raise ValueError("--negative-threshold must be zero or negative")

    output_dir = args.output_dir.resolve()
    jobs = build_jobs(args.inputs, output_dir, recursive=not args.no_recursive)
    results: list[DcResult] = []
    first_plot_data: tuple[DcResult, np.ndarray, np.ndarray] | None = None

    for index, job in enumerate(jobs, start=1):
        result, before, after = process_wav(job, args.negative_threshold)
        results.append(result)
        if first_plot_data is None:
            first_plot_data = (result, before, after)
        action = "INVERTED" if result.inverted else "UNCHANGED"
        print(
            f"[{index}/{len(jobs)}] {action}: {job.source.name} | "
            f"DC {result.overall_dc_before:+.8f} -> {result.overall_dc_after:+.8f} | "
            f"{job.destination}"
        )

    csv_path = output_dir / "dc_offset_results.csv"
    write_results_csv(results, csv_path)
    assert first_plot_data is not None
    plot_path = output_dir / args.plot_name
    plot_first_wav(*first_plot_data, plot_path)

    inverted_count = sum(result.inverted for result in results)
    print(f"Processed WAV files: {len(results)}")
    print(f"Polarity-inverted files: {inverted_count}")
    print(f"DC statistics CSV: {csv_path}")
    print(f"First-WAV comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
