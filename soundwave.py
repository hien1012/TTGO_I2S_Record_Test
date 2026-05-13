from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import wavfile
except ModuleNotFoundError as exc:
    missing = exc.name
    raise SystemExit(
        f"Missing Python package: {missing}\n"
        "Install the required packages with:\n"
        "  python -m pip install numpy scipy matplotlib"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot waveforms and spectrograms for all WAV files in a folder."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder that contains WAV files. Default: current folder.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Folder for generated PNG files. Default: <folder>/soundwave_plots.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search WAV files in subfolders too.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots after saving them.",
    )
    return parser.parse_args()


def find_wav_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(folder.glob(pattern), key=lambda path: path.name.lower())


def mono_signal(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float64)
    return data.astype(np.float64).mean(axis=1)


def full_scale_value(data: np.ndarray) -> float:
    if np.issubdtype(data.dtype, np.integer):
        return float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
    return 1.0


def dbfs(value: float, full_scale: float) -> float:
    if value <= 0:
        return float("-inf")
    return 20.0 * np.log10(value / full_scale)


def wav_stats(signal: np.ndarray, sample_rate: int, full_scale: float) -> dict[str, float]:
    duration = len(signal) / sample_rate
    rms = float(np.sqrt(np.mean(np.square(signal)))) if len(signal) else 0.0
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    dc_offset = float(np.mean(signal)) if len(signal) else 0.0
    return {
        "duration": duration,
        "rms": rms,
        "rms_dbfs": dbfs(rms, full_scale),
        "peak": peak,
        "peak_dbfs": dbfs(peak, full_scale),
        "dc_offset": dc_offset,
    }

#各別wav畫出 waveform / spectrogram
def plot_wav_file(wav_path: Path, output_dir: Path) -> dict[str, float]:
    sample_rate, data = wavfile.read(wav_path)
    signal = mono_signal(data)
    full_scale = full_scale_value(data)
    stats = wav_stats(signal, sample_rate, full_scale)

    time_axis = np.arange(len(signal)) / sample_rate
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    axes[0].plot(time_axis, signal, color="#2563eb", linewidth=0.5)
    axes[0].set_title(
        f"{wav_path.name} | {sample_rate} Hz | "
        f"RMS {stats['rms_dbfs']:.1f} dBFS | Peak {stats['peak_dbfs']:.1f} dBFS"
    )
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].specgram(signal, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="magma")
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Frequency (Hz)")

    output_path = output_dir / f"{wav_path.stem}_soundwave.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    stats["output_path"] = str(output_path)
    return stats


def plot_waveform_overview(wav_files: list[Path], output_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(wav_files),
        1,
        figsize=(12, max(3, 2.2 * len(wav_files))),
        sharex=False,
        constrained_layout=True,
    )
    if len(wav_files) == 1:
        axes = [axes]

    for axis, wav_path in zip(axes, wav_files):
        sample_rate, data = wavfile.read(wav_path)
        signal = mono_signal(data)
        full_scale = full_scale_value(data)
        normalized = signal / full_scale
        time_axis = np.arange(len(normalized)) / sample_rate
        axis.plot(time_axis, normalized, color="#0f766e", linewidth=0.45)
        axis.set_title(wav_path.name)
        axis.set_ylabel("Norm.")
        axis.set_ylim(-1.05, 1.05)
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (seconds)")
    figure.savefig(output_dir / "all_waveforms_overview.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else folder / "soundwave_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = find_wav_files(folder, args.recursive)
    if not wav_files:
        raise SystemExit(f"No WAV files found in: {folder}")

    print(f"Found {len(wav_files)} WAV file(s) in {folder}")
    print(f"Saving plots to {output_dir}")
    print()
    print("file,duration_sec,rms_dbfs,peak_dbfs,dc_offset,plot")

    for wav_path in wav_files:
        stats = plot_wav_file(wav_path, output_dir)
        print(
            f"{wav_path.name},"
            f"{stats['duration']:.3f},"
            f"{stats['rms_dbfs']:.2f},"
            f"{stats['peak_dbfs']:.2f},"
            f"{stats['dc_offset']:.2f},"
            f"{stats['output_path']}"
        )

    plot_waveform_overview(wav_files, output_dir)

    if args.show:
        for image_path in sorted(output_dir.glob("*.png")):
            image = plt.imread(image_path)
            plt.figure(figsize=(12, 7))
            plt.imshow(image)
            plt.axis("off")
            plt.title(image_path.name)
        plt.show()


if __name__ == "__main__":
    main()
