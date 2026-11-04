#!/usr/bin/env python3
"""Measure log-mel TFLite prediction sensitivity to waveform gain."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np

from audio_dataset import infer_label, read_wav_mono
from log_mel_features import extract_log_mel_spectrogram, fix_audio_length
from logmel_cnn import build_input_channels


DEFAULT_GAINS_DB = [-24.0, -18.0, -12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 12.0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply ten gains to fixed audio segments and compare TFLite predictions."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "cnn_output/logmel_16kHz/logmel_pdm_16k/"
            "logmel_cnn_classifier_float32.tflite"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("cnn_output/logmel_16kHz/logmel_pdm_16k/preprocess_config.json"),
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        nargs="+",
        default=[Path("raw_audio/new_16kHz/ics"), Path("raw_audio/new_16kHz/inmp")],
        help="Candidate WAV roots used for random stratified selection.",
    )
    parser.add_argument("--wav", type=Path, help="Use this WAV instead of random selection.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=1,
        help="Number of random WAVs. Counts above one are balanced by mic and class.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segment-index", type=int, default=None)
    parser.add_argument("--gains-db", type=float, nargs="+", default=DEFAULT_GAINS_DB)
    parser.add_argument(
        "--float-only",
        action="store_true",
        help="Do not simulate int16 quantization/clipping after applying gain.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("gain_sensitivity_output/logmel_gain_sensitivity.csv"),
    )
    return parser


def collect_wavs(roots: list[Path]) -> list[Path]:
    files = sorted(
        (path for root in roots for path in root.rglob("*") if path.suffix.lower() == ".wav"),
        key=lambda path: str(path).lower(),
    )
    if not files:
        raise FileNotFoundError(f"No WAV files found under: {roots}")
    return files


def select_wavs(files: list[Path], sample_count: int, rng: random.Random) -> list[Path]:
    """Select reproducible samples balanced across microphone and class."""
    if sample_count <= 0:
        raise ValueError("--sample-count must be greater than zero")
    if sample_count == 1:
        return [rng.choice(files)]
    if sample_count >= len(files):
        selected = files.copy()
        rng.shuffle(selected)
        return selected

    groups: dict[tuple[str, str], list[Path]] = {}
    for path in files:
        label = infer_label(path)
        if label is not None:
            groups.setdefault((path.parent.name.lower(), label), []).append(path)
    if not groups:
        raise ValueError("No labeled WAV files are available for stratified selection")

    keys = sorted(groups)
    per_group, remainder = divmod(sample_count, len(keys))
    extra_keys = set(rng.sample(keys, remainder)) if remainder else set()
    selected: list[Path] = []
    for key in keys:
        requested = per_group + int(key in extra_keys)
        if requested > len(groups[key]):
            raise ValueError(f"Requested {requested} samples from {key}, only {len(groups[key])} exist")
        selected.extend(rng.sample(groups[key], requested))
    rng.shuffle(selected)
    return selected


def simulate_pcm16(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply ADC/file-style saturation and int16 quantization."""
    clipped_count = int(np.count_nonzero((audio > 1.0) | (audio < -1.0)))
    pcm = np.clip(np.rint(audio * 32767.0), -32768.0, 32767.0).astype(np.int16)
    return (pcm.astype(np.float32) / np.float32(32767.0)), clipped_count


def rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1.0e-12))


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio)))
    return 20.0 * math.log10(max(peak, 1.0e-12))


def prepare_model_input(audio: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    compression = config["log_compression"]
    logmel = extract_log_mel_spectrogram(
        audio,
        int(config["target_sr"]),
        int(config["n_fft"]),
        int(config["hop_length"]),
        int(config["n_mels"]),
        log_floor=float(compression["log_floor"]),
        top_db=float(compression["top_db"]),
        reference=str(compression["reference"]),
    )
    channels = build_input_channels(logmel[np.newaxis, ...], str(config["input_channels"]))
    mean = np.asarray(config["normalization_mean"], dtype=np.float32).reshape(1, 1, 1, -1)
    std = np.asarray(config["normalization_std"], dtype=np.float32).reshape(1, 1, 1, -1)
    normalized = (channels - mean) / (std + np.float32(1.0e-6))
    return normalized.astype(np.float32), channels.astype(np.float32)


def invoke(interpreter, model_input: np.ndarray) -> np.ndarray:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    tensor = model_input
    if np.issubdtype(input_detail["dtype"], np.integer):
        scale, zero_point = input_detail["quantization"]
        if scale <= 0.0:
            raise ValueError("Invalid integer model input quantization scale")
        limits = np.iinfo(input_detail["dtype"])
        tensor = np.clip(np.rint(tensor / scale + zero_point), limits.min, limits.max)
    interpreter.set_tensor(input_detail["index"], tensor.astype(input_detail["dtype"]))
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])[0]
    if np.issubdtype(output_detail["dtype"], np.integer):
        scale, zero_point = output_detail["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale
    return np.asarray(output, dtype=np.float32)


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    class_names = [str(name) for name in config["class_names"]]

    rng = random.Random(args.seed)
    if args.wav:
        wav_paths = [args.wav.resolve()]
    else:
        wav_paths = select_wavs(collect_wavs(args.wav_dir), args.sample_count, rng)
    segment_samples = int(config["segment_samples"])

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(args.model.resolve()))
    interpreter.allocate_tensors()
    expected_shape = tuple(int(value) for value in interpreter.get_input_details()[0]["shape"])

    rows: list[dict] = []
    for file_index, wav_path in enumerate(wav_paths, start=1):
        wav_path = wav_path.resolve()
        true_label = infer_label(wav_path)
        _, audio = read_wav_mono(wav_path, int(config["target_sr"]))
        full_segment_count = max(1, audio.size // segment_samples)
        segment_index = args.segment_index
        if segment_index is None:
            segment_index = rng.randrange(full_segment_count)
        if not 0 <= segment_index < full_segment_count:
            raise ValueError(
                f"segment-index {segment_index} outside [0, {full_segment_count - 1}] for {wav_path}"
            )
        start = segment_index * segment_samples
        segment = fix_audio_length(audio[start : start + segment_samples], segment_samples)

        # Compute every gain first so features and predictions can use the same 0 dB reference.
        prepared: list[dict] = []
        for gain_db in args.gains_db:
            gained = segment * np.float32(10.0 ** (gain_db / 20.0))
            if args.float_only:
                processed = gained.astype(np.float32)
                clipped_count = 0
            else:
                processed, clipped_count = simulate_pcm16(gained)
            model_input, channels = prepare_model_input(processed, config)
            if model_input.shape != expected_shape:
                raise ValueError(
                    f"Prepared input {model_input.shape} does not match model {expected_shape}"
                )
            probabilities = invoke(interpreter, model_input)
            prepared.append(
                {
                    "gain_db": float(gain_db),
                    "audio": processed,
                    "channels": channels,
                    "clipped_samples": clipped_count,
                    "probabilities": probabilities,
                    "predicted_label": class_names[int(np.argmax(probabilities))],
                }
            )

        reference_item = min(prepared, key=lambda item: abs(item["gain_db"]))
        reference_channels = reference_item["channels"]
        reference_prediction = reference_item["predicted_label"]
        for item in prepared:
            row = {
                "wav_path": str(wav_path),
                "microphone": wav_path.parent.name.lower(),
                "segment_index": segment_index,
                "gain_db": item["gain_db"],
                "rms_dbfs": rms_dbfs(item["audio"]),
                "peak_dbfs": peak_dbfs(item["audio"]),
                "clipped_samples": item["clipped_samples"],
                "feature_mae_vs_0db": float(
                    np.mean(np.abs(item["channels"] - reference_channels))
                ),
                "true_label": true_label or "unknown",
                "baseline_prediction": reference_prediction,
                "predicted_label": item["predicted_label"],
                "prediction_changed_vs_0db": bool(
                    item["predicted_label"] != reference_prediction
                ),
                "correct": bool(true_label == item["predicted_label"]),
            }
            row.update(
                {
                    f"prob_{name}": float(item["probabilities"][i])
                    for i, name in enumerate(class_names)
                }
            )
            rows.append(row)
        print(
            f"[{file_index:02d}/{len(wav_paths):02d}] {wav_path.parent.name}/{wav_path.name} "
            f"segment={segment_index} true={true_label} baseline={reference_prediction}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict] = []
    for gain_db in args.gains_db:
        gain_rows = [row for row in rows if row["gain_db"] == float(gain_db)]
        summary_rows.append(
            {
                "gain_db": float(gain_db),
                "sample_count": len(gain_rows),
                "correct_count": sum(bool(row["correct"]) for row in gain_rows),
                "accuracy": sum(bool(row["correct"]) for row in gain_rows) / len(gain_rows),
                "prediction_flip_count": sum(
                    bool(row["prediction_changed_vs_0db"]) for row in gain_rows
                ),
                "clipped_file_count": sum(int(row["clipped_samples"]) > 0 for row in gain_rows),
                "mean_feature_mae_vs_0db": float(
                    np.mean([float(row["feature_mae_vs_0db"]) for row in gain_rows])
                ),
            }
        )
    summary_path = args.output_csv.with_name(f"{args.output_csv.stem}_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Model: {args.model}")
    print(f"Mode: {'float' if args.float_only else 'pcm16'} | WAV files: {len(wav_paths)}")
    print(f"Details: {args.output_csv} | Summary: {summary_path}")
    print(" gain  accuracy  flips  clipped_files  mean_feature_MAE")
    for row in summary_rows:
        print(
            f"{row['gain_db']:>+5.0f}  {row['accuracy']:>8.3%}  "
            f"{row['prediction_flip_count']:>5}  {row['clipped_file_count']:>13}  "
            f"{row['mean_feature_mae_vs_0db']:.6f}"
        )


if __name__ == "__main__":
    main()
