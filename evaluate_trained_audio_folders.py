#!/usr/bin/env python3
"""Evaluate trained Log-Mel CMVN audio models on labeled WAV folders.

(The MFCC25 / MFCC13-frame variant of this script is kept in backup/py.)

Each input WAV is labeled from its filename prefix, resampled to the training
sample rate, split into fixed non-overlapping segments, and processed with the
same feature contract recorded in preprocess_config.json.  Metrics are emitted
for all folders together and for each microphone folder separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from audio_dataset import read_wav_mono
from log_mel_features import extract_log_mel_spectrogram, stft_frame_count


SUPPORTED_FEATURE_MODES = ("auto", "logmel-cmvn")
MODEL_SUFFIXES = {".keras", ".h5", ".tflite"}


@dataclass(frozen=True)
class SegmentRecord:
    """One fixed-length model input and its source metadata."""

    feature: np.ndarray
    source_folder: str
    wav_path: Path
    true_label: str
    segment_index: int
    start_seconds: float
    end_seconds: float
    rms: float
    peak: float


class InferenceModel:
    """Small adapter that gives Keras and TFLite one prediction interface."""

    def __init__(self, model_path: Path):
        self.path = model_path
        self.kind = ""
        self.input_shape: tuple[int, ...]
        self.input_dtype: np.dtype
        self.output_dtype: np.dtype
        self.input_quantization = (0.0, 0)
        self.output_quantization = (0.0, 0)
        self._keras_model = None
        self._interpreter = None
        self._input_detail: dict[str, Any] | None = None
        self._output_detail: dict[str, Any] | None = None

        if model_path.suffix.lower() == ".tflite":
            self._load_tflite()
        elif model_path.suffix.lower() in {".keras", ".h5"}:
            self._load_keras()
        else:
            raise ValueError(
                f"Unsupported model extension {model_path.suffix!r}; "
                f"expected one of {sorted(MODEL_SUFFIXES)}"
            )

    @staticmethod
    def _tensorflow():
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required for Keras/TFLite evaluation") from exc
        return tf

    def _load_keras(self) -> None:
        tf = self._tensorflow()
        model = tf.keras.models.load_model(self.path, compile=False)
        if isinstance(model.input_shape, list) or isinstance(model.output_shape, list):
            raise ValueError("Only single-input, single-output models are supported")
        self.kind = "keras"
        self.input_shape = tuple(int(value) for value in model.input_shape[1:])
        self.input_dtype = np.dtype("float32")
        self.output_dtype = np.dtype("float32")
        self._keras_model = model

    def _load_tflite(self) -> None:
        try:
            tf = self._tensorflow()
            interpreter = tf.lite.Interpreter(model_path=str(self.path))
        except RuntimeError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError as exc:
                raise RuntimeError(
                    "TFLite evaluation requires tensorflow or tflite-runtime"
                ) from exc
            interpreter = Interpreter(model_path=str(self.path))

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if len(input_details) != 1 or len(output_details) != 1:
            raise ValueError("Only single-input, single-output TFLite models are supported")

        input_detail = input_details[0]
        output_detail = output_details[0]
        shape = tuple(int(value) for value in input_detail["shape"])
        if not shape or shape[0] != 1:
            raise ValueError(f"TFLite model must have fixed batch size 1, got {shape}")

        self.kind = "tflite"
        self.input_shape = shape[1:]
        self.input_dtype = np.dtype(input_detail["dtype"])
        self.output_dtype = np.dtype(output_detail["dtype"])
        self.input_quantization = tuple(input_detail.get("quantization", (0.0, 0)))
        self.output_quantization = tuple(output_detail.get("quantization", (0.0, 0)))
        self._interpreter = interpreter
        self._input_detail = input_detail
        self._output_detail = output_detail

    @staticmethod
    def _quantize(values: np.ndarray, dtype: np.dtype, scale: float, zero_point: int) -> np.ndarray:
        if scale <= 0.0:
            raise ValueError(f"Invalid quantization scale: {scale}")
        limits = np.iinfo(dtype)
        quantized = np.rint(values / np.float32(scale)) + int(zero_point)
        return np.clip(quantized, limits.min, limits.max).astype(dtype)

    def predict(self, features: np.ndarray, batch_size: int) -> np.ndarray:
        """Return dequantized float32 class probabilities."""
        values = np.asarray(features, dtype=np.float32)
        expected = (len(values), *self.input_shape)
        if values.shape != expected:
            raise ValueError(f"Model input mismatch: expected {expected}, got {values.shape}")

        if self.kind == "keras":
            tf = self._tensorflow()
            output = self._keras_model.predict(
                tf.convert_to_tensor(values, dtype=tf.float32),
                batch_size=batch_size,
                verbose=0,
            )
            return np.asarray(output, dtype=np.float32)

        assert self._interpreter is not None
        assert self._input_detail is not None and self._output_detail is not None
        probabilities: list[np.ndarray] = []
        for sample in values:
            model_input = sample[np.newaxis, ...]
            if np.issubdtype(self.input_dtype, np.integer):
                model_input = self._quantize(
                    model_input,
                    self.input_dtype,
                    float(self.input_quantization[0]),
                    int(self.input_quantization[1]),
                )
            else:
                model_input = model_input.astype(self.input_dtype)

            self._interpreter.set_tensor(self._input_detail["index"], model_input)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_detail["index"])[0]
            if np.issubdtype(self.output_dtype, np.integer):
                output = (
                    output.astype(np.float32) - np.float32(self.output_quantization[1])
                ) * np.float32(self.output_quantization[0])
            probabilities.append(np.asarray(output, dtype=np.float32))
        return np.vstack(probabilities)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def detect_feature_mode(config: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if config.get("input_channels") == "cmvn":
        return "logmel-cmvn"
    raise ValueError(
        "Could not infer feature mode from preprocess_config.json; "
        "pass --feature-mode explicitly"
    )


def resolve_frequency_range(config: dict[str, Any]) -> tuple[float, float]:
    """Return and validate the Mel filterbank limits stored by training."""
    target_sr = int(config["target_sr"])
    nyquist_hz = target_sr / 2.0
    f_min_hz = float(config.get("f_min_hz", 0.0))
    configured_f_max = config.get("f_max_hz")
    f_max_hz = nyquist_hz if configured_f_max is None else float(configured_f_max)
    if not 0.0 <= f_min_hz < f_max_hz <= nyquist_hz:
        raise ValueError(
            "Invalid Mel frequency range in preprocess_config.json: "
            f"expected 0 <= f_min_hz < f_max_hz <= {nyquist_hz:g}, "
            f"got {f_min_hz:g}-{f_max_hz:g} Hz"
        )
    return f_min_hz, f_max_hz


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    model_dir = args.model_dir.resolve() if args.model_dir else None
    config_path = args.config.resolve() if args.config else None
    model_path = args.model.resolve() if args.model else None

    if config_path is None and model_dir is not None:
        config_path = model_dir / "preprocess_config.json"
    if config_path is None:
        raise ValueError("Provide --config or --model-dir")
    if not config_path.is_file():
        raise FileNotFoundError(f"Preprocess config not found: {config_path}")

    if model_path is None:
        if model_dir is None:
            raise ValueError("Provide --model or --model-dir")
        config = load_json(config_path)
        mode = detect_feature_mode(config, args.feature_mode)
        candidates = [
            model_dir / "logmel_cnn_classifier_float32.tflite",
            model_dir / "logmel_cnn_classifier.keras",
            model_dir / "logmel_cnn_classifier.h5",
        ]
        model_path = next((path for path in candidates if path.is_file()), None)
        if model_path is None:
            raise FileNotFoundError(
                f"No default {mode} model found under {model_dir}; pass --model"
            )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path, config_path


def infer_label_from_prefix(path: Path, class_names: Sequence[str]) -> str | None:
    """Infer the class from the beginning of a filename, as used in training."""
    stem = path.stem.lower()
    aliases = {
        "background": ("background", "back", "noise", "bg"),
        "drone": ("drone", "uav"),
        "helicopter": ("helicopter", "heli"),
    }
    for class_name in class_names:
        names = (class_name.lower(), *aliases.get(class_name.lower(), ()))
        if any(stem.startswith(name) for name in names):
            return class_name

    # Preserve the project's b/d/h naming convention without treating every
    # arbitrary word beginning with those letters as a labeled recording.
    initial_map = {"b": "background", "d": "drone", "h": "helicopter"}
    match = re.match(r"^([bdh])(?:[_-]?\d|[_-])", stem)
    if match:
        candidate = initial_map[match.group(1)]
        return candidate if candidate in class_names else None
    return None


def collect_wavs(input_dirs: Sequence[Path]) -> list[tuple[Path, str]]:
    seen: set[str] = set()
    result: list[tuple[Path, str]] = []
    used_source_names: dict[str, int] = {}
    for root in input_dirs:
        source_name = root.name or "root"
        duplicate_index = used_source_names.get(source_name, 0)
        used_source_names[source_name] = duplicate_index + 1
        if duplicate_index:
            source_name = f"{source_name}_{duplicate_index + 1}"
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".wav":
                continue
            key = str(path.resolve()).casefold()
            if key not in seen:
                seen.add(key)
                result.append((path, source_name))
    return result


def fixed_segments(
    audio: np.ndarray,
    segment_samples: int,
    hop_samples: int,
    tail_policy: str,
) -> Iterable[tuple[int, int, np.ndarray]]:
    if len(audio) < segment_samples:
        if tail_policy == "pad" and len(audio):
            yield 0, 0, np.pad(audio, (0, segment_samples - len(audio))).astype(np.float32)
        return

    segment_index = 0
    start = 0
    while start + segment_samples <= len(audio):
        yield segment_index, start, audio[start : start + segment_samples]
        segment_index += 1
        start += hop_samples

    if tail_policy == "pad" and start < len(audio):
        tail = audio[start:]
        yield segment_index, start, np.pad(
            tail, (0, segment_samples - len(tail))
        ).astype(np.float32)


def validate_config(config: dict[str, Any], mode: str, segment_seconds: float) -> None:
    required = ("target_sr", "n_fft", "hop_length", "n_mels", "normalization_mean", "normalization_std")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"preprocess_config.json is missing required keys: {missing}")
    if segment_seconds <= 0.0:
        raise ValueError("--segment-seconds must be greater than zero")

    if config.get("input_channels") != "cmvn" or not config.get("cmvn", {}).get("enabled"):
        raise ValueError("Log-Mel CMVN evaluation requires input_channels='cmvn'")
    resolve_frequency_range(config)
    sample_count = int(round(segment_seconds * int(config["target_sr"])))
    frame_count = stft_frame_count(sample_count, int(config["n_fft"]), int(config["hop_length"]))
    expected_shape = tuple(int(value) for value in config.get("input_shape", []))
    actual_shape = (int(config["n_mels"]), frame_count, 1)
    if expected_shape != actual_shape:
        raise ValueError(
            "Configured segment length does not match the fixed CNN input: "
            f"segment produces {actual_shape}, model config expects {expected_shape}. "
            "Use the training segment_seconds or retrain the CNN for a new width."
        )


def make_feature_extractor(config: dict[str, Any], mode: str):
    target_sr = int(config["target_sr"])
    n_fft = int(config["n_fft"])
    hop_length = int(config["hop_length"])
    n_mels = int(config["n_mels"])
    mean = np.asarray(config["normalization_mean"], dtype=np.float32)
    std = np.asarray(config["normalization_std"], dtype=np.float32)
    if np.any(std <= 1.0e-12):
        raise ValueError("Training normalization contains a zero or near-zero std")

    log_config = config.get("log_compression", {})
    log_floor = float(log_config.get("log_floor", 1.0e-12))
    top_db = float(log_config.get("top_db", 80.0))
    reference = str(log_config.get("reference", "sample_max"))
    f_min_hz, f_max_hz = resolve_frequency_range(config)
    std_floor = np.float32(config.get("cmvn", {}).get("std_floor_db", 2.0))
    if mean.shape != (1,) or std.shape != (1,):
        raise ValueError("Single-channel CMVN normalization must contain one mean/std value")

    def extract(segment: np.ndarray) -> np.ndarray:
        log_mel = extract_log_mel_spectrogram(
            segment,
            target_sr,
            n_fft,
            hop_length,
            n_mels,
            f_min=f_min_hz,
            f_max=f_max_hz,
            log_floor=log_floor,
            top_db=top_db,
            reference=reference,
        )
        cmn = log_mel - np.mean(log_mel, axis=1, keepdims=True, dtype=np.float32)
        cmvn_std = np.std(log_mel, axis=1, keepdims=True, dtype=np.float32)
        cmvn = cmn / np.maximum(cmvn_std, std_floor)
        normalized = (cmvn[..., np.newaxis] - mean) / (std + np.float32(1.0e-6))
        return normalized.astype(np.float32)

    return extract


def extract_records(
    wavs: Sequence[tuple[Path, str]],
    config: dict[str, Any],
    class_names: Sequence[str],
    segment_seconds: float,
    hop_seconds: float,
    tail_policy: str,
    extractor,
) -> tuple[list[SegmentRecord], list[dict[str, Any]]]:
    target_sr = int(config["target_sr"])
    segment_samples = int(round(segment_seconds * target_sr))
    hop_samples = int(round(hop_seconds * target_sr))
    if hop_samples <= 0 or hop_samples > segment_samples:
        raise ValueError("segment hop must be greater than zero and no larger than segment length")

    records: list[SegmentRecord] = []
    skipped: list[dict[str, Any]] = []
    for file_index, (wav_path, source_folder) in enumerate(wavs, start=1):
        true_label = infer_label_from_prefix(wav_path, class_names)
        if true_label is None:
            skipped.append({"file": str(wav_path), "source_folder": source_folder, "reason": "unknown_label_prefix"})
            continue
        try:
            sr, audio = read_wav_mono(wav_path, target_sr)
            segment_count = 0
            for segment_index, start, segment in fixed_segments(
                audio, segment_samples, hop_samples, tail_policy
            ):
                # Match the persisted PCM16 segment used by the training pipeline
                # and the signed samples consumed by the ESP32 frontend.
                pcm16 = np.rint(np.clip(segment, -1.0, 1.0) * 32767.0).astype(np.int16)
                feature_audio = pcm16.astype(np.float32) / np.float32(32767.0)
                feature = extractor(feature_audio)
                if not np.all(np.isfinite(feature)):
                    raise ValueError("feature extractor produced NaN or infinite values")
                records.append(
                    SegmentRecord(
                        feature=feature,
                        source_folder=source_folder,
                        wav_path=wav_path,
                        true_label=true_label,
                        segment_index=segment_index,
                        start_seconds=start / sr,
                        end_seconds=(start + segment_samples) / sr,
                        rms=float(np.sqrt(np.mean(feature_audio * feature_audio) + 1.0e-12)),
                        peak=float(np.max(np.abs(feature_audio))),
                    )
                )
                segment_count += 1
            if segment_count == 0:
                skipped.append(
                    {
                        "file": str(wav_path),
                        "source_folder": source_folder,
                        "reason": f"shorter_than_{segment_seconds:g}s",
                    }
                )
            print(f"[{file_index}/{len(wavs)}] {source_folder}/{wav_path.name}: {segment_count} segments")
        except Exception as exc:
            skipped.append({"file": str(wav_path), "source_folder": source_folder, "reason": str(exc)})
    return records, skipped


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "group"


def save_error_wav_segments(
    prediction_rows: Sequence[dict[str, Any]],
    *,
    target_sr: int,
    segment_seconds: float,
    output_dir: Path,
    limit: int,
) -> int:
    """Persist misclassified fixed-length segments for listening review.

    The evaluator intentionally keeps only features in memory.  Error WAVs are
    reconstructed here from the source file path and segment start time so large
    folder evaluations do not retain every PCM segment in RAM.
    """
    if limit < 0:
        raise ValueError("--error-wav-limit must be zero or positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    segment_samples = int(round(segment_seconds * target_sr))
    saved_count = 0
    audio_cache: dict[Path, np.ndarray] = {}

    for row in prediction_rows:
        if int(row["correct"]):
            continue
        if limit and saved_count >= limit:
            break

        wav_path = Path(str(row["file"]))
        if wav_path not in audio_cache:
            _, audio_cache[wav_path] = read_wav_mono(wav_path, target_sr)
        audio = audio_cache[wav_path]

        start_sample = int(round(float(row["segment_start_s"]) * target_sr))
        segment = audio[start_sample : start_sample + segment_samples]
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment))).astype(np.float32)

        pcm16 = np.rint(np.clip(segment, -1.0, 1.0) * 32767.0).astype(np.int16)
        transition = f"{row['true_label']}_to_{row['predicted_label']}"
        category_dir = output_dir / safe_name(transition)
        category_dir.mkdir(parents=True, exist_ok=True)

        start_ms = int(round(float(row["segment_start_s"]) * 1000.0))
        end_ms = int(round(float(row["segment_end_s"]) * 1000.0))
        filename = (
            f"{saved_count:05d}_"
            f"{safe_name(str(row['source_folder']))}_"
            f"{safe_name(wav_path.stem)}_"
            f"seg{int(row['segment_index']):04d}_"
            f"{start_ms:08d}-{end_ms:08d}ms_"
            f"conf{float(row['confidence']):.3f}.wav"
        )
        saved_path = category_dir / filename
        wavfile.write(saved_path, target_sr, pcm16)
        row["error_wav"] = str(saved_path)
        saved_count += 1

    return saved_count


def save_confusion_matrix(
    output_prefix: Path,
    matrix: np.ndarray,
    class_names: Sequence[str],
    title: str,
) -> None:
    with output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])

    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_prefix.with_suffix(".png"), dpi=180)
    plt.close(figure)


def report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append({"label": label, **values})
        else:
            rows.append({"label": label, "value": values})
    return rows


def calculate_metrics(
    prediction_rows: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    output_dir: Path,
    group_name: str,
) -> dict[str, Any]:
    label_to_id = {name: index for index, name in enumerate(class_names)}
    truth = np.asarray([label_to_id[row["true_label"]] for row in prediction_rows], dtype=np.int64)
    predicted = np.asarray([label_to_id[row["predicted_label"]] for row in prediction_rows], dtype=np.int64)
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(truth, predicted, labels=labels)
    report = classification_report(
        truth,
        predicted,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=labels, average="weighted", zero_division=0
    )
    metrics = {
        "group": group_name,
        "segments": int(len(prediction_rows)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "mean_confidence": float(np.mean([row["confidence"] for row in prediction_rows])),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    prefix = output_dir / f"{safe_name(group_name)}_confusion_matrix"
    save_confusion_matrix(prefix, matrix, class_names, f"Confusion Matrix: {group_name}")
    write_rows(
        output_dir / f"{safe_name(group_name)}_classification_report.csv",
        report_rows(report),
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Log-Mel CMVN models on one or more microphone folders."
        )
    )
    parser.add_argument("--input-dirs", nargs="+", required=True, type=Path, help="WAV folder trees, typically ICS INMP and PDM.")
    parser.add_argument("--model-dir", type=Path, help="Directory containing model and preprocess_config.json.")
    parser.add_argument("--model", type=Path, help="Explicit .keras, .h5, or .tflite model path.")
    parser.add_argument("--config", type=Path, help="Explicit preprocess_config.json path.")
    parser.add_argument("--feature-mode", choices=SUPPORTED_FEATURE_MODES, default="auto")
    parser.add_argument("--segment-seconds", type=float, help="Segment duration; defaults to the training config.")
    parser.add_argument("--segment-hop-seconds", type=float, help="Segment step; defaults to non-overlapping segments.")
    parser.add_argument("--tail-policy", choices=("drop", "pad"), default="drop")
    parser.add_argument("--batch-size", type=int, default=32, help="Keras prediction batch size; TFLite uses batch 1.")
    error_group = parser.add_mutually_exclusive_group()
    error_group.add_argument(
        "--save-error-wavs",
        dest="save_error_wavs",
        action="store_true",
        default=True,
        help="Save misclassified segments as WAV files under the output directory.",
    )
    error_group.add_argument(
        "--no-save-error-wavs",
        dest="save_error_wavs",
        action="store_false",
        help="Do not save misclassified segment WAV files.",
    )
    parser.add_argument(
        "--error-wav-limit",
        type=int,
        default=400,
        help="Maximum number of error WAVs to save; 0 means save all errors.",
    )
    parser.add_argument(
        "--error-wav-dir",
        type=Path,
        default=None,
        help="Directory for error WAVs. Relative paths are resolved under --output-dir.",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("audio_folder_evaluation"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dirs = [path.resolve() for path in args.input_dirs]
    missing = [path for path in input_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Input folders not found: {missing}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    model_path, config_path = resolve_paths(args)
    config = load_json(config_path)
    feature_mode = detect_feature_mode(config, args.feature_mode)
    class_names = [str(value) for value in config.get("class_names", [])]
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError(f"Invalid class_names in config: {class_names}")

    configured_seconds = config.get("segment_seconds", config.get("max_duration"))
    if args.segment_seconds is None:
        if configured_seconds is None:
            raise ValueError("Config has no segment duration; pass --segment-seconds")
        segment_seconds = float(configured_seconds)
    else:
        segment_seconds = float(args.segment_seconds)
    hop_seconds = (
        segment_seconds
        if args.segment_hop_seconds is None
        else float(args.segment_hop_seconds)
    )
    validate_config(config, feature_mode, segment_seconds)

    model = InferenceModel(model_path)
    extractor = make_feature_extractor(config, feature_mode)
    wavs = collect_wavs(input_dirs)
    if not wavs:
        raise ValueError(f"No WAV files found under: {input_dirs}")

    print(f"Feature mode: {feature_mode}")
    print(f"Model: {model_path} ({model.kind}, input={model.input_shape}, dtype={model.input_dtype})")
    print(f"Config: {config_path}")
    print(f"WAV files: {len(wavs)}")
    print(f"Segment: {segment_seconds:g}s, hop={hop_seconds:g}s, tail={args.tail_policy}")
    mel_frequency_range: tuple[float, float] | None = None
    if feature_mode == "logmel-cmvn":
        mel_frequency_range = resolve_frequency_range(config)
        print(
            "Mel frequency range: "
            f"{mel_frequency_range[0]:g}-{mel_frequency_range[1]:g} Hz"
        )

    records, skipped = extract_records(
        wavs,
        config,
        class_names,
        segment_seconds,
        hop_seconds,
        args.tail_policy,
        extractor,
    )
    if not records:
        raise ValueError("No labeled complete segments were available for evaluation")

    feature_shape = records[0].feature.shape
    if any(record.feature.shape != feature_shape for record in records):
        raise ValueError("Feature extractor produced inconsistent shapes")
    features = np.stack([record.feature for record in records]).astype(np.float32)
    if feature_shape != model.input_shape:
        raise ValueError(
            f"Feature/model contract mismatch: extractor={feature_shape}, model={model.input_shape}"
        )
    probabilities = model.predict(features, args.batch_size)
    if probabilities.shape != (len(records), len(class_names)):
        raise ValueError(
            f"Model output mismatch: expected {(len(records), len(class_names))}, got {probabilities.shape}"
        )

    prediction_rows: list[dict[str, Any]] = []
    for record, probability in zip(records, probabilities):
        predicted_id = int(np.argmax(probability))
        ordered = np.sort(probability)
        confidence = float(ordered[-1])
        margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else confidence
        predicted_label = class_names[predicted_id]
        row: dict[str, Any] = {
            "correct": int(predicted_label == record.true_label),
            "source_folder": record.source_folder,
            "file": str(record.wav_path),
            "filename": record.wav_path.name,
            "segment_index": record.segment_index,
            "segment_start_s": record.start_seconds,
            "segment_end_s": record.end_seconds,
            "true_label": record.true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "margin": margin,
            "rms": record.rms,
            "peak": record.peak,
        }
        for class_name, value in zip(class_names, probability):
            row[f"prob_{class_name}"] = float(value)
        prediction_rows.append(row)

    # Put errors first, then high-confidence mistakes, to speed up inspection.
    prediction_rows.sort(
        key=lambda row: (
            int(row["correct"]),
            -float(row["confidence"]),
            str(row["source_folder"]),
            str(row["filename"]),
            int(row["segment_index"]),
        )
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_error_wav_count = 0
    error_wav_dir = None
    if args.save_error_wavs:
        requested_error_dir = args.error_wav_dir or Path("error_wavs")
        error_wav_dir = (
            requested_error_dir
            if requested_error_dir.is_absolute()
            else output_dir / requested_error_dir
        )
        saved_error_wav_count = save_error_wav_segments(
            prediction_rows,
            target_sr=int(config["target_sr"]),
            segment_seconds=segment_seconds,
            output_dir=error_wav_dir,
            limit=args.error_wav_limit,
        )
    write_rows(output_dir / "predictions.csv", prediction_rows)
    write_rows(output_dir / "skipped_files.csv", skipped)

    metric_sets = [
        calculate_metrics(prediction_rows, class_names, output_dir, "overall")
    ]
    for source_folder in sorted({row["source_folder"] for row in prediction_rows}):
        group_rows = [row for row in prediction_rows if row["source_folder"] == source_folder]
        metric_sets.append(
            calculate_metrics(group_rows, class_names, output_dir, source_folder)
        )

    summary_fields = (
        "group",
        "segments",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "mean_confidence",
    )
    write_rows(
        output_dir / "metrics_summary.csv",
        [{key: metrics[key] for key in summary_fields} for metrics in metric_sets],
    )
    summary = {
        "feature_mode": feature_mode,
        "model": str(model_path),
        "model_kind": model.kind,
        "model_input_shape": list(model.input_shape),
        "model_input_dtype": model.input_dtype.name,
        "config": str(config_path),
        "input_dirs": [str(path) for path in input_dirs],
        "segment_seconds": segment_seconds,
        "segment_hop_seconds": hop_seconds,
        "tail_policy": args.tail_policy,
        "f_min_hz": mel_frequency_range[0] if mel_frequency_range else None,
        "f_max_hz": mel_frequency_range[1] if mel_frequency_range else None,
        "wav_file_count": len(wavs),
        "evaluated_segment_count": len(prediction_rows),
        "skipped_file_count": len(skipped),
        "saved_error_wav_count": saved_error_wav_count,
        "error_wav_dir": str(error_wav_dir) if error_wav_dir is not None else None,
        "metrics": metric_sets,
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    overall = metric_sets[0]
    print(
        "Overall: "
        f"accuracy={overall['accuracy']:.4f}, "
        f"balanced_accuracy={overall['balanced_accuracy']:.4f}, "
        f"macro_f1={overall['macro_f1']:.4f}"
    )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
