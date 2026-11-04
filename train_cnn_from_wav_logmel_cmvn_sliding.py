#!/usr/bin/env python3
"""Train a compact Log-Mel CMVN CNN from overlapping WAV windows.

This is an independent experimental trainer.  It preserves the established
Keras/float32-TFLite/full-int8-TFLite artifacts and error-WAV reports while
adding leakage-safe sliding windows and training-only device/outdoor domain
augmentation.  Validation and external test audio are never augmented.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import audio_dataset
from device_domain_augmentation import (
    add_device_domain_augmentation_arguments,
    augment_training_dataset_device_domain,
    validate_device_domain_augmentation_args,
)
from log_mel_features import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_LOG_FLOOR,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_TOP_DB,
    log_mel_feature_config,
    make_flat_log_mel_extractor,
    stft_frame_count,
)
from logmel_cnn import (
    build_logmel_cnn,
    class_weight_from_labels,
    export_tflite_models,
    load_tensorflow,
    normalize_and_reshape,
    parse_mel_band_mask,
    recalibrate_batch_normalization,
    tflite_metrics,
)
from sliding_window_logmel_dataset import TestVotingIndex, load_sliding_train_val_test
from training_reports import (
    evaluate_split,
    evaluate_test_by_source_folder,
    grouped_indices_by_source,
    save_training_history,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Log-Mel CMVN CNN from overlapping windows. Multiple train "
            "roots are grouped by content ID before train/val splitting; all "
            "test roots remain external and unaugmented."
        )
    )
    parser.add_argument("--train-dir", nargs="+", required=True)
    parser.add_argument("--test-dir", nargs="+", required=True)
    parser.add_argument("-o", "--output-dir", default="cnn_logmel_cmvn_sliding_output")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument(
        "--model-window-seconds",
        "--segment-seconds",
        dest="model_window_seconds",
        type=int,
        default=1,
        help=(
            "Integer model-input duration used by train, val, and every test "
            "vote window (default: 1 second). --segment-seconds is a legacy alias."
        ),
    )
    parser.add_argument(
        "--train-window-hop-seconds",
        "--segment-hop-seconds",
        dest="train_window_hop_seconds",
        type=float,
        default=None,
        help=(
            "Train/validation window step in seconds. Defaults to the model "
            "window duration for non-overlapping training samples. "
            "--segment-hop-seconds is a legacy alias."
        ),
    )
    parser.add_argument(
        "--test-decision-seconds",
        type=int,
        default=5,
        help="Integer duration of each non-overlapping test voting segment.",
    )
    parser.add_argument(
        "--test-vote-hop-seconds",
        type=float,
        default=None,
        help=(
            "Sliding hop inside each test decision segment. Defaults to the "
            "model window duration so test vote windows do not overlap."
        ),
    )
    parser.add_argument(
        "--test-tail-policy",
        choices=["drop", "pad"],
        default="drop",
        help="Drop or zero-pad the final test decision segment when it is short.",
    )
    parser.add_argument(
        "--vote-no-majority",
        choices=["mean-probability", "uncertain"],
        default="uncertain",
        help=(
            "Fallback when no class receives more than half of the child-window votes."
        ),
    )
    parser.add_argument(
        "--test-output-level",
        choices=["decision", "wav", "both"],
        default="both",
        help="Select hierarchical test reports in addition to window diagnostics.",
    )
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=None)
    parser.add_argument("--log-floor", type=float, default=DEFAULT_LOG_FLOOR)
    parser.add_argument("--top-db", type=float, default=DEFAULT_TOP_DB)
    parser.add_argument(
        "--log-reference", choices=["sample_max", "absolute"], default="sample_max"
    )
    parser.add_argument(
        "--mask-mel-bands",
        nargs="+",
        default=None,
        metavar="INDEX_OR_RANGE",
        help="Zero selected normalized Mel rows, e.g. --mask-mel-bands 18-31.",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.7)
    parser.add_argument("--reduce-lr-patience", type=int, default=20)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-8)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bn-recalibration-passes", type=int, default=3)
    parser.add_argument("--bn-recalibration-momentum", type=float, default=0.9)
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Load a compatible Keras model and skip fitting.",
    )
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--tflite-representative-samples", type=int, default=256)
    add_device_domain_augmentation_arguments(parser)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.train_window_hop_seconds is None:
        args.train_window_hop_seconds = float(args.model_window_seconds)
    if args.test_vote_hop_seconds is None:
        args.test_vote_hop_seconds = float(args.model_window_seconds)
    # Shared augmentation code still consumes the legacy names. Keep these as
    # internal aliases while the public CLI uses the clearer model/train names.
    args.segment_seconds = float(args.model_window_seconds)
    args.segment_hop_seconds = float(args.train_window_hop_seconds)
    if args.target_sr != 16000:
        raise ValueError("This Python/firmware Log-Mel contract requires 16 kHz")
    if args.model_window_seconds <= 0 or args.test_decision_seconds <= 0:
        raise ValueError("model-window and test-decision durations must be positive integers")
    if args.test_decision_seconds < args.model_window_seconds:
        raise ValueError("--test-decision-seconds cannot be shorter than the model window")
    if args.train_window_hop_seconds <= 0.0 or args.test_vote_hop_seconds <= 0.0:
        raise ValueError("train and test sliding hops must be positive")
    if args.train_window_hop_seconds > args.model_window_seconds:
        raise ValueError("--train-window-hop-seconds cannot exceed the model window")
    if args.test_vote_hop_seconds > args.model_window_seconds:
        raise ValueError("--test-vote-hop-seconds cannot exceed the model window")
    train_hop_samples = int(round(args.train_window_hop_seconds * args.target_sr))
    test_hop_samples = int(round(args.test_vote_hop_seconds * args.target_sr))
    if min(train_hop_samples, test_hop_samples) <= 0:
        raise ValueError("Sliding hops must round to at least one audio sample")
    if min(args.n_fft, args.hop_length, args.n_mels) <= 0:
        raise ValueError("n_fft, hop_length, and n_mels must be positive")
    effective_f_max = args.target_sr / 2.0 if args.f_max is None else args.f_max
    if not 0.0 <= args.f_min < effective_f_max <= args.target_sr / 2.0:
        raise ValueError("frequency range must satisfy 0 <= f_min < f_max <= Nyquist")
    if args.log_floor <= 0.0 or args.top_db <= 0.0:
        raise ValueError("log floor and top_db must be positive")
    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val-split must be between zero and one")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0.0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if not 0.0 < args.reduce_lr_factor < 1.0:
        raise ValueError("--reduce-lr-factor must be in (0, 1)")
    if args.reduce_lr_patience < 0 or args.min_learning_rate <= 0.0:
        raise ValueError("ReduceLROnPlateau patience/minimum LR are invalid")
    if args.early_stopping_patience < 0 or args.bn_recalibration_passes < 0:
        raise ValueError("early-stopping and BN recalibration passes cannot be negative")
    if args.tflite_representative_samples <= 0:
        raise ValueError("--tflite-representative-samples must be positive")
    parse_mel_band_mask(args.mask_mel_bands, args.n_mels)
    validate_device_domain_augmentation_args(args)


def model_probabilities(model, tf, X: np.ndarray, batch_size: int) -> np.ndarray:
    return model.predict(
        tf.convert_to_tensor(X, dtype=tf.float32),
        batch_size=batch_size,
        verbose=0,
    ).astype(np.float32)


def tflite_probabilities(tf, model_path: str, X: np.ndarray) -> np.ndarray:
    """Run a TFLite model and dequantize its output for shared vote logic."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    output_shape = tuple(int(value) for value in output_detail["shape"])
    if len(output_shape) != 2 or output_shape[0] != 1:
        raise ValueError(f"Expected one TFLite output vector, got {output_shape}")
    probabilities = np.empty((len(X), output_shape[1]), dtype=np.float32)

    for sample_index in range(len(X)):
        sample = X[sample_index : sample_index + 1]
        if np.issubdtype(input_detail["dtype"], np.integer):
            scale, zero_point = input_detail["quantization"]
            if scale <= 0.0:
                raise ValueError("Invalid TFLite integer input scale")
            limits = np.iinfo(input_detail["dtype"])
            sample = np.clip(
                np.rint(sample / scale + zero_point), limits.min, limits.max
            )
        interpreter.set_tensor(
            input_detail["index"], sample.astype(input_detail["dtype"])
        )
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])[0]
        if np.issubdtype(output_detail["dtype"], np.integer):
            scale, zero_point = output_detail["quantization"]
            if scale <= 0.0:
                raise ValueError("Invalid TFLite integer output scale")
            output = (output.astype(np.float32) - zero_point) * scale
        probabilities[sample_index] = np.asarray(output, dtype=np.float32)
    return probabilities


def resolve_majority_vote(
    probabilities: np.ndarray,
    class_count: int,
    no_majority_policy: str,
) -> Dict[str, object]:
    """Resolve strict-majority votes while preserving uncertainty evidence."""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("Voting requires at least one probability vector")
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    votes = np.bincount(predictions, minlength=class_count).astype(np.int64)
    mean_probability = np.mean(probabilities, axis=0, dtype=np.float64)
    majority_id = int(np.argmax(votes))
    majority_reached = bool(votes[majority_id] > len(predictions) / 2.0)
    if majority_reached:
        final_id = majority_id
        used_probability_fallback = False
    elif no_majority_policy == "mean-probability":
        final_id = int(np.argmax(mean_probability))
        used_probability_fallback = True
    elif no_majority_policy == "uncertain":
        final_id = -1
        used_probability_fallback = False
    else:
        raise ValueError(f"Unsupported no-majority policy: {no_majority_policy}")
    return {
        "prediction_count": int(len(predictions)),
        "votes": votes.tolist(),
        "vote_ratios": (votes / len(predictions)).astype(float).tolist(),
        "mean_probabilities": mean_probability.astype(float).tolist(),
        "majority_reached": majority_reached,
        "majority_label_id": majority_id if majority_reached else None,
        "used_probability_fallback": used_probability_fallback,
        "final_label_id": final_id,
    }


def resolve_decision_vote(
    decision_label_ids: Sequence[int],
    decision_mean_probabilities: np.ndarray,
    class_count: int,
    no_majority_policy: str,
) -> Dict[str, object]:
    """Vote across fixed-duration decisions to produce one result per WAV."""
    if len(decision_label_ids) == 0:
        raise ValueError("Whole-WAV voting requires at least one decision")
    if decision_mean_probabilities.shape != (len(decision_label_ids), class_count):
        raise ValueError("Decision labels and probability vectors do not align")
    accepted_labels = np.asarray(
        [label for label in decision_label_ids if label >= 0], dtype=np.int64
    )
    votes = np.bincount(accepted_labels, minlength=class_count).astype(np.int64)
    mean_probability = np.mean(decision_mean_probabilities, axis=0, dtype=np.float64)
    majority_id = int(np.argmax(votes))
    majority_reached = bool(votes[majority_id] > len(decision_label_ids) / 2.0)
    if majority_reached:
        final_id = majority_id
        used_probability_fallback = False
    elif no_majority_policy == "mean-probability":
        final_id = int(np.argmax(mean_probability))
        used_probability_fallback = True
    elif no_majority_policy == "uncertain":
        final_id = -1
        used_probability_fallback = False
    else:
        raise ValueError(f"Unsupported no-majority policy: {no_majority_policy}")
    decision_count = len(decision_label_ids)
    return {
        "prediction_count": int(decision_count),
        "votes": votes.tolist(),
        "vote_ratios": (votes / decision_count).astype(float).tolist(),
        "mean_probabilities": mean_probability.astype(float).tolist(),
        "majority_reached": majority_reached,
        "majority_label_id": majority_id if majority_reached else None,
        "used_probability_fallback": used_probability_fallback,
        "final_label_id": final_id,
    }


def build_hierarchical_prediction_rows(
    probabilities: np.ndarray,
    voting_index: TestVotingIndex,
    class_names: Sequence[str],
    no_majority_policy: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Aggregate child predictions first by decision segment, then by WAV."""
    if len(probabilities) != sum(
        len(decision["window_indices"]) for decision in voting_index.decisions
    ):
        raise ValueError("Probability count does not match the test voting manifest")

    decision_rows: List[Dict[str, object]] = []
    for decision in voting_index.decisions:
        indices = np.asarray(decision["window_indices"], dtype=np.int64)
        vote = resolve_majority_vote(
            probabilities[indices], len(class_names), no_majority_policy
        )
        final_id = int(vote["final_label_id"])
        row: Dict[str, object] = {
            key: value for key, value in decision.items() if key not in {"windows"}
        }
        row.update(vote)
        row["true_label"] = class_names[int(decision["label_id"])]
        row["predicted_label"] = (
            class_names[final_id] if final_id >= 0 else "uncertain"
        )
        row["correct"] = bool(final_id == int(decision["label_id"]))
        row["low_consensus"] = not bool(vote["majority_reached"])
        for class_id, class_name in enumerate(class_names):
            row[f"votes_{class_name}"] = int(vote["votes"][class_id])
            row[f"vote_ratio_{class_name}"] = float(vote["vote_ratios"][class_id])
            row[f"mean_probability_{class_name}"] = float(
                vote["mean_probabilities"][class_id]
            )
        decision_rows.append(row)

    wav_rows: List[Dict[str, object]] = []
    for wav in voting_index.wavs:
        decisions = [decision_rows[int(index)] for index in wav["decision_indices"]]
        # Each fixed-duration decision contributes one equal-weight probability
        # vector. This prevents a segment with more overlapping windows from
        # receiving extra weight at the whole-WAV level.
        decision_probabilities = np.asarray(
            [row["mean_probabilities"] for row in decisions], dtype=np.float32
        )
        vote = resolve_decision_vote(
            [int(row["final_label_id"]) for row in decisions],
            decision_probabilities,
            len(class_names),
            no_majority_policy,
        )
        final_id = int(vote["final_label_id"])
        row = dict(wav)
        row.update(vote)
        row["decision_count"] = len(decisions)
        row["decision_majority_count"] = int(
            sum(bool(decision["majority_reached"]) for decision in decisions)
        )
        row["true_label"] = class_names[int(wav["label_id"])]
        row["predicted_label"] = (
            class_names[final_id] if final_id >= 0 else "uncertain"
        )
        row["correct"] = bool(final_id == int(wav["label_id"]))
        row["low_consensus"] = not bool(vote["majority_reached"])
        for class_id, class_name in enumerate(class_names):
            row[f"votes_{class_name}"] = int(vote["votes"][class_id])
            row[f"vote_ratio_{class_name}"] = float(vote["vote_ratios"][class_id])
            row[f"mean_probability_{class_name}"] = float(
                vote["mean_probabilities"][class_id]
            )
        wav_rows.append(row)
    return decision_rows, wav_rows


def build_window_prediction_rows(
    probabilities: np.ndarray,
    voting_index: TestVotingIndex,
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    """Expose every child prediction and its ownership for calculation audits."""
    rows: List[Dict[str, object]] = []
    for decision in voting_index.decisions:
        for window in decision["windows"]:
            index = int(window["window_index"])
            prediction = int(np.argmax(probabilities[index]))
            row: Dict[str, object] = {
                "window_index": index,
                "decision_index": int(decision["decision_index"]),
                "wav_index": int(decision["wav_index"]),
                "source_tag": decision["source_tag"],
                "original_path": decision["original_path"],
                "original_relative_path": decision["original_relative_path"],
                "label_id": int(decision["label_id"]),
                "true_label": class_names[int(decision["label_id"])],
                "start_sample": int(window["start_sample"]),
                "end_sample": int(window["end_sample"]),
                "start_seconds": float(window["start_seconds"]),
                "end_seconds": float(window["end_seconds"]),
                "window_audio_path": window["path"],
                "final_label_id": prediction,
                "predicted_label": class_names[prediction],
                "correct": bool(prediction == int(decision["label_id"])),
            }
            for class_id, class_name in enumerate(class_names):
                row[f"probability_{class_name}"] = float(
                    probabilities[index, class_id]
                )
            rows.append(row)
    if len(rows) != len(probabilities):
        raise ValueError("Window prediction rows do not cover every test probability")
    return rows


def summarize_prediction_rows(
    rows: Sequence[Dict[str, object]],
    class_names: Sequence[str],
) -> Dict[str, object]:
    """Summarize accepted and rejected hierarchical predictions."""
    if not rows:
        raise ValueError("Cannot summarize an empty prediction table")
    y_true = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["final_label_id"]) for row in rows], dtype=np.int64)
    labels = list(range(len(class_names)))
    accepted = y_pred >= 0
    matrix_with_uncertain = np.zeros((len(class_names), len(class_names) + 1), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        column = int(prediction) if prediction >= 0 else len(class_names)
        matrix_with_uncertain[int(truth), column] += 1
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    recalls = [float(report[name]["recall"]) for name in class_names]
    return {
        "sample_count": int(len(rows)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "coverage": float(np.mean(accepted)),
        "accepted_count": int(np.sum(accepted)),
        "uncertain_count": int(np.sum(~accepted)),
        "accepted_accuracy": (
            float(np.mean(y_true[accepted] == y_pred[accepted]))
            if np.any(accepted)
            else None
        ),
        "strict_majority_count": int(
            sum(bool(row["majority_reached"]) for row in rows)
        ),
        "low_consensus_count": int(
            sum(bool(row["low_consensus"]) for row in rows)
        ),
        "confusion_matrix": matrix_with_uncertain[:, : len(class_names)].tolist(),
        "confusion_matrix_with_uncertain": matrix_with_uncertain.tolist(),
        "classification_report": report,
    }


def per_source_prediction_metrics(
    rows: Sequence[Dict[str, object]],
    class_names: Sequence[str],
) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_tag"]), []).append(row)
    return {
        source: summarize_prediction_rows(source_rows, class_names)
        for source, source_rows in sorted(grouped.items())
    }


def write_prediction_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write JSON-like list fields as compact strings in a portable CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=True)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def save_prediction_dashboard(
    rows: Sequence[Dict[str, object]],
    class_names: Sequence[str],
    path: Path,
    title: str,
) -> None:
    """Render one confusion matrix and report for aggregated predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize_prediction_rows(rows, class_names)
    matrix = np.asarray(summary["confusion_matrix_with_uncertain"], dtype=np.int64)
    row_total = np.sum(matrix, axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_total,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_total > 0,
    )
    prediction_names = [*class_names, "uncertain"]

    fig = plt.figure(figsize=(16, 6.4))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.35), wspace=0.25)
    matrix_axis = fig.add_subplot(grid[0, 0])
    image = matrix_axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    matrix_axis.set_xticks(
        range(len(prediction_names)), prediction_names, rotation=25, ha="right"
    )
    matrix_axis.set_yticks(range(len(class_names)), class_names)
    matrix_axis.set_xlabel("Predicted label")
    matrix_axis.set_ylabel("True label")
    matrix_axis.set_title("Confusion matrix: count and row percentage")
    for row_index in range(len(class_names)):
        for column_index in range(len(prediction_names)):
            value = normalized[row_index, column_index]
            matrix_axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]}\n{value * 100.0:.1f}%",
                ha="center",
                va="center",
                color="white" if value >= 0.55 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=matrix_axis, label="Fraction of true class")

    report = summary["classification_report"]
    report_axis = fig.add_subplot(grid[0, 1])
    report_axis.axis("off")
    row_names = [*class_names, "macro avg", "weighted avg"]
    cell_text = [
        [
            f"{float(report[name]['precision']):.3f}",
            f"{float(report[name]['recall']):.3f}",
            f"{float(report[name]['f1-score']):.3f}",
            f"{int(report[name]['support'])}",
        ]
        for name in row_names
    ]
    table = report_axis.table(
        cellText=cell_text,
        rowLabels=row_names,
        colLabels=["Precision", "Recall", "F1", "Support"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)
    report_axis.set_title(
        f"Accuracy={summary['accuracy']:.4f}, "
        f"coverage={summary['coverage']:.4f}, samples={summary['sample_count']}",
        pad=18,
    )
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(top=0.84, bottom=0.16)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_decision_errors(
    rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> Dict[str, object]:
    """Copy each wrongly classified decision clip for direct listening."""
    error_root = output_dir / "error" / "test_decision"
    wrong_rows = [row for row in rows if not bool(row["correct"])]
    for row in wrong_rows:
        source = Path(str(row["decision_audio_path"]))
        if not source.exists():
            continue
        destination = (
            error_root
            / str(row["source_tag"])
            / f"true_{row['true_label']}__pred_{row['predicted_label']}__{source.name}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_prediction_rows(error_root / "decision_errors.csv", wrong_rows)
    return {
        "wrong_count": len(wrong_rows),
        "total_count": len(rows),
        "error_directory": str(error_root),
    }


def evaluate_hierarchical_predictions(
    probabilities: np.ndarray,
    voting_index: TestVotingIndex,
    class_names: Sequence[str],
    output_dir: Path,
    title_prefix: str,
    no_majority_policy: str,
    output_level: str,
    export_errors: bool,
) -> Dict[str, object]:
    """Evaluate the exact same vote hierarchy for Keras and TFLite outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    window_rows = build_window_prediction_rows(
        probabilities, voting_index, class_names
    )
    decision_rows, wav_rows = build_hierarchical_prediction_rows(
        probabilities,
        voting_index,
        class_names,
        no_majority_policy,
    )
    result: Dict[str, object] = {
        "vote_rule": "strict_majority_greater_than_half",
        "no_majority_policy": no_majority_policy,
        "window_count": len(window_rows),
    }
    write_prediction_rows(output_dir / "test_window_predictions.csv", window_rows)
    if output_level in {"decision", "both"}:
        write_prediction_rows(output_dir / "test_decision_predictions.csv", decision_rows)
        save_prediction_dashboard(
            decision_rows,
            class_names,
            output_dir / "test_decision_evaluation_summary.png",
            f"{title_prefix}: test decision segments",
        )
        result["decision"] = summarize_prediction_rows(decision_rows, class_names)
        result["decision_by_source"] = per_source_prediction_metrics(
            decision_rows, class_names
        )
        if export_errors:
            result["decision_errors"] = export_decision_errors(
                decision_rows, output_dir
            )
    if output_level in {"wav", "both"}:
        write_prediction_rows(output_dir / "test_wav_predictions.csv", wav_rows)
        save_prediction_dashboard(
            wav_rows,
            class_names,
            output_dir / "test_wav_evaluation_summary.png",
            f"{title_prefix}: complete test WAVs",
        )
        result["wav"] = summarize_prediction_rows(wav_rows, class_names)
        result["wav_by_source"] = per_source_prediction_metrics(wav_rows, class_names)
    return result


def save_evaluation_dashboard(
    model,
    tf,
    dataset: audio_dataset.Dataset,
    X: np.ndarray,
    class_names: Sequence[str],
    batch_size: int,
    path: Path,
    title: str,
) -> None:
    """Put confusion matrix and classification report in one review image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probability = model_probabilities(model, tf, X, batch_size)
    prediction = np.argmax(probability, axis=1).astype(np.int64)
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(dataset.y, prediction, labels=labels)
    row_total = np.sum(matrix, axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_total,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_total > 0,
    )
    report = classification_report(
        dataset.y,
        prediction,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    fig = plt.figure(figsize=(15, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.9, 1.35), wspace=0.24)
    matrix_axis = fig.add_subplot(grid[0, 0])
    image = matrix_axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
    matrix_axis.set_xticks(range(len(class_names)), class_names, rotation=25, ha="right")
    matrix_axis.set_yticks(range(len(class_names)), class_names)
    matrix_axis.set_xlabel("Predicted label")
    matrix_axis.set_ylabel("True label")
    matrix_axis.set_title("Confusion matrix: count and row percentage")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            percentage = normalized[row, column] * 100.0
            color = "white" if normalized[row, column] >= 0.55 else "black"
            matrix_axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{percentage:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    fig.colorbar(image, ax=matrix_axis, label="Fraction of true class")

    report_axis = fig.add_subplot(grid[0, 1])
    report_axis.axis("off")
    row_names = [*class_names, "macro avg", "weighted avg"]
    cell_text = []
    for name in row_names:
        values = report[name]
        cell_text.append(
            [
                f"{float(values['precision']):.3f}",
                f"{float(values['recall']):.3f}",
                f"{float(values['f1-score']):.3f}",
                f"{int(values['support'])}",
            ]
        )
    table = report_axis.table(
        cellText=cell_text,
        rowLabels=row_names,
        colLabels=["Precision", "Recall", "F1", "Support"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)
    accuracy = float(report["accuracy"])
    report_axis.set_title(
        f"Classification report\nAccuracy={accuracy:.4f}, samples={len(dataset.y)}",
        pad=18,
    )
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(top=0.84, bottom=0.15)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_all_dashboards(
    model,
    tf,
    datasets: Dict[str, tuple[audio_dataset.Dataset, np.ndarray]],
    class_names: Sequence[str],
    batch_size: int,
    output_dir: Path,
    title_prefix: str,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for split_name, (dataset, X) in datasets.items():
        path = output_dir / f"{split_name}_evaluation_summary.png"
        save_evaluation_dashboard(
            model,
            tf,
            dataset,
            X,
            class_names,
            batch_size,
            path,
            f"{title_prefix}: {split_name}",
        )
        paths[split_name] = str(path)
    return paths


def evaluate_model(
    model,
    tf,
    train_data: audio_dataset.Dataset,
    val_data: audio_dataset.Dataset,
    test_data: audio_dataset.Dataset,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    batch_size: int,
    title_prefix: str,
    test_voting_index: TestVotingIndex,
    no_majority_policy: str,
    test_output_level: str,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics["train"] = evaluate_split(
        model, tf, "train", train_data, X_train, class_names, output_dir, batch_size, False
    )
    metrics["val"] = evaluate_split(
        model, tf, "val", val_data, X_val, class_names, output_dir, batch_size, True
    )
    metrics["test"] = evaluate_split(
        model, tf, "test", test_data, X_test, class_names, output_dir, batch_size, True
    )
    metrics["test_by_folder"] = evaluate_test_by_source_folder(
        model, tf, test_data, X_test, class_names, output_dir, batch_size
    )
    test_probabilities = model_probabilities(model, tf, X_test, batch_size)
    metrics["hierarchical_test"] = evaluate_hierarchical_predictions(
        test_probabilities,
        test_voting_index,
        class_names,
        output_dir,
        title_prefix,
        no_majority_policy,
        test_output_level,
        True,
    )

    dashboard_sets: Dict[str, tuple[audio_dataset.Dataset, np.ndarray]] = {
        "train": (train_data, X_train),
        "val": (val_data, X_val),
        "test": (test_data, X_test),
    }
    source_groups = grouped_indices_by_source(test_data)
    if len(source_groups) > 1:
        for folder_name, indices in sorted(source_groups.items()):
            index = np.asarray(indices, dtype=np.int64)
            dashboard_sets[f"test_{folder_name}"] = (
                test_data.subset(index),
                X_test[index],
            )
    metrics["evaluation_dashboard_paths"] = save_all_dashboards(
        model,
        tf,
        dashboard_sets,
        class_names,
        batch_size,
        output_dir,
        title_prefix,
    )
    return metrics


def json_safe_args(args: argparse.Namespace) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def save_model_summary(model, path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    masked_mel_bands = parse_mel_band_mask(args.mask_mel_bands, args.n_mels)

    sample_count = int(round(args.target_sr * args.model_window_seconds))
    frame_count = stft_frame_count(sample_count, args.n_fft, args.hop_length)
    expected_dim = args.n_mels * frame_count
    input_shape = (args.n_mels, frame_count, 1)
    extractor = make_flat_log_mel_extractor(args)
    feature_layout = {
        "log_mel_spectrogram": {
            "shape": [args.n_mels, frame_count, 1],
            "flatten_order": "C_row_major_mel_then_time",
            "range": [0, expected_dim],
        }
    }

    def training_augmenter(dataset: audio_dataset.Dataset, runtime_args: argparse.Namespace):
        return augment_training_dataset_device_domain(
            dataset,
            runtime_args,
            output_dir=output_dir,
            feature_extractor=extractor,
            expected_dim=expected_dim,
            feature_contract=f"log_mel_cmvn_{args.n_mels}x{frame_count}",
        )

    train_data, val_data, test_data, configs, test_voting_index = (
        load_sliding_train_val_test(
            args,
            output_dir,
            feature_extractor=extractor,
            expected_dim=expected_dim,
            feature_type="log_mel_spectrogram",
            feature_source="log_mel_features.extract_log_mel_spectrogram",
            feature_layout=feature_layout,
            training_augmenter=training_augmenter,
        )
    )
    feature_dim = audio_dataset.validate_feature_dims(train_data, val_data, test_data)
    if feature_dim != expected_dim:
        raise ValueError(f"Expected {expected_dim} Log-Mel values, got {feature_dim}")
    class_names = audio_dataset.class_names_from_configs(configs)
    X_train, X_val, X_test, normalization_mean, normalization_std = normalize_and_reshape(
        train_data,
        val_data,
        test_data,
        args.n_mels,
        frame_count,
        "cmvn",
        masked_mel_bands,
    )
    y_train = train_data.y.astype(np.int64)
    y_val = val_data.y.astype(np.int64)
    class_weight = class_weight_from_labels(y_train)

    tf, keras = load_tensorflow()
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    best_model_path = output_dir / "logmel_cnn_classifier_best_val_loss.keras"
    best_model = None
    best_epoch = None
    best_val_loss = None
    history = None

    if args.load_model is not None:
        load_path = args.load_model.expanduser().resolve()
        model = keras.models.load_model(load_path, compile=False)
        if tuple(model.input_shape[1:]) != input_shape:
            raise ValueError(
                f"Loaded model input {model.input_shape[1:]} does not match {input_shape}"
            )
        print(f"Loaded Keras model: {load_path}")
    else:
        model = build_logmel_cnn(keras, input_shape, len(class_names))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                best_model_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.min_learning_rate,
                verbose=1,
            ),
        ]
        if args.early_stopping_patience:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.early_stopping_patience,
                    restore_best_weights=False,
                    verbose=1,
                )
            )
        print(f"Model input: {input_shape}; train/val/test={len(X_train)}/{len(X_val)}/{len(X_test)}")
        print(
            f"Model window: {args.model_window_seconds:d}s, "
            f"train hop={args.train_window_hop_seconds:g}s; "
            f"test decision={args.test_decision_seconds:d}s, "
            f"vote hop={args.test_vote_hop_seconds:g}s"
        )
        history = model.fit(
            tf.convert_to_tensor(X_train, dtype=tf.float32),
            tf.convert_to_tensor(y_train, dtype=tf.int64),
            validation_data=(
                tf.convert_to_tensor(X_val, dtype=tf.float32),
                tf.convert_to_tensor(y_val, dtype=tf.int64),
            ),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )
        best_index = int(np.argmin(history.history["val_loss"]))
        best_epoch = best_index + 1
        best_val_loss = float(history.history["val_loss"][best_index])
        best_model = keras.models.load_model(best_model_path, compile=False)

    model.summary()
    save_model_summary(model, output_dir / "model_summary.txt")
    augmentation_metadata = train_data.config.get("training_augmentation", {})
    clean_train_count = int(
        augmentation_metadata.get("clean_train_windows", len(X_train))
    )
    clean_train_count = min(max(clean_train_count, 1), len(X_train))
    bn_recalibration = recalibrate_batch_normalization(
        model,
        tf,
        X_train[:clean_train_count],
        passes=args.bn_recalibration_passes,
        momentum=args.bn_recalibration_momentum,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    model_path = output_dir / "logmel_cnn_classifier.keras"
    model.save(model_path)

    best_bn_recalibration = None
    if best_model is not None:
        best_bn_recalibration = recalibrate_batch_normalization(
            best_model,
            tf,
            X_train[:clean_train_count],
            passes=args.bn_recalibration_passes,
            momentum=args.bn_recalibration_momentum,
            batch_size=args.batch_size,
            random_state=args.random_state,
        )
        best_model.save(best_model_path)

    tflite_info = None
    best_tflite_info = None
    if not args.skip_tflite:
        tflite_info = export_tflite_models(
            tf, model, output_dir, X_train, args.tflite_representative_samples
        )
        tflite_info["float32_test"] = tflite_metrics(
            tf, tflite_info["float32_path"], X_test, test_data
        )
        tflite_info["int8_test"] = tflite_metrics(
            tf, tflite_info["int8_path"], X_test, test_data
        )
        tflite_info["float32_hierarchical_test"] = evaluate_hierarchical_predictions(
            tflite_probabilities(tf, tflite_info["float32_path"], X_test),
            test_voting_index,
            class_names,
            output_dir / "tflite_float32",
            "Final float32 TFLite",
            args.vote_no_majority,
            args.test_output_level,
            False,
        )
        tflite_info["int8_hierarchical_test"] = evaluate_hierarchical_predictions(
            tflite_probabilities(tf, tflite_info["int8_path"], X_test),
            test_voting_index,
            class_names,
            output_dir / "tflite_int8",
            "Final int8 TFLite",
            args.vote_no_majority,
            args.test_output_level,
            False,
        )
        if best_model is not None:
            best_tflite_info = export_tflite_models(
                tf,
                best_model,
                output_dir,
                X_train,
                args.tflite_representative_samples,
                filename_prefix="logmel_cnn_classifier_best_val_loss",
            )
            best_tflite_info["float32_test"] = tflite_metrics(
                tf, best_tflite_info["float32_path"], X_test, test_data
            )
            best_tflite_info["int8_test"] = tflite_metrics(
                tf, best_tflite_info["int8_path"], X_test, test_data
            )
            best_tflite_info["float32_hierarchical_test"] = (
                evaluate_hierarchical_predictions(
                    tflite_probabilities(
                        tf, best_tflite_info["float32_path"], X_test
                    ),
                    test_voting_index,
                    class_names,
                    output_dir / "best_val_loss" / "tflite_float32",
                    "Best validation-loss float32 TFLite",
                    args.vote_no_majority,
                    args.test_output_level,
                    False,
                )
            )
            best_tflite_info["int8_hierarchical_test"] = (
                evaluate_hierarchical_predictions(
                    tflite_probabilities(tf, best_tflite_info["int8_path"], X_test),
                    test_voting_index,
                    class_names,
                    output_dir / "best_val_loss" / "tflite_int8",
                    "Best validation-loss int8 TFLite",
                    args.vote_no_majority,
                    args.test_output_level,
                    False,
                )
            )

    metrics: Dict[str, object] = {
        "model_type": "compact_logmel_cmvn_cnn_sliding_window",
        "model_path": str(model_path),
        "best_val_loss_model_path": str(best_model_path) if best_model is not None else None,
        "class_names": class_names,
        "input_shape": list(input_shape),
        "parameter_count": int(model.count_params()),
        "class_weight": class_weight,
        "sliding_window": {
            "model_window_seconds": int(args.model_window_seconds),
            "model_window_samples": int(args.model_window_seconds * args.target_sr),
            "train_window_hop_seconds": float(args.train_window_hop_seconds),
            "train_window_hop_samples": int(
                round(args.train_window_hop_seconds * args.target_sr)
            ),
            "test_decision_seconds": int(args.test_decision_seconds),
            "test_decision_samples": int(args.test_decision_seconds * args.target_sr),
            "test_vote_hop_seconds": float(args.test_vote_hop_seconds),
            "test_vote_hop_samples": int(
                round(args.test_vote_hop_seconds * args.target_sr)
            ),
            "test_tail_policy": args.test_tail_policy,
            "vote_rule": "strict_majority_greater_than_half",
            "vote_no_majority": args.vote_no_majority,
            "test_output_level": args.test_output_level,
        },
        "training_augmentation": augmentation_metadata,
        "batch_norm_recalibration": bn_recalibration,
        "mel_band_mask": {
            "enabled": bool(masked_mel_bands),
            "indices": masked_mel_bands,
            "fill_value_after_normalization": 0.0,
        },
        "training": {
            "epochs_requested": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "reduce_lr_factor": float(args.reduce_lr_factor),
            "reduce_lr_patience": int(args.reduce_lr_patience),
            "min_learning_rate": float(args.min_learning_rate),
            "early_stopping_patience": int(args.early_stopping_patience),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        "tflite": tflite_info,
    }
    metrics.update(
        evaluate_model(
            model,
            tf,
            train_data,
            val_data,
            test_data,
            X_train,
            X_val,
            X_test,
            class_names,
            output_dir,
            args.batch_size,
            "Final model",
            test_voting_index,
            args.vote_no_majority,
            args.test_output_level,
        )
    )

    if best_model is not None:
        best_output = output_dir / "best_val_loss"
        best_output.mkdir(parents=True, exist_ok=True)
        best_metrics: Dict[str, object] = {
            "selection_monitor": "val_loss",
            "selection_mode": "min",
            "epoch": best_epoch,
            "val_loss_at_checkpoint": best_val_loss,
            "model_path": str(best_model_path),
            "batch_norm_recalibration": best_bn_recalibration,
            "tflite": best_tflite_info,
        }
        best_metrics.update(
            evaluate_model(
                best_model,
                tf,
                train_data,
                val_data,
                test_data,
                X_train,
                X_val,
                X_test,
                class_names,
                best_output,
                args.batch_size,
                "Best validation-loss model",
                test_voting_index,
                args.vote_no_majority,
                args.test_output_level,
            )
        )
        metrics["best_val_loss_model"] = best_metrics

    split_manifest_path = output_dir / "original_file_split.json"
    split_manifest = (
        audio_dataset.read_json(split_manifest_path) if split_manifest_path.exists() else None
    )
    metrics["dataset_split"] = {
        "manifest_path": str(split_manifest_path),
        "manifest": split_manifest,
        "external_test_untouched": True,
        "augmentation_applied_to": "train_only",
    }

    effective_f_max = args.target_sr / 2.0 if args.f_max is None else args.f_max
    preprocess_config = log_mel_feature_config(
        input_dir=Path(args.train_dir[0]).expanduser().resolve(),
        target_sr=args.target_sr,
        duration_seconds=args.model_window_seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        log_floor=args.log_floor,
        top_db=args.top_db,
        reference=args.log_reference,
        input_channels="cmvn",
        num_samples=len(train_data.y) + len(val_data.y) + len(test_data.y),
    )
    preprocess_config.update(
        {
            "segment_method": "train_val_sliding_and_test_hierarchical_vote",
            "model_window_seconds": int(args.model_window_seconds),
            "model_window_samples": int(args.model_window_seconds * args.target_sr),
            "train_window_hop_seconds": float(args.train_window_hop_seconds),
            "train_window_hop_samples": int(
                round(args.train_window_hop_seconds * args.target_sr)
            ),
            "test_voting": metrics["sliding_window"],
            "test_voting_manifest_path": str(
                output_dir / "test_voting_manifest.json"
            ),
            "normalization": "per_derived_channel_scalar_from_training_split_only",
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
            "mel_band_mask": metrics["mel_band_mask"],
            "class_names": class_names,
            "keras_model_path": str(model_path),
            "best_val_loss_keras_model_path": (
                str(best_model_path) if best_model is not None else None
            ),
            "best_val_loss_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "tflite": tflite_info,
            "best_val_loss_tflite": best_tflite_info,
            "training_augmentation": augmentation_metadata,
            "batch_norm_recalibration": bn_recalibration,
            "dataset_split": metrics["dataset_split"],
            "frequency_range_hz": [float(args.f_min), float(effective_f_max)],
            "deployment_warning": (
                "DIR/EQ/tilt/gain/noise/background/reverb are training-only. Firmware "
                "must reproduce only DC removal, Hamming STFT, mel power, Log-Mel, "
                "per-sample CMVN, train scalar normalization, optional Mel masking, "
                "and int8 tensor quantization."
            ),
        }
    )

    audio_dataset.write_json(output_dir / "training_arguments.json", json_safe_args(args))
    audio_dataset.write_json(output_dir / "preprocess_config.json", preprocess_config)
    audio_dataset.write_json(output_dir / "metrics.json", metrics)
    if history is not None:
        save_training_history(history, output_dir)
    print(f"Keras model: {model_path}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    if args.test_output_level in {"decision", "both"}:
        print(
            "Decision-level report: "
            f"{output_dir / 'test_decision_evaluation_summary.png'}"
        )
    if args.test_output_level in {"wav", "both"}:
        print(f"WAV-level report: {output_dir / 'test_wav_evaluation_summary.png'}")


if __name__ == "__main__":
    main()
