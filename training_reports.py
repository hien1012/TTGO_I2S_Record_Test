#!/usr/bin/env python3
"""Keras-model evaluation reports shared by the log-mel CMVN trainers.

For each split this prints metrics, writes <split>_confusion_matrix.csv and
<split>_classification_report.csv, and optionally exports misclassified samples
under error/<split>/. Multi-root test sets also get one report per source folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from audio_dataset import (
    Dataset,
    export_error_samples,
    write_confusion_matrix_csv,
    write_report_csv,
)


def safe_metric_name(name: str) -> str:
    """Make a folder-derived name safe for metric filenames."""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe.strip("_") or "unknown"


def source_group_name(dataset: Dataset, index: int) -> str:
    """Group samples by the preserved source folder level."""
    rel_path = Path(dataset.relative_paths[index])
    if len(rel_path.parts) > 1:
        return safe_metric_name(rel_path.parts[0])
    source_path = dataset.source_paths[index]
    if source_path is not None:
        return safe_metric_name(source_path.parent.name)
    return "unknown"


def grouped_indices_by_source(dataset: Dataset) -> Dict[str, list[int]]:
    groups: Dict[str, list[int]] = {}
    for idx in range(len(dataset.y)):
        groups.setdefault(source_group_name(dataset, idx), []).append(idx)
    return groups


def predict_probabilities(model, tf, X: np.ndarray, batch_size: int) -> np.ndarray:
    tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    return model.predict(tensor, batch_size=batch_size, verbose=0).astype(np.float32)


def evaluate_split(
    model,
    tf,
    split_name: str,
    dataset: Dataset,
    X_normalized: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    batch_size: int,
    export_errors: bool,
) -> Dict:
    probabilities = predict_probabilities(model, tf, X_normalized, batch_size)
    y_pred = np.argmax(probabilities, axis=1).astype(np.int64)

    labels = list(range(len(class_names)))
    acc = accuracy_score(dataset.y, y_pred)
    bal_acc = balanced_accuracy_score(dataset.y, y_pred)
    cm = confusion_matrix(dataset.y, y_pred, labels=labels)
    report = classification_report(
        dataset.y,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print()
    print("=" * 72)
    print(f"{split_name.upper()} METRICS")
    print("=" * 72)
    print(f"Accuracy         : {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(
        classification_report(
            dataset.y,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(cm)

    write_confusion_matrix_csv(output_dir / f"{split_name}_confusion_matrix.csv", cm, class_names)
    write_report_csv(output_dir / f"{split_name}_classification_report.csv", report)

    error_summary = {"wrong_count": 0, "total_count": int(len(dataset.y))}
    if export_errors:
        error_summary = export_error_samples(
            output_dir=output_dir,
            split_name=split_name,
            dataset=dataset,
            y_pred=y_pred,
            probabilities=probabilities,
            class_names=class_names,
        )
        print(f"Error samples: {error_summary['wrong_count']} saved under {output_dir / 'error' / split_name}")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        **error_summary,
    }


def evaluate_test_by_source_folder(
    model,
    tf,
    test_data: Dataset,
    X_test: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    batch_size: int,
) -> Dict[str, Dict]:
    """Write one test report per source folder when a test set has multiple roots."""
    groups = grouped_indices_by_source(test_data)
    if len(groups) <= 1:
        return {}

    per_folder_metrics: Dict[str, Dict] = {}
    print()
    print("=" * 72)
    print("PER-FOLDER TEST METRICS")
    print("=" * 72)
    for folder_name, indices in sorted(groups.items()):
        idx = np.asarray(indices, dtype=np.int64)
        folder_dataset = test_data.subset(idx)
        split_name = f"test_{folder_name}"
        per_folder_metrics[folder_name] = evaluate_split(
            model=model,
            tf=tf,
            split_name=split_name,
            dataset=folder_dataset,
            X_normalized=X_test[idx],
            class_names=class_names,
            output_dir=output_dir,
            batch_size=batch_size,
            export_errors=True,
        )
    return per_folder_metrics


def save_training_history(history, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)
