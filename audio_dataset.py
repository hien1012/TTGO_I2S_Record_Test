#!/usr/bin/env python3
"""Labeled WAV input, the in-memory Dataset, and on-disk dataset/report files.

Used by every log-mel CMVN trainer:
  - read_wav_mono / infer_label: WAV decoding and filename-derived labels.
  - Dataset: features + labels + source paths kept aligned through splits.
  - save_npy_dataset: features.npy / labels.npy / filenames.json / ... per split.
  - write_*_csv, export_error_samples: evaluation reports and misclassified WAVs.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


DEFAULT_CLASS_NAMES = ["background", "drone", "helicopter"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(DEFAULT_CLASS_NAMES)}


# ---------------------------------------------------------------------------
# WAV input and labels
# ---------------------------------------------------------------------------

def read_wav_mono(path: Path, target_sr: int = None) -> Tuple[int, np.ndarray]:
    """Read WAV file and convert to mono."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_abs = float(np.iinfo(data.dtype).max)
        x = data.astype(np.float32) / max_abs
    else:
        x = data.astype(np.float32)
    x = np.nan_to_num(x, copy=False)

    if target_sr and target_sr != sr:
        gcd = math.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr

    return sr, x


def infer_label(path: Path) -> Optional[str]:
    """Infer label from filename."""
    name = path.stem.lower()
    if "background" in name or "back" in name or "noise" in name or name.startswith("b"):
        return "background"
    if "helicopter" in name or "heli" in name or name.startswith("h"):
        return "helicopter"
    if "drone" in name or "uav" in name or name.startswith("d"):
        return "drone"
    return None


def collect_wavs(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    relative_paths: List[str]
    source_paths: List[Optional[Path]]
    config: Dict

    def subset(self, indices: Sequence[int]) -> "Dataset":
        idx = np.asarray(indices, dtype=np.int64)
        return Dataset(
            X=self.X[idx],
            y=self.y[idx],
            relative_paths=[self.relative_paths[int(i)] for i in idx],
            source_paths=[self.source_paths[int(i)] for i in idx],
            config=self.config,
        )


def class_names_from_configs(configs: Sequence[Dict]) -> List[str]:
    for config in configs:
        names = config.get("class_names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return DEFAULT_CLASS_NAMES


def class_label(label_id: int, class_names: Sequence[str]) -> str:
    return class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)


def validate_feature_dims(train_data: Dataset, val_data: Dataset, test_data: Dataset) -> int:
    feature_dim = int(train_data.X.shape[1])
    for split_name, dataset in (("val", val_data), ("test", test_data)):
        if dataset.X.shape[1] != feature_dim:
            raise ValueError(f"{split_name} feature dim mismatch: {dataset.X.shape[1]} vs train {feature_dim}")
    return feature_dim


# ---------------------------------------------------------------------------
# Paths, JSON, CSV, and NPY persistence
# ---------------------------------------------------------------------------

def safe_relative_path(path_text: str) -> Path:
    """
    Keep source folder hierarchy for exported error audio without allowing a
    stored absolute path or '..' component to escape the output directory.
    """
    path = Path(path_text)
    parts = []
    for part in path.parts:
        if part in ("", ".", ".."):
            continue
        if part == path.anchor or part.endswith(":"):
            continue
        parts.append(part)
    return Path(*parts) if parts else Path("sample.wav")


def make_source_relative_path(input_dir: Path, wav_file: Path) -> str:
    # Include the input folder name so explicit roots such as "ics" or "inmp"
    # remain visible in exported error samples.
    rel_path = wav_file.relative_to(input_dir)
    return (Path(input_dir.name) / rel_path).as_posix()


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv_rows(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_npy_dataset(output_dir: Path, dataset: Dataset, skipped_rows: Optional[List[Dict]] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", dataset.X.astype(np.float32))
    np.save(output_dir / "labels.npy", dataset.y.astype(np.int64))
    write_json(output_dir / "filenames.json", dataset.relative_paths)
    write_json(
        output_dir / "source_paths.json",
        [str(path) if path is not None else "" for path in dataset.source_paths],
    )
    write_json(output_dir / "preprocess_config.json", dataset.config)
    if skipped_rows:
        write_csv_rows(output_dir / "skipped_files.csv", skipped_rows)


# ---------------------------------------------------------------------------
# Evaluation report files
# ---------------------------------------------------------------------------

def write_confusion_matrix_csv(path: Path, cm: np.ndarray, class_names: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for idx, row in enumerate(cm):
            writer.writerow([class_label(idx, class_names), *row.tolist()])


def write_report_csv(path: Path, report: Dict) -> None:
    rows = []
    for name, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "label": name,
                    "precision": values.get("precision", ""),
                    "recall": values.get("recall", ""),
                    "f1_score": values.get("f1-score", ""),
                    "support": values.get("support", ""),
                }
            )
        else:
            rows.append({"label": name, "precision": "", "recall": "", "f1_score": values, "support": ""})
    write_csv_rows(path, rows, ["label", "precision", "recall", "f1_score", "support"])


def copy_error_audio(
    error_audio_root: Path,
    rank: int,
    sample_rel_path: str,
    source_path: Optional[Path],
    true_name: str,
    pred_name: str,
) -> str:
    if source_path is None or not source_path.exists():
        return ""

    safe_rel = safe_relative_path(sample_rel_path)
    dst_dir = error_audio_root / safe_rel.parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_name = f"{rank:04d}_true-{true_name}_pred-{pred_name}_{safe_rel.name}"
    dst_path = dst_dir / dst_name
    shutil.copy2(source_path, dst_path)
    return str(dst_path)


def export_error_samples(
    output_dir: Path,
    split_name: str,
    dataset: Dataset,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, int]:
    error_dir = output_dir / "error" / split_name
    audio_root = error_dir / "audio"
    error_dir.mkdir(parents=True, exist_ok=True)

    wrong_indices = np.flatnonzero(dataset.y != y_pred)
    if len(wrong_indices):
        confidence_order = np.argsort(-probabilities[wrong_indices, y_pred[wrong_indices]])
        wrong_indices = wrong_indices[confidence_order]

    rows = []
    for rank, idx in enumerate(wrong_indices, start=1):
        true_id = int(dataset.y[idx])
        pred_id = int(y_pred[idx])
        true_name = class_label(true_id, class_names)
        pred_name = class_label(pred_id, class_names)
        confidence = float(probabilities[idx, pred_id]) if pred_id < probabilities.shape[1] else 0.0
        copied_audio_path = copy_error_audio(
            error_audio_root=audio_root,
            rank=rank,
            sample_rel_path=dataset.relative_paths[idx],
            source_path=dataset.source_paths[idx],
            true_name=true_name,
            pred_name=pred_name,
        )

        row = {
            "rank": rank,
            "sample_index": int(idx),
            "source_relative_path": dataset.relative_paths[idx],
            "source_path": str(dataset.source_paths[idx]) if dataset.source_paths[idx] is not None else "",
            "true_label": true_name,
            "pred_label": pred_name,
            "confidence": confidence,
        }
        for class_idx, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(probabilities[idx, class_idx])
        row["copied_audio_path"] = copied_audio_path
        rows.append(row)

    fieldnames = [
        "rank",
        "sample_index",
        "source_relative_path",
        "source_path",
        "true_label",
        "pred_label",
        "confidence",
        *[f"prob_{name}" for name in class_names],
        "copied_audio_path",
    ]
    write_csv_rows(error_dir / f"{split_name}_error_samples.csv", rows, fieldnames)
    np.save(error_dir / f"{split_name}_error_features.npy", dataset.X[wrong_indices])
    np.save(error_dir / f"{split_name}_error_true_labels.npy", dataset.y[wrong_indices])
    np.save(error_dir / f"{split_name}_error_pred_labels.npy", y_pred[wrong_indices])
    write_json(error_dir / f"{split_name}_error_filenames.json", [dataset.relative_paths[i] for i in wrong_indices])

    return {"wrong_count": int(len(wrong_indices)), "total_count": int(len(dataset.y))}
