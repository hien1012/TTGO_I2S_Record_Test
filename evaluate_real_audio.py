#!/usr/bin/env python3
"""
Evaluate a trained CNN audio classifier on real-world WAV folders.

Outputs:
- predictions.csv: one row per WAV with probabilities and decision metadata
- metrics_summary.csv: overall and per-folder metrics
- classification_report.csv: per-class precision/recall/F1
- confusion_matrix.csv / confusion_matrix.png
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from train_cnn_classifier import compute_mel_spectrogram, read_wav_mono


CLASS_NAMES = ["background", "drone", "helicopter"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def infer_label(path: Path) -> str | None:
    """Infer class from filename first, then parent folder names."""
    
    candidates = [path.stem.lower()] + [p.name.lower() for p in path.parents]
    for name in candidates:
        if "background" in name or "bg" in name or "noise" in name or name.startswith("b"):
            return "background"
        if "helicopter" in name or "heli" in name or name.startswith("h"):
            return "helicopter"
        if "drone" in name or "uav" in name or name.startswith("d"):
            return "drone"
    return None


def pad_or_truncate(spec: np.ndarray, target_frames: int) -> np.ndarray:
    padded = np.zeros((spec.shape[0], target_frames), dtype=np.float32)
    copy_frames = min(spec.shape[1], target_frames)
    padded[:, :copy_frames] = spec[:, :copy_frames]
    return padded


def load_config(config_path: Path | None, model: tf.keras.Model, args) -> Dict:
    config = {
        "target_sr": args.target_sr,
        "max_duration": args.max_duration,
        "n_fft": 1024,
        "hop_length": 1024,
        "n_mels": 64,
        "input_shape": None,
        "normalization_mean": None,
        "normalization_std": None,
        "class_names": CLASS_NAMES,
    }
    
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    
    if config["input_shape"] is None:
        model_shape = model.input_shape
        config["input_shape"] = [int(model_shape[1]), int(model_shape[2])]
    
    return config


def collect_wavs(input_dirs: List[Path]) -> List[Tuple[Path, str]]:
    wavs = []
    seen = set()
    for input_dir in input_dirs:
        candidates = sorted(input_dir.rglob("*.wav")) + sorted(input_dir.rglob("*.WAV"))
        for wav in candidates:
            key = str(wav.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            wavs.append((wav, input_dir.name))
    return wavs


def make_prediction_rows(model, wavs: List[Tuple[Path, str]], config: Dict, args) -> List[Dict]:
    rows = []
    specs = []
    metadata = []
    
    target_sr = int(config["target_sr"])
    n_fft = int(config["n_fft"])
    hop_length = int(config["hop_length"])
    n_mels = int(config["n_mels"])
    target_frames = int(config["input_shape"][1])
    
    for wav_path, source_folder in wavs:
        try:
            sr, audio = read_wav_mono(wav_path, target_sr)
            if config.get("max_duration"):
                audio = audio[: int(float(config["max_duration"]) * sr)]
            
            mel_spec = compute_mel_spectrogram(
                audio, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
            )
            specs.append(pad_or_truncate(mel_spec, target_frames))
            true_label = infer_label(wav_path)
            
            # 特徵數據四捨五入到小數點後兩位
            duration_s = round(len(audio) / sr if sr else 0.0, 2)
            rms = round(float(np.sqrt(np.mean(audio * audio) + 1e-12)), 2)
            peak = round(float(np.max(np.abs(audio))) if len(audio) else 0.0, 2)
            
            metadata.append({
                "file": str(wav_path),
                "filename": wav_path.name,
                "source_folder": source_folder,
                "true_label": true_label or "",
                "sample_rate": sr,
                "duration_s": duration_s,
                "mel_frames_before_pad": mel_spec.shape[1],
                "rms": rms,
                "peak": peak,
            })
        except Exception as exc:
            rows.append({
                "file": str(wav_path),
                "filename": wav_path.name,
                "source_folder": source_folder,
                "error": str(exc),
            })
    
    if not specs:
        return rows
    
    X = np.asarray(specs, dtype=np.float32)
    norm_mean = config.get("normalization_mean")
    norm_std = config.get("normalization_std")
    
    if args.norm_mode == "config" and norm_mean is not None and norm_std is not None:
        X = (X - float(norm_mean)) / (float(norm_std) + 1e-6)
    elif args.norm_mode == "dataset" or (args.norm_mode == "config" and norm_mean is None):
        if args.norm_mode == "config":
            print("Warning: preprocess_config.json has no normalization stats; using evaluation dataset stats.")
        X = (X - X.mean()) / (X.std() + 1e-6)
    elif args.norm_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported norm mode: {args.norm_mode}")
    
    probs = model.predict(tf.convert_to_tensor(X, dtype=tf.float32), batch_size=args.batch_size, verbose=0)
    
    for item, prob in zip(metadata, probs):
        pred_id = int(np.argmax(prob))
        sorted_prob = np.sort(prob)
        confidence = float(sorted_prob[-1])
        margin = float(sorted_prob[-1] - sorted_prob[-2]) if len(sorted_prob) > 1 else confidence
        pred_label = CLASS_NAMES[pred_id]
        true_label = item["true_label"]
        
        # 預測結果與機率四捨五入到小數點後兩位
        row = {
            **item,
            "pred_label": pred_label,
            "pred_id": pred_id,
            "confidence": round(confidence, 2),
            "margin": round(margin, 2),
            "correct": int(true_label == pred_label) if true_label else "",
            "prob_background": round(float(prob[LABEL_TO_ID["background"]]), 2),
            "prob_drone": round(float(prob[LABEL_TO_ID["drone"]]), 2),
            "prob_helicopter": round(float(prob[LABEL_TO_ID["helicopter"]]), 2),
            "error": "",
        }
        rows.append(row)
    
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    """通用 CSV 寫入函數 (保留)"""
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def order_prediction_rows(rows: List[Dict]) -> List[Dict]:
    preferred_keys = [
        "correct",
        "filename",
        "source_folder",
        "true_label",
        "pred_label",
        "confidence",
        "margin",
        "prob_background",
        "prob_drone",
        "prob_helicopter",
        "file",
        "duration_s",
        "sample_rate",
        "mel_frames_before_pad",
        "rms",
        "peak",
        "pred_id",
        "error",
    ]
    
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            2 if r.get("error") else int(r.get("correct", 1)),
            str(r.get("true_label", "")),
            str(r.get("pred_label", "")),
            -float(r.get("confidence") or 0.0),
            str(r.get("source_folder", "")),
            str(r.get("filename", "")),
        )
    )
    
    ordered_rows = []
    for row in sorted_rows:
        ordered = {key: row[key] for key in preferred_keys if key in row}
        for key, value in row.items():
            if key not in ordered:
                ordered[key] = value
        ordered_rows.append(ordered)
    
    return ordered_rows


def write_metrics(output_dir: Path, rows: List[Dict]) -> None:
    valid = [r for r in rows if not r.get("error") and r.get("true_label")]
    if not valid:
        print("No labeled predictions found; skipped confusion matrix and classification metrics.")
        return
    
    y_true = [LABEL_TO_ID[r["true_label"]] for r in valid]
    y_pred = [LABEL_TO_ID[r["pred_label"]] for r in valid]
    
    # 1. 儲存 confusion_matrix.csv
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    with open(output_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for label, row in zip(CLASS_NAMES, cm):
            writer.writerow([label, *row.tolist()])
    
    # 繪製並儲存圖片
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    # 2. 儲存 classification_report.csv (內含指標四捨五入)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rounded_values = {k: round(v, 2) for k, v in values.items()}
            report_rows.append({"label": label, **rounded_values})
        else:
            report_rows.append({"label": label, "value": round(values, 2)})
    write_csv(output_dir / "classification_report.csv", report_rows)
    
    # 3. 儲存 metrics_summary.csv (精確度指標四捨五入)
    summary_rows = [{
        "group": "overall",
        "files": len(valid),
        "accuracy": round(accuracy_score(y_true, y_pred), 2),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 2),
        "mean_confidence": round(float(np.mean([r["confidence"] for r in valid])), 2),
        "mean_margin": round(float(np.mean([r["margin"] for r in valid])), 2),
    }]
    
    for source in sorted(set(r["source_folder"] for r in valid)):
        group = [r for r in valid if r["source_folder"] == source]
        gt = [LABEL_TO_ID[r["true_label"]] for r in group]
        gp = [LABEL_TO_ID[r["pred_label"]] for r in group]
        summary_rows.append({
            "group": source,
            "files": len(group),
            "accuracy": round(accuracy_score(gt, gp), 2),
            "balanced_accuracy": round(balanced_accuracy_score(gt, gp), 2),
            "mean_confidence": round(float(np.mean([r["confidence"] for r in group])), 2),
            "mean_margin": round(float(np.mean([r["margin"] for r in group])), 2),
        })
    
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    
    # 終端機同步以兩位小數點格式化輸出
    print(f"Accuracy: {summary_rows[0]['accuracy']:.2f}")
    print(f"Balanced accuracy: {summary_rows[0]['balanced_accuracy']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate audio_classifier.h5 on real-world WAV folders.")
    parser.add_argument("input_dirs", nargs="+", help="Folders containing real-world WAV files.")
    parser.add_argument("--model", default="audio_classifier_output/audio_classifier.h5", help="Path to trained .h5 model.")
    parser.add_argument("--config", default="audio_classifier_output/preprocess_config.json", help="Path to preprocess_config.json.")
    parser.add_argument("-o", "--output-dir", default="real_audio_eval_output", help="Output directory.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Fallback target sample rate if config is missing.")
    parser.add_argument("--max-duration", type=float, default=2.0, help="Fallback max audio duration if config is missing.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--norm-mode", choices=["config", "dataset", "none"], default="config",
                        help="Use training normalization from config, evaluation dataset stats, or no normalization.")
    args = parser.parse_args()
    
    input_dirs = [Path(p).expanduser().resolve() for p in args.input_dirs]
    missing = [str(p) for p in input_dirs if not p.exists()]
    if missing:
        print(f"Error: input directories not found: {missing}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = tf.keras.models.load_model(args.model, compile=False)
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path, model, args)
    
    wavs = collect_wavs(input_dirs)
    print(f"Found {len(wavs)} WAV files")
    print(f"Model input shape: {config['input_shape']}")
    print(f"Normalization mode: {args.norm_mode}")
    
    rows = make_prediction_rows(model, wavs, config, args)
    
    # 寫入儲存 predictions.csv
    write_csv(output_dir / "predictions.csv", order_prediction_rows(rows))
    
    # 寫入儲存其餘的指標 CSV (metrics_summary, classification_report, confusion_matrix)
    write_metrics(output_dir, rows)
    
    print(f"Saved evaluation output to: {output_dir}")


if __name__ == "__main__":
    main()