#!/usr/bin/env python3
"""
Inference script: Load trained MFCC+MLP model and predict on numpy features.

Usage:
    # 推理整个测试集
    python inference_mfcc_mlp.py \
      --features data_npy/test/features.npy \
      --labels data_npy/test/labels.npy \
      --model mfcc_mlp_output/mfcc_mlp_classifier.h5 \
      --config mfcc_mlp_output/preprocess_config.json
    
    # 只推理特征 (无标签)
    python inference_mfcc_mlp.py \
      --features data_npy/test/features.npy \
      --model mfcc_mlp_output/mfcc_mlp_classifier.h5 \
      --config mfcc_mlp_output/preprocess_config.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


CLASS_NAMES = ["background", "drone", "helicopter"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_config(config_path: Path) -> dict:
    """Load preprocessing config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_features(X: np.ndarray, config: dict) -> np.ndarray:
    """Normalize features using training statistics."""
    mean = np.array(config["normalization_mean"], dtype=np.float32)
    std = np.array(config["normalization_std"], dtype=np.float32)
    return (X - mean) / (std + 1e-6)


def run_inference(model_path: Path, features_path: Path, config_path: Path,
                 labels_path: Optional[Path] = None, output_dir: Optional[Path] = None):
    """
    Run inference on features and optionally evaluate against labels.
    
    Args:
        model_path: Path to .h5 model
        features_path: Path to features.npy
        config_path: Path to preprocess_config.json
        labels_path: Optional path to labels.npy
        output_dir: Optional directory to save results
    """
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load config
    print("Loading config...")
    config = load_config(config_path)
    
    # Load features
    print(f"Loading features from {features_path.name}...")
    X = np.load(features_path).astype(np.float32)
    print(f"  Shape: {X.shape}")
    
    # Load labels (optional)
    y_true = None
    if labels_path and labels_path.exists():
        print(f"Loading labels from {labels_path.name}...")
        y_true = np.load(labels_path)
        print(f"  Shape: {y_true.shape}")
    
    # Normalize
    print("Normalizing features...")
    X_normalized = normalize_features(X, config)
    
    # Predict
    print("Running inference...")
    probs = model.predict(tf.convert_to_tensor(X_normalized, dtype=tf.float32), 
                         batch_size=32, verbose=0)
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    print(f"✓ Predictions: {predictions.shape[0]} samples")
    print()
    
    # Print results
    print("=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    # Per-sample results
    results = []
    for idx in range(len(predictions)):
        pred_class = CLASS_NAMES[predictions[idx]]
        confidence = confidences[idx]
        
        row = {
            "sample_id": idx,
            "pred_class": pred_class,
            "confidence": round(confidence, 4),
            "prob_background": round(float(probs[idx, LABEL_TO_ID["background"]]), 4),
            "prob_drone": round(float(probs[idx, LABEL_TO_ID["drone"]]), 4),
            "prob_helicopter": round(float(probs[idx, LABEL_TO_ID["helicopter"]]), 4),
        }
        
        if y_true is not None:
            true_class = CLASS_NAMES[y_true[idx]]
            correct = "✓" if predictions[idx] == y_true[idx] else "✗"
            row["true_class"] = true_class
            row["correct"] = correct
        
        results.append(row)
    
    # Print summary
    metrics_dict = {}
    cm = None
    if y_true is not None:
        acc = accuracy_score(y_true, predictions)
        print(f"\nOverall Accuracy: {acc:.4f} ({(predictions == y_true).sum()}/{len(y_true)} correct)")
        print()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, predictions, labels=list(range(len(CLASS_NAMES))))
        print("Confusion Matrix:")
        print("       " + "  ".join(f"{c:>12}" for c in CLASS_NAMES))
        for i, row_name in enumerate(CLASS_NAMES):
            print(f"{row_name:>6} " + "  ".join(f"{cm[i, j]:>12}" for j in range(len(CLASS_NAMES))))
        print()
        
        # Classification report
        report_str = classification_report(y_true, predictions, target_names=CLASS_NAMES)
        print(report_str)
        
        # Store metrics
        report_dict = classification_report(y_true, predictions, target_names=CLASS_NAMES, output_dict=True)
        metrics_dict = {
            "accuracy": round(acc, 4),
            "correct_count": int((predictions == y_true).sum()),
            "total_count": len(y_true),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        }
    else:
        # Just show distribution
        print("Prediction Distribution:")
        for class_id, class_name in enumerate(CLASS_NAMES):
            count = (predictions == class_id).sum()
            print(f"  {class_name}: {count} ({100*count/len(predictions):.1f}%)")
        print()
    
    # Show sample results
    print("Sample Results:")
    print("-" * 100)
    header = ["ID", "True", "Pred", "Conf", "BG", "Drone", "Heli", "Status"]
    print(f"{header[0]:<5} {header[1]:<15} {header[2]:<15} {header[3]:<8} {header[4]:<8} {header[5]:<8} {header[6]:<8} {header[7]:<8}")
    print("-" * 100)
    
    # Show first 20 samples
    for i, row in enumerate(results[:20]):
        true_class = row.get("true_class", "-")
        pred_class = row["pred_class"]
        conf = row["confidence"]
        prob_bg = row["prob_background"]
        prob_dr = row["prob_drone"]
        prob_hc = row["prob_helicopter"]
        status = row.get("correct", "-")
        
        print(f"{i:<5} {true_class:<15} {pred_class:<15} {conf:<8.4f} {prob_bg:<8.4f} {prob_dr:<8.4f} {prob_hc:<8.4f} {status:<8}")
    
    if len(results) > 20:
        print(f"... ({len(results) - 20} more samples)")
    
    print()
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions CSV
        csv_path = output_dir / "predictions.csv"
        keys = results[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ Saved predictions CSV: {csv_path}")
        
        # Save confusion matrix if labels provided
        if y_true is not None and cm is not None:
            # Save as CSV
            cm_csv_path = output_dir / "confusion_matrix.csv"
            with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["true\\pred"] + CLASS_NAMES)
                for i, class_name in enumerate(CLASS_NAMES):
                    writer.writerow([class_name] + cm[i].tolist())
            print(f"✓ Saved confusion matrix CSV: {cm_csv_path}")
            
            # Save as PNG
            cm_png_path = output_dir / "confusion_matrix.png"
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, cmap="Blues", aspect='auto')
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("True", fontsize=12)
            ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
            
            # Add counts
            for i in range(len(CLASS_NAMES)):
                for j in range(len(CLASS_NAMES)):
                    text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                                 color="white" if cm[i, j] > cm.max()/2 else "black",
                                 fontsize=14, fontweight='bold')
            
            fig.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(cm_png_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved confusion matrix PNG: {cm_png_path}")
            
            # Save classification report
            report_csv_path = output_dir / "classification_report.csv"
            report_dict = metrics_dict.get("classification_report", {})
            with open(report_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["class", "precision", "recall", "f1-score", "support"])
                for class_name in CLASS_NAMES:
                    if class_name in report_dict:
                        metrics = report_dict[class_name]
                        writer.writerow([class_name, 
                                       round(metrics["precision"], 4),
                                       round(metrics["recall"], 4),
                                       round(metrics["f1-score"], 4),
                                       int(metrics["support"])])
                # Add weighted averages
                if "weighted avg" in report_dict:
                    wavg = report_dict["weighted avg"]
                    writer.writerow(["weighted avg",
                                   round(wavg["precision"], 4),
                                   round(wavg["recall"], 4),
                                   round(wavg["f1-score"], 4),
                                   int(wavg["support"])])
            print(f"✓ Saved classification report CSV: {report_csv_path}")
        
        # Save metrics as JSON
        summary = {
            "num_samples": len(predictions),
            "model": str(model_path),
            "features": str(features_path),
        }
        
        if metrics_dict:
            summary.update(metrics_dict)
        
        summary_path = output_dir / "metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved metrics JSON: {summary_path}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("SAVED FILES:")
        print("=" * 70)
        print(f"  • {csv_path.name:<40} - 所有预测结果")
        if y_true is not None and cm is not None:
            print(f"  • confusion_matrix.csv{'':<32} - 混淆矩阵 (CSV)")
            print(f"  • confusion_matrix.png{'':<32} - 混淆矩阵 (PNG图片)")
            print(f"  • classification_report.csv{'':<21} - 分类报告")
        print(f"  • {summary_path.name:<40} - 完整指标 (JSON)")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inference on MFCC features using trained MLP.")
    parser.add_argument("--features", type=Path, required=True, 
                       help="Path to features.npy file")
    parser.add_argument("--labels", type=Path, default=None,
                       help="Optional path to labels.npy for evaluation")
    parser.add_argument("--model", type=Path, default="mfcc_mlp_output_2/mfcc_mlp_classifier.h5",
                       help="Path to trained model")
    parser.add_argument("--config", type=Path, default="mfcc_mlp_output_2/preprocess_config.json",
                       help="Path to preprocess config")
    parser.add_argument("-o", "--output-dir", type=Path, default="inference_output",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate files
    if not args.features.exists():
        print(f"Error: Features file not found: {args.features}")
        return
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        return
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
    run_inference(
        model_path=args.model,
        features_path=args.features,
        config_path=args.config,
        labels_path=args.labels,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
