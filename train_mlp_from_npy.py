#!/usr/bin/env python3
"""
Train MLP classifier on pre-computed MFCC features from numpy arrays.
Much faster than computing features during training!

Usage:
    # Step 1: Preprocess WAV files
    python preprocess_audio_to_npy.py audio/train -o data_npy/train
    python preprocess_audio_to_npy.py audio/val -o data_npy/val
    python preprocess_audio_to_npy.py audio/test -o data_npy/test
    
    # Step 2: Train from numpy
    python train_mlp_from_npy.py \
      --train-dir data_npy/train \
      --val-dir data_npy/val \
      --test-dir data_npy/test \
      -o mfcc_mlp_output
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


def load_npy_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and labels from numpy files."""
    
    features_path = data_dir / "features.npy"
    labels_path = data_dir / "labels.npy"
    
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing numpy files in {data_dir}")
    
    X = np.load(features_path)
    y = np.load(labels_path)
    
    print(f"Loaded {data_dir.name}: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


def build_mlp_model(input_dim: int = 26, num_classes: int = 3) -> keras.Model:
    """Build MLP model for audio classification."""
    
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        
        # Hidden layers
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        
        # Output layer
        keras.layers.Dense(num_classes, activation='softmax'),
    ])
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train MLP on pre-computed MFCC features (numpy).")
    parser.add_argument("--train-dir", required=True, help="Directory with train numpy files (features.npy, labels.npy).")
    parser.add_argument("--val-dir", default=None, help="Optional fixed validation directory.")
    parser.add_argument("--test-dir", default=None, help="Optional fixed test directory.")
    parser.add_argument("-o", "--output-dir", default="mfcc_mlp_output", help="Output directory for model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split (if --val-dir not provided).")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split (if --test-dir not provided).")
    
    args = parser.parse_args()
    
    train_dir = Path(args.train_dir).expanduser().resolve()
    val_dir = Path(args.val_dir).expanduser().resolve() if args.val_dir else None
    test_dir = Path(args.test_dir).expanduser().resolve() if args.test_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate paths
    if not train_dir.exists():
        print(f"Error: train_dir not found: {train_dir}")
        return
    if val_dir and not val_dir.exists():
        print(f"Error: val_dir not found: {val_dir}")
        return
    if test_dir and not test_dir.exists():
        print(f"Error: test_dir not found: {test_dir}")
        return
    
    print("Loading data...")
    print()
    
    if test_dir and val_dir:
        # All three specified
        X_train, y_train = load_npy_dataset(train_dir)
        X_val, y_val = load_npy_dataset(val_dir)
        X_test, y_test = load_npy_dataset(test_dir)
    elif test_dir:
        # Only test_dir specified, split train into train/val
        X_train_val, y_train_val = load_npy_dataset(train_dir)
        X_test, y_test = load_npy_dataset(test_dir)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=args.val_split,
            stratify=y_train_val, random_state=42
        )
        print(f"Split train into: {X_train.shape[0]} train, {X_val.shape[0]} val")
    else:
        # Only train_dir specified, split into train/val/test
        X_train_val, y_train_val = load_npy_dataset(train_dir)
        
        val_ratio = args.val_split
        test_ratio = args.test_split
        holdout_ratio = val_ratio + test_ratio
        
        if holdout_ratio <= 0 or holdout_ratio >= 1:
            print("Error: --val-split + --test-split must be between 0 and 1")
            return
        
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_train_val, y_train_val, test_size=holdout_ratio,
            stratify=y_train_val, random_state=42
        )
        
        test_ratio_within_holdout = test_ratio / holdout_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout, y_holdout, test_size=test_ratio_within_holdout,
            stratify=y_holdout, random_state=42
        )
        print(f"Split: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
    
    print()
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Normalize using train set stats
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train = ((X_train - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    X_val = ((X_val - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    X_test = ((X_test - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    
    # Convert to tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int64)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)
    
    # Build model
    model = build_mlp_model(input_dim=X_train.shape[1], num_classes=3)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel summary:")
    model.summary()
    
    # Train
    print(f"\nTraining model ({args.epochs} epochs)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    
    # Save model
    model_path = output_dir / "mfcc_mlp_classifier.h5"
    model.save(model_path)
    print(f"Model saved: {model_path}")
    
    # Save config
    preprocess_config = {
        "feature_type": "mfcc",
        "n_features": 26,
        "normalization_mean": X_train_mean.tolist(),
        "normalization_std": X_train_std.tolist(),
        "class_names": ["background", "drone", "helicopter"],
        "test_accuracy": float(test_acc),
    }
    config_path = output_dir / "preprocess_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(preprocess_config, f, indent=2)
    print(f"Config saved: {config_path}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()
    print(f"Plot saved: {output_dir / 'training_history.png'}")


if __name__ == "__main__":
    main()
