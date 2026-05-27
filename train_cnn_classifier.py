#!/usr/bin/env python3
"""
Train a CNN classifier on Mel-spectrogram for audio classification.
Supports: background, drone, helicopter
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


def hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to mel scale."""
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel: float) -> float:
    """Convert mel scale to frequency in Hz."""
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank(sr: int, n_fft: int, n_mels: int = 128, 
                         f_min: float = 0.0, f_max: float = None) -> np.ndarray:
    """Create mel filterbank."""
    if f_max is None:
        f_max = sr / 2
    
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    
    for m in range(n_mels):
        f_left = int(bin_points[m])
        f_center = int(bin_points[m + 1])
        f_right = int(bin_points[m + 2])
        
        if f_center > f_left:
            left_slope = np.arange(f_left, f_center, dtype=np.float32) - f_left
            left_slope = left_slope / (f_center - f_left)
            filterbank[m, f_left:f_center] = left_slope
        
        if f_right > f_center:
            right_slope = np.arange(f_center, f_right, dtype=np.float32) - f_center
            right_slope = 1.0 - (right_slope / (f_right - f_center))
            filterbank[m, f_center:f_right] = right_slope
    
    return filterbank


def compute_mel_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, 
                           hop_length: int = 1024, n_mels: int = 128) -> np.ndarray:
    """Compute mel-spectrogram and return in dB scale."""
    
    mel_fb = create_mel_filterbank(sr, n_fft, n_mels, f_min=0.0, f_max=sr/2)
    
    n_frames = 1 + (len(y) - n_fft) // hop_length
    S = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
        S[:, i] = spectrum
    
    mel_spec = np.dot(mel_fb, S)
    mel_spec_db = 20 * np.log10(mel_spec + 1e-10)
    
    return mel_spec_db.astype(np.float32)


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


def infer_label(path: Path) -> str:
    """Infer label from filename."""
    name = path.stem.lower()
    if "background" in name or "bg" in name or "noise" in name or name.startswith("b"):
        return "background"
    if "helicopter" in name or "heli" in name or name.startswith("h"):
        return "helicopter"
    if "drone" in name or "uav" in name or name.startswith("d"):
        return "drone"
    return None


def load_dataset(input_dir: Path, target_sr: int = 16000, 
                max_duration_s: float = None) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load WAV files and compute mel-spectrograms."""
    
    label_map = {"background": 0, "drone": 1, "helicopter": 2}
    X = []
    y = []
    valid_files = []
    
    wav_files = sorted(input_dir.rglob("*.wav")) + sorted(input_dir.rglob("*.WAV"))
    
    print(f"Found {len(wav_files)} WAV files")
    
    for idx, wav_file in enumerate(wav_files):
        label = infer_label(wav_file)
        if label is None:
            print(f"  Skipped (no label): {wav_file.name}")
            continue
        
        try:
            sr, audio = read_wav_mono(wav_file, target_sr)
            
            if max_duration_s:
                max_samples = int(max_duration_s * sr)
                audio = audio[:max_samples]
            
            mel_spec = compute_mel_spectrogram(audio, sr, n_fft=1024, hop_length=1024, n_mels=64)
            
            X.append(mel_spec)
            y.append(label_map[label])
            valid_files.append(wav_file.name)
            
            print(f"  [{idx+1}/{len(wav_files)}] {wav_file.name} -> {label} | Shape: {mel_spec.shape}")
            
        except Exception as e:
            print(f"  Error processing {wav_file.name}: {e}")
    
    print(f"\nLoaded {len(X)} files")
    return X, y, valid_files


def pad_sequences(sequences: list, target_length: int = None) -> np.ndarray:
    """Pad or truncate sequences to same length."""
    
    if target_length is None:
        target_length = max(s.shape[1] for s in sequences)
    
    padded = np.zeros((len(sequences), sequences[0].shape[0], target_length), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        if seq.shape[1] < target_length:
            padded[i, :, :seq.shape[1]] = seq
        else:
            padded[i] = seq[:, :target_length]
    
    return padded


def pad_dataset_splits(splits: list) -> list:
    """Pad all dataset splits to one shared time length."""
    
    target_length = max(seq.shape[1] for split in splits for seq in split)
    return [pad_sequences(split, target_length) for split in splits]


def build_cnn_model(input_shape: Tuple[int, int], num_classes: int = 3) -> keras.Model:
    """Build lightweight CNN model for audio classification."""
    
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        
        # Expand to 4D: (batch, height, width, channels)
        keras.layers.Reshape((input_shape[0], input_shape[1], 1)),
        
        # Block 1
        keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        # Block 2
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        
        # Global pooling and dense layers
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CNN classifier on audio mel-spectrograms.")
    parser.add_argument("input_dir", help="Directory containing labeled WAV files. With --val-dir and --test-dir, this is used for train only.")
    parser.add_argument("-o", "--output-dir", default="audio_classifier_output", help="Output directory for model and plots.")
    parser.add_argument("--val-dir", default=None, help="Optional fixed validation directory. Requires --test-dir.")
    parser.add_argument("--test-dir", default=None, help="Optional fixed test directory. If set without --val-dir, input_dir is split into train/validation only.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--max-duration", type=float, default=5.0, help="Max audio duration in seconds.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set ratio.")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test set ratio.")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).expanduser().resolve()
    val_dir = Path(args.val_dir).expanduser().resolve() if args.val_dir else None
    test_dir = Path(args.test_dir).expanduser().resolve() if args.test_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    if val_dir and not test_dir:
        print("Error: --val-dir requires --test-dir")
        return
    if val_dir and not val_dir.exists():
        print(f"Error: Validation directory not found: {val_dir}")
        return
    if test_dir and not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    print(f"Loading audio files from: {input_dir}")
    if val_dir:
        print(f"Fixed validation files from: {val_dir}")
    if test_dir:
        print(f"Fixed test files from: {test_dir}")
    print(f"Target sample rate: {args.target_sr} Hz")
    print(f"Max duration: {args.max_duration} seconds")
    print()
    
    if test_dir and val_dir:
        # Fully explicit split: input_dir=train, --val-dir=validation,
        # --test-dir=test. This is the safest mode when clips were cut from
        # longer recordings and must not cross split boundaries.
        X_train_raw, y_train, _ = load_dataset(input_dir, args.target_sr, args.max_duration)
        X_val_raw, y_val, _ = load_dataset(val_dir, args.target_sr, args.max_duration)
        X_test_raw, y_test, _ = load_dataset(test_dir, args.target_sr, args.max_duration)
        
        if len(X_train_raw) < 1 or len(X_val_raw) < 1 or len(X_test_raw) < 1:
            print("Error: Not enough labeled files in train/validation/test directories")
            return
        
        print(f"\nPadding sequences...")
        X_train, X_val, X_test = pad_dataset_splits([X_train_raw, X_val_raw, X_test_raw])
    elif test_dir:
        # Load a fixed held-out test set from a separate directory. This helps
        # avoid leakage when multiple clips come from the same original recording.
        X_train_val_raw, y_train_val, _ = load_dataset(input_dir, args.target_sr, args.max_duration)
        X_test_raw, y_test, _ = load_dataset(test_dir, args.target_sr, args.max_duration)
        
        if len(X_train_val_raw) < 2 or len(X_test_raw) < 1:
            print("Error: Not enough labeled files to train/test")
            return
        if args.val_split <= 0 or args.val_split >= 1:
            print("Error: --val-split must be greater than 0 and less than 1 when --test-dir is used")
            return
        
        print(f"\nPadding sequences...")
        X_train_val, X_test = pad_dataset_splits([X_train_val_raw, X_test_raw])
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=args.val_split,
            stratify=y_train_val, random_state=42
        )
    else:
        # Load data
        X, y, valid_files = load_dataset(input_dir, args.target_sr, args.max_duration)
        
        if len(X) < 2:
            print("Error: Not enough labeled files to train")
            return
        
        # Pad sequences to same length
        print(f"\nPadding sequences...")
        X_padded = pad_sequences(X)
        
        # Split data into train / validation / test.
        # Defaults: 60% train, 20% validation, 20% test.
        val_ratio = args.val_split
        test_ratio = args.test_split
        holdout_ratio = val_ratio + test_ratio
        
        if holdout_ratio <= 0 or holdout_ratio >= 1:
            print("Error: --val-split + --test-split must be greater than 0 and less than 1")
            return
        
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X_padded, y, test_size=holdout_ratio, stratify=y, random_state=42
        )
        
        test_ratio_within_holdout = test_ratio / holdout_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout, y_holdout, test_size=test_ratio_within_holdout,
            stratify=y_holdout, random_state=42
        )
    
    print(f"Input shape: {(len(X_train) + len(X_val) + len(X_test), X_train.shape[1], X_train.shape[2])}")
    total_bytes = X_train.nbytes + X_val.nbytes + X_test.nbytes
    total_samples = len(X_train) + len(X_val) + len(X_test)
    print(f"Data amount per sample: {total_bytes / (1024**2):.2f} MB total, "
          f"{total_bytes / total_samples / 1024:.1f} KB per sample")
    
    # Normalize
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = ((X_train - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    X_val = ((X_val - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    X_test = ((X_test - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # TensorFlow/Keras 3 can be strict about NumPy adapter detection in some
    # environments; explicit tensors keep the training input contract stable.
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int64)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)
    
    # Build model
    model = build_cnn_model((X_train.shape[1], X_train.shape[2]), num_classes=3)
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
    model_path = output_dir / "audio_classifier.h5"
    model.save(model_path)
    print(f"Model saved: {model_path}")
    
    preprocess_config = {
        "target_sr": args.target_sr,
        "max_duration": args.max_duration,
        "n_fft": 1024,
        "hop_length": 1024,
        "n_mels": 64,
        "input_shape": [int(X_train.shape[1]), int(X_train.shape[2])],
        "normalization_mean": float(X_train_mean),
        "normalization_std": float(X_train_std),
        "class_names": ["background", "drone", "helicopter"],
    }
    config_path = output_dir / "preprocess_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(preprocess_config, f, indent=2)
    print(f"Preprocess config saved: {config_path}")
    
    # Plot training history
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
    
    print(f"Training plot saved: {output_dir / 'training_history.png'}")


if __name__ == "__main__":
    main()
