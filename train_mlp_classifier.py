#!/usr/bin/env python3
"""
Train an MLP classifier on MFCC features for audio classification.
Supports: background, drone, helicopter

MFCC features: 13 coefficients × (mean + std) = 26 dimensions per sample
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, List

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


def create_mel_filterbank(sr: int, n_fft: int, n_mels: int = 13, 
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


def compute_mfcc_frames(y: np.ndarray, sr: int, n_fft: int = 1024, 
                        hop_length: int = 512, n_mels: int = 13) -> np.ndarray:
    """Compute MFCC features frame by frame."""
    
    mel_fb = create_mel_filterbank(sr, n_fft, n_mels, f_min=0.0, f_max=sr/2)
    
    n_frames = 1 + (len(y) - n_fft) // hop_length
    S = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    window = np.hamming(n_fft).astype(np.float32)
    
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
        S[:, i] = spectrum
    
    # Apply mel filterbank
    mel_spec = np.dot(mel_fb, S)
    frame_means = np.mean(mel_spec, axis=0, keepdims=True) #針對每一幀求平均
    #relative_mel_spec = mel_spec / (frame_means + 1e-10)
    mel_spec_db = 20 * np.log10(mel_spec + 1e-10)
    
    # Apply DCT for MFCC
    mfcc = np.zeros((n_mels, n_frames), dtype=np.float32)
    for i in range(n_frames):
        mfcc[:, i] = scipy_dct(mel_spec_db[:, i], norm='ortho')[:n_mels]
    
    return mfcc


def scipy_dct(x: np.ndarray, norm: str = 'ortho') -> np.ndarray:
    """Simple DCT implementation."""
    N = len(x)
    result = np.zeros(N, dtype=np.float32)
    
    for k in range(N):
        sum_val = 0.0
        for n in range(N):
            sum_val += x[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        
        if norm == 'ortho':
            if k == 0:
                result[k] = sum_val * np.sqrt(1.0 / N)
            else:
                result[k] = sum_val * np.sqrt(2.0 / N)
        else:
            result[k] = sum_val
    
    return result


def normalize_rms(y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to a target RMS level in dB."""
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-10:
        return y
    scalar = 10**(target_db / 20.0) / (rms + 1e-10)
    return y * scalar


def compute_simple_delta(feat: np.ndarray) -> np.ndarray:
    """
    MCU-friendly Delta computation (Central Difference).
    feat shape: (n_mels, n_frames)
    """
    delta = np.zeros_like(feat)
    
    # 計算時間軸 (axis=1) 上的中央差分: delta[t] = (feat[t+1] - feat[t-1]) / 2
    # 這在 C 語言中只需要一個簡單的 for 迴圈就能實作
    delta[:, 1:-1] = (feat[:, 2:] - feat[:, :-2]) / 2.0
    
    # 處理邊界條件 (前向差分與後向差分)
    delta[:, 0] = feat[:, 1] - feat[:, 0]
    delta[:, -1] = feat[:, -1] - feat[:, -2]
    
    return delta

def extract_mfcc_statistics_old_mean_std(y: np.ndarray, sr: int, n_fft: int = 1024,
                           hop_length: int = 512, n_mels: int = 13) -> np.ndarray:
    """Extract MFCC statistics: mean and std of each coefficient (26 features)."""
    
    # 1. Apply RMS normalization to balance global volume
    # = normalize_rms(y, target_db=-20.0)
    
    # 2. Compute frames
    mfcc_frames = compute_mfcc_frames(y, sr, n_fft, hop_length, n_mels)
    
    # 3. Compute mean and std for each MFCC coefficient
    mfcc_mean = np.mean(mfcc_frames, axis=1)  # (13,)
    mfcc_mean = mfcc_mean - np.mean(mfcc_mean) # CMS
    mfcc_std = np.std(mfcc_frames, axis=1)    # (13,)
    
    # Concatenate into 26-dimensional feature vector
    features = np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)
    
    return features

#this is delta std, std
def extract_mfcc_statistics(y: np.ndarray, sr: int, n_fft: int = 1024,
                           hop_length: int = 512, n_mels: int = 13,
                           pre_emphasis_alpha: float = 0.97) -> np.ndarray:
    """
    Extract robust 26-dim features (MFCC Std + Delta Std) against Mic Shift.
    """
    
    # 1. 去除直流偏移 (DC Offset Removal)
    y = y - np.mean(y)
    
    # 2. 預強調 (Pre-emphasis)
    y = np.append(y[0], y[1:] - pre_emphasis_alpha * y[:-1])
    
    # 3. 音量正規化 (RMS Normalization)
    y = normalize_rms(y, target_db=-20.0)
    
    # 4. 計算逐幀 MFCC (13, n_frames)
    mfcc_frames = compute_mfcc_frames(y, sr, n_fft, hop_length, n_mels)
    
    # 5. 倒頻譜均值相減 (True CMS) - 抹除麥克風 LTI 差異
    time_axis_mean = np.mean(mfcc_frames, axis=1, keepdims=True)
    mfcc_cms = mfcc_frames - time_axis_mean
    
    # 6. 計算一階差分 (Delta Feature) - 捕捉動態變化
    # 注意：Delta 是在做完 CMS 後的特徵上計算
    delta_frames = compute_simple_delta(mfcc_cms)
    
    # 7. 提取統計特徵
    # 放棄 mean (因為 CMS 後 mean 趨近於 0)
    mfcc_std = np.std(mfcc_cms, axis=1)      # 13 維：各頻段能量波動幅度
    delta_std = np.std(delta_frames, axis=1) # 13 維：各頻段能量變化率的波動幅度
    
    # 8. 組合為 26 維特徵向量
    features = np.concatenate([mfcc_std, delta_std]).astype(np.float32)
    
    return features

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
                max_duration_s: float = None, n_fft: int = 1024,
                hop_length: int = 512, n_mels: int = 13) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load WAV files and extract MFCC features."""
    
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
            
            # Extract MFCC statistics (26-dimensional feature)
            features = extract_mfcc_statistics(audio, sr, n_fft, hop_length, n_mels)
            
            X.append(features)
            y.append(label_map[label])
            valid_files.append(wav_file.name)
            
            print(f"  [{idx+1}/{len(wav_files)}] {wav_file.name} -> {label} | Shape: {features.shape}")
            
        except Exception as e:
            print(f"  Error processing {wav_file.name}: {e}")
    
    print(f"\nLoaded {len(X)} files")
    return np.array(X), y, valid_files


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
    parser = argparse.ArgumentParser(description="Train MLP classifier on MFCC features.")
    parser.add_argument("input_dir", help="Directory containing labeled WAV files.")
    parser.add_argument("-o", "--output-dir", default="mfcc_mlp_output", help="Output directory for model and plots.")
    parser.add_argument("--val-dir", default=None, help="Optional fixed validation directory.")
    parser.add_argument("--test-dir", default=None, help="Optional fixed test directory.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--max-duration", type=float, default=5.0, help="Max audio duration in seconds.")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for STFT.")
    parser.add_argument("--n-mels", type=int, default=13, help="Number of mel frequency bins.")
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
    print(f"MFCC config: n_fft={args.n_fft}, hop_length={args.hop_length}, n_mels={args.n_mels}")
    print()
    
    if test_dir and val_dir:
        # Fully explicit split
        X_train, y_train, _ = load_dataset(input_dir, args.target_sr, args.max_duration, 
                                           args.n_fft, args.hop_length, args.n_mels)
        X_val, y_val, _ = load_dataset(val_dir, args.target_sr, args.max_duration,
                                       args.n_fft, args.hop_length, args.n_mels)
        X_test, y_test, _ = load_dataset(test_dir, args.target_sr, args.max_duration,
                                         args.n_fft, args.hop_length, args.n_mels)
        
        if len(X_train) < 1 or len(X_val) < 1 or len(X_test) < 1:
            print("Error: Not enough labeled files in train/validation/test directories")
            return
    elif test_dir:
        # Load fixed test set
        X_train_val, y_train_val, _ = load_dataset(input_dir, args.target_sr, args.max_duration,
                                                   args.n_fft, args.hop_length, args.n_mels)
        X_test, y_test, _ = load_dataset(test_dir, args.target_sr, args.max_duration,
                                        args.n_fft, args.hop_length, args.n_mels)
        
        if len(X_train_val) < 2 or len(X_test) < 1:
            print("Error: Not enough labeled files to train/test")
            return
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=args.val_split,
            stratify=y_train_val, random_state=42
        )
    else:
        # Load data and split
        X, y, valid_files = load_dataset(input_dir, args.target_sr, args.max_duration,
                                        args.n_fft, args.hop_length, args.n_mels)
        
        if len(X) < 2:
            print("Error: Not enough labeled files to train")
            return
        
        val_ratio = args.val_split
        test_ratio = args.test_split
        holdout_ratio = val_ratio + test_ratio
        
        if holdout_ratio <= 0 or holdout_ratio >= 1:
            print("Error: --val-split + --test-split must be greater than 0 and less than 1")
            return
        
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=holdout_ratio, stratify=y, random_state=42
        )
        
        test_ratio_within_holdout = test_ratio / holdout_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout, y_holdout, test_size=test_ratio_within_holdout,
            stratify=y_holdout, random_state=42
        )
    
    print(f"\nInput shape: {X_train.shape[1]} MFCC features (13 mean + 13 std)")
    total_bytes = X_train.nbytes + X_val.nbytes + X_test.nbytes
    total_samples = len(X_train) + len(X_val) + len(X_test)
    print(f"Data amount: {total_bytes / 1024:.1f} KB total, {total_bytes / total_samples:.1f} B per sample")
    
    # Normalize
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train = ((X_train - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    X_val = ((X_val - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    X_test = ((X_test - X_train_mean) / (X_train_std + 1e-6)).astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Convert to tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int64)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.int64)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int64)
    
    # Build and compile model
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
    
    # Save preprocess config
    preprocess_config = {
        "target_sr": args.target_sr,
        "max_duration": args.max_duration,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "n_features": 26,
        "feature_type": "mfcc",
        "normalization_mean": X_train_mean.tolist(),
        "normalization_std": X_train_std.tolist(),
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
