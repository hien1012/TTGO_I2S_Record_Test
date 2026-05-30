#!/usr/bin/env python3
"""
Preprocess WAV files to MFCC features and save as numpy arrays.
This speeds up training by pre-computing features.

Usage:
    python preprocess_audio_to_npy.py audio/train -o data_npy/train
    python preprocess_audio_to_npy.py audio/val -o data_npy/val
    python preprocess_audio_to_npy.py audio/test -o data_npy/test
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import math

from train_mlp_classifier import (
    extract_mfcc_statistics,
    read_wav_mono,
    infer_label,
)


def preprocess_directory(input_dir: Path, output_dir: Path, target_sr: int = 16000,
                        max_duration_s: float = None, n_fft: int = 1024,
                        hop_length: int = 512, n_mels: int = 13):
    """
    Preprocess all WAV files in a directory and save as numpy arrays.
    
    Output structure:
    output_dir/
    ├── features.npy      (N, 26) array of MFCC features
    ├── labels.npy        (N,) array of class indices
    ├── filenames.json    list of filenames
    └── metadata.json     config and stats
    """
    
    label_map = {"background": 0, "drone": 1, "helicopter": 2}
    features_list = []
    labels_list = []
    filenames = []
    
    wav_files = sorted(input_dir.rglob("*.wav")) + sorted(input_dir.rglob("*.WAV"))
    
    print(f"Processing {len(wav_files)} WAV files from {input_dir.name}")
    
    for idx, wav_file in enumerate(wav_files, 1):
        label = infer_label(wav_file)
        if label is None:
            print(f"  [{idx}/{len(wav_files)}] ⊘ {wav_file.name} (no label)")
            continue
        
        try:
            sr, audio = read_wav_mono(wav_file, target_sr)
            
            if max_duration_s:
                max_samples = int(max_duration_s * sr)
                audio = audio[:max_samples]
            
            # Extract MFCC features
            features = extract_mfcc_statistics(audio, sr, n_fft, hop_length, n_mels)
            features_list.append(features)
            labels_list.append(label_map[label])
            filenames.append(wav_file.name)
            
            print(f"  [{idx}/{len(wav_files)}] ✓ {wav_file.name} -> {label}")
            
        except Exception as e:
            print(f"  [{idx}/{len(wav_files)}] ✗ {wav_file.name}: {e}")
    
    if not features_list:
        print(f"Error: No valid WAV files found in {input_dir}")
        return
    
    # Convert to numpy arrays
    X = np.array(features_list, dtype=np.float32)  # (N, 26)
    y = np.array(labels_list, dtype=np.int64)      # (N,)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    features_path = output_dir / "features.npy"
    labels_path = output_dir / "labels.npy"
    np.save(features_path, X)
    np.save(labels_path, y)
    print(f"\n✓ Saved features: {features_path} ({X.shape})")
    print(f"✓ Saved labels: {labels_path} ({y.shape})")
    
    # Save filenames
    filenames_path = output_dir / "filenames.json"
    with open(filenames_path, "w", encoding="utf-8") as f:
        json.dump(filenames, f, indent=2)
    print(f"✓ Saved filenames: {filenames_path}")
    
    # Save metadata
    metadata = {
        "input_dir": str(input_dir),
        "num_samples": len(X),
        "num_features": X.shape[1],
        "target_sr": target_sr,
        "max_duration": max_duration_s,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "n_mels": n_mels,
        "label_map": {v: k for k, v in label_map.items()},
        "class_distribution": {label_map[label]: (y == label_map[label]).sum().item() 
                              for label in label_map}
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(X)}")
    for label, label_id in label_map.items():
        count = (y == label_id).sum()
        print(f"  {label}: {count} ({100*count/len(y):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess WAV files to MFCC features (.npy format).")
    parser.add_argument("input_dir", help="Directory containing labeled WAV files.")
    parser.add_argument("-o", "--output-dir", default="data_npy", help="Output directory for numpy files.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--max-duration", type=float, default=None, help="Max audio duration in seconds.")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for STFT.")
    parser.add_argument("--n-mels", type=int, default=13, help="Number of mel frequency bins.")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    preprocess_directory(
        input_dir, output_dir,
        target_sr=args.target_sr,
        max_duration_s=args.max_duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )


if __name__ == "__main__":
    main()
