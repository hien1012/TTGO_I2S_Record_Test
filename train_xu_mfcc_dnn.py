#!/usr/bin/env python3
"""Xu's MFCC mean/std + DNN baseline (audio_dnn_drone_detection.ipynb) on 16 kHz data.

Faithful to the notebook:
  * front end  : xu_mfcc_features.extract_xu_mfcc (port of mfcc_function-2.c, 26 dims)
  * ESP preproc: pre-emphasis 0.90 + 5th-order Butterworth band-pass + clip
  * model      : Dense 32-64-90-64-3, ReLU, L2(0.01), dropout 0.2/0.2/0.25, softmax
  * training   : Adam (default lr), categorical CE, batch 32, up to 150 epochs,
                 EarlyStopping(val_loss, patience 10, restore best), checkpoints on
                 best val_loss and best val_accuracy; no feature normalisation
  * train data : every epoch draws a random 2 s crop from each training WAV and
                 applies online augmentation, each step with 50 % probability:
                 noise mix (non-background only, SNR 5-20 dB), pitch shift
                 (N(0, 0.3) clipped to +-0.5 semitone), gain (N(0, 2) dB clipped
                 to +-5 dB); the low-pass step is skipped as in the notebook's
                 ESP branch (already_bandpassed=True)

Adaptations (all recorded in metrics.json):
  * sample rate 16 kHz: mel edges 0-8 kHz, band-pass high cut 7.6 kHz (13.5 kHz > Nyquist)
  * noise source: background-class WAVs of this run's training split (no seg_noise set)
  * validation: fixed non-overlapping 2 s segments instead of random crops, so
    checkpoint selection is deterministic
  * train/val split: same original-file split function and seed as the log-mel
    trainer, so both architectures see identical files
  * test WAVs are scaled to [-1, 1] before preprocessing (the notebook's test cell
    clips int16-scale samples to +-1, which squares the waveform; its training
    path does the scaling correctly and is what is reproduced here)
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

import audio_dataset
from long_wav_segment_dataset import (
    OriginalWav,
    collect_original_wavs,
    split_content_group_train_val,
    split_original_train_val,
)
from training_reports import evaluate_split, evaluate_test_by_source_folder, save_training_history
from xu_mfcc_features import (
    FEATURE_DIM,
    extract_xu_mfcc,
    float32_to_int16,
    preprocess_esp32_audio,
)

CLASS_NAMES = audio_dataset.DEFAULT_CLASS_NAMES
BACKGROUND_ID = audio_dataset.LABEL_TO_ID["background"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--train-dir", nargs="+", required=True)
    parser.add_argument("--test-dir", nargs="+", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--preemphasis", type=float, default=0.90)
    parser.add_argument("--bandpass-low-hz", type=float, default=50.0)
    parser.add_argument(
        "--bandpass-high-hz",
        type=float,
        default=None,
        help="Default: min(13500, 0.95 * Nyquist) -> 13500 Hz at 44.1 kHz (notebook), 7600 Hz at 16 kHz.",
    )
    parser.add_argument("--no-esp-preprocess", action="store_true", help="Skip pre-emphasis + band-pass.")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--noise-snr-min-db", type=float, default=5.0)
    parser.add_argument("--noise-snr-max-db", type=float, default=20.0)
    parser.add_argument(
        "--crops-per-file",
        type=int,
        default=1,
        help="Random crops drawn per training WAV per epoch (notebook: 1).",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    return parser


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def read_int16_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Notebook: sf.read(dtype='int16'), stereo averaged, then /32768.

    Files at another rate are resampled with resample_poly (as audio_dataset.read_wav_mono
    does for the log-mel trainer), e.g. 44.1 kHz -> 16 kHz for the sample-rate check.
    """
    sr, data = wavfile.read(path)
    if data.dtype != np.int16:
        raise ValueError(f"{path}: expected int16 PCM, got {data.dtype}")
    if data.ndim == 2:
        data = np.mean(data.astype(np.int32), axis=1).astype(np.int16)
    audio = data.astype(np.float32) / 32768.0
    if sr != sample_rate:
        gcd = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // gcd, sr // gcd).astype(np.float32)
    return audio


class FeaturePipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sr = args.sample_rate

    def preprocess(self, segment: np.ndarray) -> np.ndarray:
        if self.args.no_esp_preprocess:
            return np.asarray(segment, dtype=np.float64)
        return preprocess_esp32_audio(
            segment,
            self.sr,
            low_hz=self.args.bandpass_low_hz,
            high_hz=self.args.bandpass_high_hz,
            alpha=self.args.preemphasis,
        )

    def features(self, preprocessed: np.ndarray) -> np.ndarray:
        return extract_xu_mfcc(float32_to_int16(preprocessed).astype(np.float32), self.sr)

    def augment(self, segment: np.ndarray, label: int, noise_pool: Sequence[np.ndarray]) -> np.ndarray:
        """Notebook augment_audio_with_noise(..., already_bandpassed=True)."""
        import librosa

        augmented = np.asarray(segment, dtype=np.float64).copy()
        if label != BACKGROUND_ID and noise_pool and random.random() < 0.5:
            noise = random.choice(noise_pool)
            if len(noise) < len(augmented):
                noise = np.tile(noise, int(np.ceil(len(augmented) / len(noise))))
            start = random.randint(0, len(noise) - len(augmented))
            noise = noise[start : start + len(augmented)]
            snr_db = random.uniform(self.args.noise_snr_min_db, self.args.noise_snr_max_db)
            speech_power = np.mean(augmented ** 2)
            noise_power = np.mean(noise ** 2)
            augmented = augmented + noise * np.sqrt(speech_power / (10 ** (snr_db / 10)) / (noise_power + 1e-8))
        if random.random() < 0.5:
            n_steps = float(np.clip(np.random.normal(loc=0.0, scale=0.3), -0.5, 0.5))
            augmented = librosa.effects.pitch_shift(augmented.astype(np.float32), sr=self.sr, n_steps=n_steps)
        if random.random() < 0.5:
            gain_db = np.clip(np.random.normal(loc=0.0, scale=2.0), -5, 5)
            augmented = augmented * (10 ** (gain_db / 20))
        return np.clip(augmented, -1.0, 1.0)


def make_train_sequence(keras, pipeline, audios, labels, noise_pool, args):
    segment_samples = int(round(args.segment_seconds * args.sample_rate))

    class XuTrainSequence(keras.utils.PyDataset):
        """Notebook AudioDataGenerator: one random crop per file per epoch."""

        def __init__(self):
            super().__init__(workers=1, use_multiprocessing=False)
            self.entries = [i for i in range(len(audios)) for _ in range(args.crops_per_file)]
            self.indexes = np.arange(len(self.entries))
            np.random.shuffle(self.indexes)

        def __len__(self):
            return int(np.ceil(len(self.entries) / args.batch_size))

        def __getitem__(self, index):
            batch = self.indexes[index * args.batch_size : (index + 1) * args.batch_size]
            x_batch, y_batch = [], []
            for k in batch:
                file_index = self.entries[int(k)]
                audio, label = audios[file_index], labels[file_index]
                start = random.randint(0, len(audio) - segment_samples)
                segment = pipeline.preprocess(audio[start : start + segment_samples])
                if not args.no_augment:
                    segment = pipeline.augment(segment, label, noise_pool)
                x_batch.append(pipeline.features(segment))
                y_batch.append(label)
            x = np.asarray(x_batch, dtype=np.float32)
            y = keras.utils.to_categorical(y_batch, len(CLASS_NAMES))
            return x, y

        def on_epoch_end(self):
            np.random.shuffle(self.indexes)

    return XuTrainSequence()


def segment_dataset(
    originals: Sequence[OriginalWav],
    split_name: str,
    pipeline: FeaturePipeline,
    args: argparse.Namespace,
) -> audio_dataset.Dataset:
    """Non-overlapping full segments, tails dropped (same rule as the log-mel trainer)."""
    segment_samples = int(round(args.segment_seconds * args.sample_rate))
    features: List[np.ndarray] = []
    labels: List[int] = []
    relative_paths: List[str] = []
    source_paths: List[Path] = []
    skipped = 0
    for item in originals:
        audio = read_int16_mono(item.path, args.sample_rate)
        count = len(audio) // segment_samples
        if count == 0:
            skipped += 1
            print(f"Skipped shorter-than-{args.segment_seconds:g}s original: {item.path}")
            continue
        for index in range(count):
            segment = audio[index * segment_samples : (index + 1) * segment_samples]
            features.append(pipeline.features(pipeline.preprocess(segment)))
            labels.append(item.label_id)
            relative_paths.append(
                (Path(item.source_tag) / item.relative_path.parent / f"{item.relative_path.stem}__seg{index:04d}.wav").as_posix()
            )
            source_paths.append(item.path)
    print(f"Built {split_name}: {len(originals)} originals -> {len(labels)} segments ({skipped} skipped)")
    return audio_dataset.Dataset(
        X=np.vstack(features).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        relative_paths=relative_paths,
        source_paths=source_paths,
        config={"split": split_name, "class_names": CLASS_NAMES, "skipped_original_file_count": skipped},
    )


def build_xu_dnn(keras, num_classes: int):
    l2 = keras.regularizers.l2
    model = keras.Sequential(
        [
            keras.Input(shape=(FEATURE_DIM,)),
            keras.layers.Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(90, activation="relu", kernel_regularizer=l2(0.01)),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate_model(model, tf, name, val_data, test_data, output_dir, batch_size) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "val": evaluate_split(model, tf, "val", val_data, val_data.X, CLASS_NAMES, output_dir, batch_size, False),
        "test": evaluate_split(model, tf, "test", test_data, test_data.X, CLASS_NAMES, output_dir, batch_size, False),
        "test_by_folder": evaluate_test_by_source_folder(
            model, tf, test_data, test_data.X, CLASS_NAMES, output_dir, batch_size
        ),
        "selection": name,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.bandpass_high_hz is None:
        args.bandpass_high_hz = min(13500.0, 0.95 * args.sample_rate / 2)
    if not 0.0 < args.bandpass_low_hz < args.bandpass_high_hz < args.sample_rate / 2:
        raise ValueError("band-pass edges must satisfy 0 < low < high < Nyquist")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.random_state)
    np.random.seed(args.random_state)
    import tensorflow as tf
    import keras

    tf.random.set_seed(args.random_state)

    # Identical original-file split to train_cnn_from_wav_logmel.py for the same seed.
    train_pool = collect_original_wavs(args.train_dir, "train pool")
    test_originals = collect_original_wavs(args.test_dir, "test")
    if len(args.train_dir) > 1:
        train_originals, val_originals, _ = split_content_group_train_val(
            train_pool, args.val_split, args.random_state
        )
    else:
        train_originals, val_originals = split_original_train_val(train_pool, args.val_split, args.random_state)
    audio_dataset.write_json(
        output_dir / "original_file_split.json",
        {
            "random_state": args.random_state,
            "val_ratio": args.val_split,
            "train_original_wavs": [str(item.path) for item in train_originals],
            "val_original_wavs": [str(item.path) for item in val_originals],
            "test_original_wavs": [str(item.path) for item in test_originals],
        },
    )

    pipeline = FeaturePipeline(args)
    segment_samples = int(round(args.segment_seconds * args.sample_rate))
    train_audio, train_labels = [], []
    for item in train_originals:
        audio = read_int16_mono(item.path, args.sample_rate)
        if len(audio) < segment_samples:
            print(f"Skipped shorter-than-{args.segment_seconds:g}s training original: {item.path}")
            continue
        train_audio.append(audio)
        train_labels.append(item.label_id)
    noise_pool = [audio for audio, label in zip(train_audio, train_labels) if label == BACKGROUND_ID]
    print(f"Training WAVs: {len(train_audio)}; background noise pool: {len(noise_pool)}")

    val_data = segment_dataset(val_originals, "val", pipeline, args)
    test_data = segment_dataset(test_originals, "test", pipeline, args)
    for name, data in (("val", val_data), ("test", test_data)):
        audio_dataset.save_npy_dataset(output_dir / "npy_splits" / name, data)

    model = build_xu_dnn(keras, len(CLASS_NAMES))
    model.summary()
    best_loss_path = output_dir / "best_val_loss_model.keras"
    best_acc_path = output_dir / "best_val_acc_model.keras"
    y_val = keras.utils.to_categorical(val_data.y, len(CLASS_NAMES))
    history = model.fit(
        make_train_sequence(keras, pipeline, train_audio, train_labels, noise_pool, args),
        validation_data=(val_data.X, y_val),
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(best_loss_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
            keras.callbacks.ModelCheckpoint(best_acc_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.early_stopping_patience, restore_best_weights=True, verbose=1
            ),
        ],
        verbose=2,
    )
    save_training_history(history, output_dir)
    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_acc_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    best_loss_model = keras.models.load_model(best_loss_path)
    best_acc_model = keras.models.load_model(best_acc_path)
    test_pred = np.argmax(best_loss_model.predict(test_data.X, batch_size=args.batch_size, verbose=0), axis=1)
    audio_dataset.write_csv_rows(
        output_dir / "test_predictions.csv",
        [
            {"relative_path": path, "true": int(true), "pred": int(pred)}
            for path, true, pred in zip(test_data.relative_paths, test_data.y, test_pred)
        ],
    )
    metrics = {
        "model_type": "xu_mfcc26_dnn",
        "class_names": CLASS_NAMES,
        "parameter_count": int(best_loss_model.count_params()),
        "epochs_run": epochs_run,
        "best_val_loss_epoch": best_epoch,
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_acc_epoch": best_acc_epoch,
        "train_wavs": len(train_audio),
        "val_segments": int(len(val_data.y)),
        "test_segments": int(len(test_data.y)),
        "config": {
            **{key: value for key, value in vars(args).items()},
            "front_end": "xu_mfcc_features.extract_xu_mfcc (port of mfcc_function-2.c)",
            "mel_edges": "librosa.mel_frequencies(28, 0, sample_rate/2, htk=False)",
            "noise_source": "background-class WAVs of the training split",
            "validation": "fixed non-overlapping segments",
            "feature_normalization": "none (as in the notebook)",
        },
    }
    # Primary result: best-val-loss checkpoint (= EarlyStopping restore_best_weights).
    metrics.update(evaluate_model(best_loss_model, tf, "best_val_loss", val_data, test_data, output_dir, args.batch_size))
    metrics["best_val_acc_model"] = evaluate_model(
        best_acc_model, tf, "best_val_acc", val_data, test_data, output_dir / "best_val_acc", args.batch_size
    )
    audio_dataset.write_json(output_dir / "metrics.json", metrics)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
