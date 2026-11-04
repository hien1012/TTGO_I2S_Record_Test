#!/usr/bin/env python3
"""Train and export a compact CNN from fixed-duration log-mel images.

Pipeline (main):
  1. long_wav_segment_dataset: split original WAVs into train/val (and test),
     then cut each WAV into fixed --max-duration segments -> log-mel features.
  2. audio_data_augmentation: augment training segments only.
  3. logmel_cnn.normalize_and_reshape: derived channels (e.g. CMVN) + train-only
     normalization.
  4. Train logmel_cnn.build_logmel_cnn; keep last-epoch and best-val-loss models.
  5. BatchNorm recalibration on clean train samples.
  6. float32 / full-int8 TFLite export.
  7. Reports (training_reports): metrics.json, CSVs, misclassified samples.

train_cnn_from_wav_logmel_cmvn.py is the usual entry point; it only sets the
CMVN variant's defaults and calls main() here.
"""


#python train_cnn_from_wav_logmel_cmvn.py `
#  --train-dir C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\data\microphone\16kHz\0814_重新錄音\ICS `
#  --test-dir C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\data\microphone\0812_real_test_dji\PDM C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\data\microphone\0812_real_test_dji\PDM_inverted C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\data\microphone\0812_real_test_dji\ICS C:\Users\Potato\OneDrive\Pictures\Desktop\碩論資料\data\microphone\0812_real_test_dji\INMP `
#  --val-split 0.2 `
#  --max-duration 2 `
#  --augmentation-copies 3 `
#  --augmentation-gain-min-db -35 `
#  --augmentation-gain-max-db 3 `
#  --augmentation-background-snr-min-db -10 `
#  --augmentation-background-snr-max-db 10 ` --f-min 250 `
#  --f-max 2500 ` --epochs 50 `
#  --output-dir cnn_output/logmel_cmvn/augment_new_record0814/ics_v1


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import audio_dataset
from audio_data_augmentation import (
    add_augmentation_arguments,
    augment_training_dataset,
    augmentation_config_from_args,
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
    CHANNEL_COUNTS,
    build_logmel_cnn,
    class_weight_from_labels,
    compare_model_weights,
    export_tflite_models,
    load_tensorflow,
    normalize_and_reshape,
    parse_mel_band_mask,
    recalibrate_batch_normalization,
    tflite_metrics,
)
from long_wav_segment_dataset import (
    load_grouped_cross_mic_train_val_test,
    load_segmented_train_val_test,
)
from training_reports import (
    evaluate_split,
    evaluate_test_by_source_folder,
    save_training_history,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a compact log-mel CNN. Original training WAVs are split into "
            "train/validation groups before non-overlapping fixed-duration segmentation."
        )
    )
    parser.add_argument("-o", "--output-dir", default="cnn_from_wav_logmel_output")
    parser.add_argument(
        "--train-dir",
        nargs="+",
        required=True,
        help="Raw training WAV roots; originals are split into train/val before slicing.",
    )
    parser.add_argument(
        "--test-dir",
        nargs="+",
        required=True,
        help="Raw test WAV roots; every labeled WAV is sliced and evaluated as test data.",
    )
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument(
        "--max-duration",
        type=float,
        default=2.0,
        help="Segment duration in seconds; use 1.0 for the one-second experiment.",
    )
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument(
        "--f-min",
        type=float,
        default=0.0,
        help="Lowest frequency included by the mel filterbank in Hz (default: 0).",
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=None,
        help="Highest frequency included by the mel filterbank in Hz (default: Nyquist).",
    )
    parser.add_argument("--log-floor", type=float, default=DEFAULT_LOG_FLOOR)
    parser.add_argument("--top-db", type=float, default=DEFAULT_TOP_DB)
    parser.add_argument("--log-reference", choices=["sample_max", "absolute"], default="sample_max")
    parser.add_argument("--input-channels", choices=sorted(CHANNEL_COUNTS), default="logmel")
    parser.add_argument(
        "--mask-mel-bands",
        nargs="+",
        default=None,
        metavar="INDEX_OR_RANGE",
        help=(
            "Set selected model-input Mel rows to zero after derived-channel "
            "normalization. Accepts indices and inclusive ranges, for example "
            "--mask-mel-bands 18-31. The input shape is unchanged."
        ),
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Original WAV ratio moved from --train-dir to validation before slicing (default: 0.2).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Only used with --grouped-cross-mic-split (held-out content ratio).",
    )
    parser.add_argument(
        "--grouped-cross-mic-split",
        action="store_true",
        help=(
            "Split PDM content IDs into train/val/test before slicing; use only matching "
            "held-out IDs from ICS/INMP as test data."
        ),
    )
    parser.add_argument(
        "--include-target-mics-in-train",
        action="store_true",
        help=(
            "With grouped splitting, include ICS/INMP recordings from train/val content "
            "groups while keeping held-out content groups test-only."
        ),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bn-recalibration-passes", type=int, default=3)
    parser.add_argument("--bn-recalibration-momentum", type=float, default=0.9)
    parser.add_argument(
        "--load-model",
        default=None,
        help="Load an existing compatible Keras model and skip training (useful for TFLite export).",
    )
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--tflite-representative-samples", type=int, default=256)
    add_augmentation_arguments(parser, default_enabled=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.target_sr != 16000:
        raise ValueError("Log-mel preprocessing requires --target-sr 16000")
    if args.max_duration <= 0.0:
        raise ValueError("--max-duration must be greater than zero")
    if min(args.n_fft, args.hop_length, args.n_mels) <= 0:
        raise ValueError("n_fft, hop_length, and n_mels must be greater than zero")
    effective_f_max = args.target_sr / 2.0 if args.f_max is None else args.f_max
    if not 0.0 <= args.f_min < effective_f_max <= args.target_sr / 2.0:
        raise ValueError(
            "Frequency range must satisfy 0 <= --f-min < --f-max <= target_sr / 2"
        )
    if args.log_floor <= 0.0 or args.top_db <= 0.0:
        raise ValueError("log_floor and top_db must be greater than zero")
    parse_mel_band_mask(args.mask_mel_bands, args.n_mels)
    if not 0.0 < args.val_split < 1.0 or not 0.0 < args.test_split < 1.0:
        raise ValueError("val_split and test_split must be between zero and one")
    if args.grouped_cross_mic_split and args.val_split + args.test_split >= 1.0:
        raise ValueError("Grouped split requires val_split + test_split < 1")
    if args.include_target_mics_in_train and not args.grouped_cross_mic_split:
        raise ValueError("--include-target-mics-in-train requires --grouped-cross-mic-split")
    if args.early_stopping_patience < 0 or args.bn_recalibration_passes < 0:
        raise ValueError("patience and BN recalibration passes cannot be negative")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    masked_mel_bands = parse_mel_band_mask(args.mask_mel_bands, args.n_mels)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1-2. WAV split -> segments -> log-mel; augment the train split only.
    # ------------------------------------------------------------------
    sample_count = int(round(args.target_sr * args.max_duration))
    frame_count = stft_frame_count(sample_count, args.n_fft, args.hop_length)
    expected_dim = args.n_mels * frame_count
    channel_count = CHANNEL_COUNTS[args.input_channels]
    input_shape = (args.n_mels, frame_count, channel_count)
    extractor = make_flat_log_mel_extractor(args)

    def training_augmenter(dataset: audio_dataset.Dataset, runtime_args: argparse.Namespace):
        return augment_training_dataset(
            dataset,
            runtime_args,
            feature_extractor=extractor,
            expected_dim=expected_dim,
            feature_contract=f"log_mel_{args.n_mels}x{frame_count}",
        )

    feature_layout = {
        "log_mel_spectrogram": {
            "shape": [args.n_mels, frame_count, 1],
            "flatten_order": "C_row_major_mel_then_time",
            "range": [0, expected_dim],
        }
    }
    load_splits = (
        load_grouped_cross_mic_train_val_test
        if args.grouped_cross_mic_split
        else load_segmented_train_val_test
    )
    train_data, val_data, test_data, configs = load_splits(
        args,
        output_dir,
        feature_extractor=extractor,
        expected_dim=expected_dim,
        feature_type="log_mel_spectrogram",
        feature_source="log_mel_features.extract_log_mel_spectrogram",
        feature_layout=feature_layout,
        training_augmenter=training_augmenter,
    )

    feature_dim = audio_dataset.validate_feature_dims(train_data, val_data, test_data)
    if feature_dim != expected_dim:
        raise ValueError(f"Expected {expected_dim} stored log-mel values, got {feature_dim}")

    # ------------------------------------------------------------------
    # 3. Model input: derived channels + train-only normalization.
    # ------------------------------------------------------------------
    class_names = audio_dataset.class_names_from_configs(configs)
    X_train, X_val, X_test, normalization_mean, normalization_std = normalize_and_reshape(
        train_data,
        val_data,
        test_data,
        args.n_mels,
        frame_count,
        args.input_channels,
        masked_mel_bands,
    )
    y_train = train_data.y.astype(np.int64)
    y_val = val_data.y.astype(np.int64)
    y_test = test_data.y.astype(np.int64)
    class_weight = class_weight_from_labels(y_train)

    # ------------------------------------------------------------------
    # 4. Train (or load) the CNN.
    # ------------------------------------------------------------------
    tf, keras = load_tensorflow()
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    print(f"Log-mel CNN input: {input_shape}; stored feature dim: {expected_dim}")
    print(f"Train/val/test: {X_train.shape} / {X_val.shape} / {X_test.shape}")
    print(f"Classes: {class_names}; class weights: {class_weight}")
    effective_f_max = args.target_sr / 2.0 if args.f_max is None else args.f_max
    print(f"Mel frequency range: {args.f_min:g}-{effective_f_max:g} Hz")
    print(
        "Masked model-input Mel bands: "
        + (", ".join(map(str, masked_mel_bands)) if masked_mel_bands else "none")
    )
    model_path = output_dir / "logmel_cnn_classifier.keras"
    last_raw_model_path = output_dir / "logmel_cnn_classifier_last_epoch_raw.keras"
    best_raw_model_path = output_dir / "logmel_cnn_classifier_best_val_loss_raw.keras"
    best_model_path = output_dir / "logmel_cnn_classifier_best_val_loss.keras"
    best_model = None
    best_epoch = None
    best_val_loss = None
    best_bn_recalibration = None
    last_epoch = None
    last_epoch_val_loss = None
    raw_weight_comparison = None
    deployed_weight_comparison = None
    if args.load_model:
        load_path = Path(args.load_model).expanduser().resolve()
        model = keras.models.load_model(load_path)
        if tuple(model.input_shape[1:]) != input_shape:
            raise ValueError(
                f"Loaded model input {model.input_shape[1:]} does not match requested {input_shape}"
            )
        history = None
        print(f"Loaded existing Keras model: {load_path}")
    else:
        model = build_logmel_cnn(keras, input_shape, len(class_names))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=best_raw_model_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.7, patience=20, min_lr=1e-8, verbose=1
            )
        ]
        if args.early_stopping_patience:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.early_stopping_patience,
                    # Keep the actual last completed epoch in ``model``.  The
                    # independently saved checkpoint is loaded below for the
                    # best-val-loss evaluation.
                    restore_best_weights=False,
                    verbose=1,
                )
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
        best_epoch_index = int(np.argmin(history.history["val_loss"]))
        best_epoch = best_epoch_index + 1
        best_val_loss = float(history.history["val_loss"][best_epoch_index])
        last_epoch = len(history.history["loss"])
        last_epoch_val_loss = float(history.history["val_loss"][-1])
        model.save(last_raw_model_path)
        print(f"Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
        print(
            f"Last completed epoch: {last_epoch}, "
            f"val_loss={last_epoch_val_loss:.6f}"
        )
        print(f"Raw last-epoch Keras model saved: {last_raw_model_path}")

        # Load the untouched ModelCheckpoint artifact before either model is
        # recalibrated.  This preserves an auditable epoch-exact pair.
        best_model = keras.models.load_model(best_raw_model_path, compile=False)
        raw_weight_comparison = compare_model_weights(model, best_model)
        print(
            "Raw last vs best weights: "
            f"identical={raw_weight_comparison['identical']} "
            f"max_abs_diff={raw_weight_comparison['max_abs_difference']}"
        )
    model.summary()

    # ------------------------------------------------------------------
    # 5. BatchNorm recalibration on clean (non-augmented) train samples.
    # ------------------------------------------------------------------
    augmentation_metadata = train_data.config.get("training_augmentation", {})
    original_train_count = int(augmentation_metadata.get("original_train_samples", len(X_train)))
    original_train_count = min(max(original_train_count, 1), len(X_train))
    bn_recalibration = recalibrate_batch_normalization(
        model,
        tf,
        X_train[:original_train_count],
        passes=args.bn_recalibration_passes,
        momentum=args.bn_recalibration_momentum,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )

    model.save(model_path)
    print(f"Last-epoch deployment Keras model saved: {model_path}")

    if best_model is not None:
        # Recalibrate the selected checkpoint independently. This keeps the
        # final-epoch and best-val-loss evaluations under the same fixed-BN
        # inference conditions used by exported TFLite models.
        best_bn_recalibration = recalibrate_batch_normalization(
            best_model,
            tf,
            X_train[:original_train_count],
            passes=args.bn_recalibration_passes,
            momentum=args.bn_recalibration_momentum,
            batch_size=args.batch_size,
            random_state=args.random_state,
        )
        best_model.save(best_model_path)
        print(f"Best-val-loss Keras model saved: {best_model_path}")
        deployed_weight_comparison = compare_model_weights(model, best_model)
        print(
            "Deployment last vs best weights: "
            f"identical={deployed_weight_comparison['identical']} "
            f"max_abs_diff={deployed_weight_comparison['max_abs_difference']}"
        )

    # ------------------------------------------------------------------
    # 6. TFLite export (last-epoch and best-val-loss models).
    # ------------------------------------------------------------------
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
            print(f"Best float32 TFLite saved: {best_tflite_info['float32_path']}")
            print(f"Best full-int8 TFLite saved: {best_tflite_info['int8_path']}")

    # ------------------------------------------------------------------
    # 7. Reports: metrics.json, per-split CSVs, error samples.
    # ------------------------------------------------------------------
    metrics = {
        "model_type": "compact_logmel_cnn",
        "model_path": str(model_path),
        "model_role": (
            "loaded_external_model"
            if history is None
            else "last_completed_epoch_after_batch_norm_recalibration"
        ),
        "last_epoch_model": {
            "epoch": last_epoch,
            "val_loss_at_epoch_end_before_bn_recalibration": last_epoch_val_loss,
            "raw_model_path": str(last_raw_model_path) if history is not None else None,
            "deployment_model_path": str(model_path),
            "batch_norm_recalibration": bn_recalibration,
        },
        "class_names": class_names,
        "input_shape": list(input_shape),
        "parameter_count": int(model.count_params()),
        "class_weight": class_weight,
        "mel_band_mask": {
            "enabled": bool(masked_mel_bands),
            "indices": masked_mel_bands,
            "fill_value_after_normalization": 0.0,
            "application_stage": "after_channel_derivation_and_train_scalar_normalization",
        },
        "training_augmentation": train_data.config.get(
            "training_augmentation", augmentation_config_from_args(args)
        ),
        "batch_norm_recalibration": bn_recalibration,
        "tflite": tflite_info,
    }
    grouped_manifest_path = output_dir / "content_group_split.json"
    standard_manifest_path = output_dir / "original_file_split.json"
    split_manifest_path = (
        grouped_manifest_path
        if train_data.config.get("content_group_split", False)
        else standard_manifest_path
    )
    split_manifest = audio_dataset.read_json(split_manifest_path) if split_manifest_path.exists() else None
    metrics["dataset_split"] = {
        "split_unit": train_data.config.get("split_unit", "presegmented_sample"),
        "content_group_split": bool(train_data.config.get("content_group_split", False)),
        "target_mics_in_train": bool(train_data.config.get("target_mics_in_train", False)),
        "manifest_path": str(split_manifest_path) if split_manifest is not None else None,
        "manifest": split_manifest,
    }
    metrics["train"] = evaluate_split(
        model, tf, "train", train_data, X_train, class_names, output_dir, args.batch_size, False
    )
    metrics["val"] = evaluate_split(
        model, tf, "val", val_data, X_val, class_names, output_dir, args.batch_size, True
    )
    metrics["test"] = evaluate_split(
        model, tf, "test", test_data, X_test, class_names, output_dir, args.batch_size, True
    )
    metrics["test_by_folder"] = evaluate_test_by_source_folder(
        model, tf, test_data, X_test, class_names, output_dir, args.batch_size
    )

    if best_model is not None:
        best_output_dir = output_dir / "best_val_loss"
        best_output_dir.mkdir(parents=True, exist_ok=True)
        print()
        print("=" * 72)
        print(
            f"BEST-VAL-LOSS MODEL EVALUATION "
            f"(epoch={best_epoch}, checkpoint_val_loss={best_val_loss:.6f})"
        )
        print("=" * 72)
        best_metrics = {
            "selection_monitor": "val_loss",
            "selection_mode": "min",
            "epoch": int(best_epoch),
            "val_loss_at_checkpoint": float(best_val_loss),
            "val_loss_semantics": "before_post_training_batch_norm_recalibration",
            "raw_model_path": str(best_raw_model_path),
            "model_path": str(best_model_path),
            "batch_norm_recalibration": best_bn_recalibration,
            "raw_last_vs_best_weight_comparison": raw_weight_comparison,
            "deployment_last_vs_best_weight_comparison": deployed_weight_comparison,
            "tflite": best_tflite_info,
        }
        best_metrics["train"] = evaluate_split(
            best_model,
            tf,
            "train",
            train_data,
            X_train,
            class_names,
            best_output_dir,
            args.batch_size,
            False,
        )
        best_metrics["val"] = evaluate_split(
            best_model,
            tf,
            "val",
            val_data,
            X_val,
            class_names,
            best_output_dir,
            args.batch_size,
            True,
        )
        best_metrics["test"] = evaluate_split(
            best_model,
            tf,
            "test",
            test_data,
            X_test,
            class_names,
            best_output_dir,
            args.batch_size,
            True,
        )
        best_metrics["test_by_folder"] = evaluate_test_by_source_folder(
            best_model,
            tf,
            test_data,
            X_test,
            class_names,
            best_output_dir,
            args.batch_size,
        )
        metrics["best_val_loss_model"] = best_metrics
        print()
        print("=" * 72)
        print("LAST-EPOCH VS BEST-VAL-LOSS TEST ACCURACY")
        print("=" * 72)
        print(f"Last epoch     : {metrics['test']['accuracy']:.4f} (epoch {last_epoch})")
        print(f"Best val_loss : {best_metrics['test']['accuracy']:.4f} (epoch {best_epoch})")

    preprocess_config = log_mel_feature_config(
        input_dir=Path(args.train_dir[0]),
        target_sr=args.target_sr,
        duration_seconds=args.max_duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
        log_floor=args.log_floor,
        top_db=args.top_db,
        reference=args.log_reference,
        input_channels=args.input_channels,
        num_samples=len(train_data.y) + len(val_data.y) + len(test_data.y),
    )
    preprocess_config.update(
        {
            "normalization": "per_derived_channel_scalar_from_training_split_only",
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
            "mel_band_mask": metrics["mel_band_mask"],
            "class_names": class_names,
            "keras_model_path": str(model_path),
            "keras_model_role": metrics["model_role"],
            "last_epoch": last_epoch,
            "last_epoch_val_loss_before_bn_recalibration": last_epoch_val_loss,
            "last_epoch_raw_keras_model_path": (
                str(last_raw_model_path) if history is not None else None
            ),
            "best_val_loss_keras_model_path": (
                str(best_model_path) if best_model is not None else None
            ),
            "best_val_loss_raw_keras_model_path": (
                str(best_raw_model_path) if best_model is not None else None
            ),
            "best_val_loss_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "tflite": tflite_info,
            "best_val_loss_tflite": best_tflite_info,
            "training_augmentation": metrics["training_augmentation"],
            "batch_norm_recalibration": bn_recalibration,
            "dataset_split": metrics["dataset_split"],
            "deployment_warning": (
                "TFLite quantizes the CNN only. Firmware must reproduce DC removal, "
                "Hamming STFT, mel power, log compression, derived channels, normalization, "
                "optional Mel-band masking, and int8 input quantization."
            ),
        }
    )
    audio_dataset.write_json(output_dir / "preprocess_config.json", preprocess_config)
    audio_dataset.write_json(output_dir / "metrics.json", metrics)
    if history is not None:
        save_training_history(history, output_dir)
    print(f"Metrics saved: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
