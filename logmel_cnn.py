#!/usr/bin/env python3
"""Log-mel CNN model pieces shared by both trainers.

Order of use inside a trainer:
  1. normalize_and_reshape: stored log-mel -> derived channels (e.g. CMVN)
     -> train-only scalar normalization -> optional Mel-row mask.
  2. build_logmel_cnn + class_weight_from_labels: model and loss weights.
  3. recalibrate_batch_normalization: refresh BN statistics on clean train data.
  4. export_tflite_models / tflite_metrics: float32 and full-int8 TFLite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from audio_dataset import Dataset
from log_mel_features import DEFAULT_CMVN_STD_FLOOR_DB
from training_reports import grouped_indices_by_source


CHANNEL_COUNTS = {
    "logmel": 1,
    "logmel-delta": 2,
    "logmel-cmn-delta": 3,
    "cmn-delta": 2,
    "cmvn": 1,
}


def load_tensorflow():
    # Lazy import keeps --help and argument validation from paying TensorFlow's
    # startup cost. Training still uses TensorFlow/Keras tensors for Keras 3.
    import tensorflow as tf
    from tensorflow import keras

    return tf, keras


# ---------------------------------------------------------------------------
# Model input
# ---------------------------------------------------------------------------

def parse_mel_band_mask(specs: list[str] | None, n_mels: int) -> list[int]:
    """Expand CLI Mel indices and inclusive ranges into sorted unique rows."""
    indices: set[int] = set()
    for raw_spec in specs or []:
        spec = raw_spec.strip()
        if not spec:
            raise ValueError("Empty value in --mask-mel-bands")
        if "-" in spec:
            parts = spec.split("-", maxsplit=1)
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(
                    f"Invalid Mel range {raw_spec!r}; expected an inclusive range such as 18-31"
                )
            start, end = (int(part) for part in parts)
            if start > end:
                raise ValueError(f"Mel range start exceeds end: {raw_spec!r}")
            indices.update(range(start, end + 1))
        elif spec.isdigit():
            indices.add(int(spec))
        else:
            raise ValueError(
                f"Invalid Mel index {raw_spec!r}; expected an integer or inclusive range"
            )
    invalid = sorted(index for index in indices if index < 0 or index >= n_mels)
    if invalid:
        raise ValueError(
            f"Masked Mel indices must be in [0, {n_mels - 1}], got {invalid}"
        )
    if len(indices) == n_mels:
        raise ValueError("--mask-mel-bands cannot mask every Mel band")
    return sorted(indices)


def build_input_channels(images: np.ndarray, mode: str) -> np.ndarray:
    """Derive inexpensive channel-invariant views from one stored log-mel image."""
    raw = np.asarray(images, dtype=np.float32)
    cmn = raw - np.mean(raw, axis=2, keepdims=True, dtype=np.float32)
    cmvn_std = np.std(raw, axis=2, keepdims=True, dtype=np.float32)
    cmvn = cmn / np.maximum(cmvn_std, np.float32(DEFAULT_CMVN_STD_FLOOR_DB))

    # Keep the legacy modes unchanged. cmn-delta intentionally excludes raw
    # log-mel so the CNN cannot rely directly on a fixed microphone response.
    channels = [] if mode in {"cmn-delta", "cmvn"} else [raw]
    if mode == "cmvn":
        # The floor prevents low-energy, nearly constant bands from amplifying noise.
        channels.append(cmvn)
    if mode in {"logmel-cmn-delta", "cmn-delta"}:
        channels.append(cmn)
    if mode in {"logmel-delta", "logmel-cmn-delta", "cmn-delta"}:
        delta = np.zeros_like(raw)
        delta[:, :, 1:] = raw[:, :, 1:] - raw[:, :, :-1]
        channels.append(delta)
    if not channels:
        raise ValueError(f"Unsupported input channel mode: {mode}")
    return np.stack(channels, axis=-1).astype(np.float32)


def normalize_and_reshape(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    n_mels: int,
    frame_count: int,
    channel_mode: str,
    masked_mel_bands: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    """Normalize from train statistics, then zero optional Mel rows."""
    base_shape = (-1, n_mels, frame_count)
    train = build_input_channels(train_data.X.reshape(base_shape), channel_mode)
    val = build_input_channels(val_data.X.reshape(base_shape), channel_mode)
    test = build_input_channels(test_data.X.reshape(base_shape), channel_mode)

    mean = np.mean(train, axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.std(train, axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    if np.any(std <= 1e-12):
        raise ValueError(f"Near-zero training standard deviation for channels: {std}")
    denom = std + np.float32(1e-6)
    normalized_train = ((train - mean) / denom).astype(np.float32)
    normalized_val = ((val - mean) / denom).astype(np.float32)
    normalized_test = ((test - mean) / denom).astype(np.float32)
    if masked_mel_bands:
        # Zero is the training-mean value after scalar normalization. Apply the
        # same deterministic mask to every split without changing tensor shape.
        normalized_train[:, masked_mel_bands, :, :] = np.float32(0.0)
        normalized_val[:, masked_mel_bands, :, :] = np.float32(0.0)
        normalized_test[:, masked_mel_bands, :, :] = np.float32(0.0)
    return (
        normalized_train,
        normalized_val,
        normalized_test,
        mean.astype(float).tolist(),
        std.astype(float).tolist(),
    )


# ---------------------------------------------------------------------------
# Model and training helpers
# ---------------------------------------------------------------------------

def build_logmel_cnn(keras, input_shape: tuple[int, int, int], num_classes: int):
    """Build a small standard-op CNN that preserves absolute mel location."""
    mel_after_pool = input_shape[0] // 4
    time_after_pool = input_shape[1] // 4
    if mel_after_pool < 1 or time_after_pool < 1:
        raise ValueError(f"Input is too small for two pooling blocks: {input_shape}")

    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(16, 3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.SeparableConv2D(32, 3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.SeparableConv2D(48, 3, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.AveragePooling2D(pool_size=(1, time_after_pool), name="time_average_pool"),
            keras.layers.Conv2D(
                num_classes,
                kernel_size=(mel_after_pool, 1),
                name="frequency_aware_classifier",
            ),
            keras.layers.GlobalAveragePooling2D(name="class_logits"),
            keras.layers.Activation("softmax", name="class_probabilities"),
        ],
        name="logmel_compact_cnn",
    )


def class_weight_from_labels(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def recalibrate_batch_normalization(
    model,
    tf,
    X_calibration: np.ndarray,
    *,
    passes: int,
    momentum: float,
    batch_size: int,
    random_state: int,
) -> Dict:
    """Update BN moving statistics without changing trainable model weights."""
    bn_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    metadata = {
        "enabled": bool(passes > 0 and bn_layers),
        "passes": int(passes),
        "momentum": float(momentum),
        "batch_size": int(batch_size),
        "sample_count": int(len(X_calibration)),
        "batch_norm_layer_count": len(bn_layers),
        "calibration_data": "normalized_non_augmented_training_samples",
        "weights_updated": False,
    }
    if not metadata["enabled"]:
        print("BatchNorm recalibration: disabled")
        return metadata
    if len(X_calibration) == 0:
        raise ValueError("BatchNorm recalibration requires at least one train sample")

    original_momenta = [float(layer.momentum) for layer in bn_layers]
    for layer in bn_layers:
        layer.momentum = momentum

    rng = np.random.default_rng(random_state)
    try:
        for pass_index in range(passes):
            indices = rng.permutation(len(X_calibration))
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                # training=True makes BN consume this batch and update only its
                # moving mean/variance. No GradientTape or optimizer is used.
                model(
                    tf.convert_to_tensor(X_calibration[batch_indices], dtype=tf.float32),
                    training=True,
                )
            print(f"BatchNorm recalibration pass {pass_index + 1}/{passes} complete")
    finally:
        for layer, original_momentum in zip(bn_layers, original_momenta):
            layer.momentum = original_momentum

    return metadata


def compare_model_weights(first_model, second_model) -> Dict[str, object]:
    """Report whether two Keras models contain the same numerical weights."""
    first_weights = first_model.get_weights()
    second_weights = second_model.get_weights()
    if len(first_weights) != len(second_weights):
        return {
            "comparable": False,
            "first_weight_array_count": len(first_weights),
            "second_weight_array_count": len(second_weights),
            "identical": False,
            "max_abs_difference": None,
            "different_weight_array_count": None,
        }

    differences = [
        float(np.max(np.abs(first - second))) if first.size else 0.0
        for first, second in zip(first_weights, second_weights)
    ]
    return {
        "comparable": True,
        "first_weight_array_count": len(first_weights),
        "second_weight_array_count": len(second_weights),
        "identical": bool(all(difference == 0.0 for difference in differences)),
        "max_abs_difference": float(max(differences, default=0.0)),
        "different_weight_array_count": int(
            sum(difference != 0.0 for difference in differences)
        ),
    }


# ---------------------------------------------------------------------------
# TFLite export and verification
# ---------------------------------------------------------------------------

def representative_dataset(X_normalized: np.ndarray, max_samples: int):
    """Yield normalized train samples for full-int8 TFLite calibration."""
    if max_samples <= 0:
        raise ValueError("--tflite-representative-samples must be greater than 0")
    if len(X_normalized) == 0:
        raise ValueError("Cannot quantize TFLite model with an empty training set")

    sample_count = min(int(max_samples), len(X_normalized))
    indices = np.linspace(0, len(X_normalized) - 1, num=sample_count, dtype=np.int64)
    for idx in indices:
        # TFLite representative samples must match the model input batch shape.
        yield [X_normalized[idx : idx + 1].astype(np.float32)]


def tflite_tensor_details(detail: Dict) -> Dict:
    """Keep the C-side quantization parameters visible in JSON metadata."""
    scale, zero_point = detail.get("quantization", (0.0, 0))
    return {
        "name": str(detail.get("name", "")),
        "shape": detail["shape"].astype(int).tolist(),
        "dtype": np.dtype(detail["dtype"]).name,
        "quantization_scale": float(scale),
        "quantization_zero_point": int(zero_point),
    }


def inspect_tflite_io(tf, model_bytes: bytes) -> Dict:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    return {
        "inputs": [tflite_tensor_details(item) for item in interpreter.get_input_details()],
        "outputs": [tflite_tensor_details(item) for item in interpreter.get_output_details()],
    }


def tflite_ops(tf, model_bytes: bytes) -> list[str]:
    """Record converted operator names so firmware resolver changes are explicit."""
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    # Desktop XNNPACK may append a synthetic DELEGATE entry after allocation;
    # it is not a FlatBuffer operator and must not enter the MCU resolver list.
    return sorted(
        {
            str(item["op_name"])
            for item in interpreter._get_ops_details()
            if str(item["op_name"]) != "DELEGATE"
        }
    )


def export_tflite_models(
    tf,
    model,
    output_dir: Path,
    X_train: np.ndarray,
    sample_limit: int,
    *,
    filename_prefix: str = "logmel_cnn_classifier",
) -> Dict:
    """Export float32 and full-int8 CNNs; log-mel remains a firmware frontend."""
    float_path = output_dir / f"{filename_prefix}_float32.tflite"
    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model = float_converter.convert()
    float_path.write_bytes(float_model)

    int8_path = output_dir / f"{filename_prefix}_int8.tflite"
    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    int8_converter.representative_dataset = lambda: representative_dataset(X_train, sample_limit)
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    int8_converter.inference_input_type = tf.int8
    int8_converter.inference_output_type = tf.int8
    int8_model = int8_converter.convert()
    int8_path.write_bytes(int8_model)
    return {
        "float32_path": str(float_path),
        "float32_size_bytes": int(float_path.stat().st_size),
        "float32_io": inspect_tflite_io(tf, float_model),
        "float32_ops": tflite_ops(tf, float_model),
        "int8_path": str(int8_path),
        "int8_size_bytes": int(int8_path.stat().st_size),
        "int8_io": inspect_tflite_io(tf, int8_model),
        "int8_ops": tflite_ops(tf, int8_model),
        "int8_representative_samples": int(min(sample_limit, len(X_train))),
    }


def tflite_metrics(tf, model_path: str, X: np.ndarray, dataset: Dataset) -> Dict:
    """Verify converted predictions globally and for every microphone root."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    predictions = np.empty(len(dataset.y), dtype=np.int64)

    for sample_index in range(len(dataset.y)):
        sample = X[sample_index : sample_index + 1]
        if np.issubdtype(input_detail["dtype"], np.integer):
            scale, zero_point = input_detail["quantization"]
            if scale <= 0.0:
                raise ValueError("Invalid TFLite integer input scale")
            sample = np.clip(np.rint(sample / scale + zero_point), -128, 127)
        interpreter.set_tensor(input_detail["index"], sample.astype(input_detail["dtype"]))
        interpreter.invoke()
        predictions[sample_index] = int(np.argmax(interpreter.get_tensor(output_detail["index"])[0]))

    def summarize(indices: np.ndarray) -> Dict:
        truth = dataset.y[indices]
        predicted = predictions[indices]
        return {
            "sample_count": int(len(indices)),
            "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
        }

    result = summarize(np.arange(len(dataset.y), dtype=np.int64))
    result["per_source"] = {
        source: summarize(np.asarray(indices, dtype=np.int64))
        for source, indices in sorted(grouped_indices_by_source(dataset).items())
    }
    return result
