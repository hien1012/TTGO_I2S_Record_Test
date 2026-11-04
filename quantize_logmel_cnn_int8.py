#!/usr/bin/env python3
"""Convert the existing log-mel CMVN Keras CNN to a full-int8 TFLite model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from logmel_cnn import build_input_channels


DEFAULT_ROOT = Path("cnn_output/logmel_all_cmvn_grouped")


def prepare_model_inputs(features_path: Path, config: dict) -> np.ndarray:
    """Recreate the normalized CNN input used during training."""
    raw = np.load(features_path).astype(np.float32)
    n_mels, frame_count, channels = [int(value) for value in config["input_shape"]]
    if channels != 1 or config.get("input_channels") != "cmvn":
        raise ValueError("this int8 firmware variant requires one-channel CMVN input")
    expected_values = n_mels * frame_count
    if raw.size != len(raw) * expected_values:
        raise ValueError(f"features shape {raw.shape} is not compatible with {n_mels}x{frame_count}")

    images = raw.reshape(-1, n_mels, frame_count)
    model_input = build_input_channels(images, "cmvn")
    mean = np.asarray(config["normalization_mean"], dtype=np.float32)
    std = np.asarray(config["normalization_std"], dtype=np.float32)
    return ((model_input - mean) / (std + np.float32(1e-6))).astype(np.float32)


def inspect_tflite(tf, model_data: bytes) -> dict:
    interpreter = tf.lite.Interpreter(model_content=model_data)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    ops = sorted(
        {
            str(item["op_name"])
            for item in interpreter._get_ops_details()
            if str(item["op_name"]) != "DELEGATE"
        }
    )
    return {
        "input_shape": [int(value) for value in input_info["shape"]],
        "input_dtype": np.dtype(input_info["dtype"]).name,
        "input_scale": float(input_info["quantization"][0]),
        "input_zero_point": int(input_info["quantization"][1]),
        "output_shape": [int(value) for value in output_info["shape"]],
        "output_dtype": np.dtype(output_info["dtype"]).name,
        "output_scale": float(output_info["quantization"][0]),
        "output_zero_point": int(output_info["quantization"][1]),
        "ops": ops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keras-model", type=Path, default=DEFAULT_ROOT / "logmel_cnn_classifier.keras")
    parser.add_argument("--config", type=Path, default=DEFAULT_ROOT / "preprocess_config.json")
    parser.add_argument(
        "--train-features",
        type=Path,
        default=DEFAULT_ROOT / "npy_splits/train/features.npy",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("cnn_output/int8_generated/logmel_cnn_classifier_int8.tflite"),
    )
    parser.add_argument("--representative-samples", type=int, default=256)
    args = parser.parse_args()

    import tensorflow as tf

    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = prepare_model_inputs(args.train_features, config)
    sample_count = min(max(args.representative_samples, 1), len(inputs))
    indices = np.linspace(0, len(inputs) - 1, sample_count, dtype=np.int64)

    def representative_dataset():
        for index in indices:
            yield [inputs[index : index + 1]]

    model = tf.keras.models.load_model(args.keras_model, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_data = converter.convert()
    info = inspect_tflite(tf, model_data)

    if info["input_dtype"] != "int8" or info["output_dtype"] != "int8":
        raise ValueError(f"full-int8 conversion failed: {info}")
    if info["input_scale"] <= 0.0 or info["output_scale"] <= 0.0:
        raise ValueError(f"invalid quantization parameters: {info}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_bytes(model_data)
    summary = {
        "keras_model": str(args.keras_model.resolve()),
        "config": str(args.config.resolve()),
        "representative_features": str(args.train_features.resolve()),
        "representative_sample_count": sample_count,
        "output_model": str(args.output_model.resolve()),
        "model_size_bytes": len(model_data),
        "sha256": hashlib.sha256(model_data).hexdigest(),
        "model_info": info,
    }
    summary_path = args.output_model.with_name("quantization_summary_int8.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
