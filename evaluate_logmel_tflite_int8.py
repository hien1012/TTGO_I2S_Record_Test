#!/usr/bin/env python3
"""Evaluate full-int8 log-mel CMVN TFLite inference and optional float parity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from quantize_logmel_cnn_int8 import prepare_model_inputs


DEFAULT_ROOT = Path("cnn_output/logmel_all_cmvn_grouped")


def run_model(tf, model_path: Path, inputs: np.ndarray) -> tuple[np.ndarray, dict]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    output = np.empty((len(inputs), int(output_info["shape"][-1])), dtype=np.float32)

    input_scale, input_zero = input_info["quantization"]
    output_scale, output_zero = output_info["quantization"]
    integer_input = np.issubdtype(input_info["dtype"], np.integer)
    integer_output = np.issubdtype(output_info["dtype"], np.integer)
    if integer_input and input_scale <= 0.0:
        raise ValueError("integer input has no valid quantization scale")
    if integer_output and output_scale <= 0.0:
        raise ValueError("integer output has no valid quantization scale")

    for index, sample in enumerate(inputs):
        tensor = sample[np.newaxis]
        if integer_input:
            tensor = np.clip(np.rint(tensor / input_scale + input_zero), -128, 127)
        interpreter.set_tensor(input_info["index"], tensor.astype(input_info["dtype"]))
        interpreter.invoke()
        value = interpreter.get_tensor(output_info["index"])[0]
        if integer_output:
            value = (value.astype(np.float32) - output_zero) * output_scale
        output[index] = value.astype(np.float32)

    return output, {
        "input_dtype": np.dtype(input_info["dtype"]).name,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero),
        "output_dtype": np.dtype(output_info["dtype"]).name,
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_ROOT / "logmel_cnn_classifier_int8.tflite")
    parser.add_argument("--float-model", type=Path, default=DEFAULT_ROOT / "logmel_cnn_classifier_float32.tflite")
    parser.add_argument("--config", type=Path, default=DEFAULT_ROOT / "preprocess_config.json")
    parser.add_argument("--features", type=Path, default=DEFAULT_ROOT / "npy_splits/test/features.npy")
    parser.add_argument("--labels", type=Path, default=DEFAULT_ROOT / "npy_splits/test/labels.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("int8_evaluation_output"))
    args = parser.parse_args()

    import tensorflow as tf

    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = prepare_model_inputs(args.features, config)
    labels = np.load(args.labels).astype(np.int64)
    if len(labels) != len(inputs):
        raise ValueError("feature/label sample counts do not match")

    int8_prob, quantization = run_model(tf, args.model, inputs)
    float_prob, _ = run_model(tf, args.float_model, inputs)
    int8_pred = np.argmax(int8_prob, axis=1)
    float_pred = np.argmax(float_prob, axis=1)
    class_names = list(config["class_names"])

    rows = []
    for index in range(len(inputs)):
        rows.append(
            {
                "index": index,
                "truth": class_names[int(labels[index])],
                "float_prediction": class_names[int(float_pred[index])],
                "int8_prediction": class_names[int(int8_pred[index])],
                "prediction_match": bool(float_pred[index] == int8_pred[index]),
                "max_probability_error": float(np.max(np.abs(float_prob[index] - int8_prob[index]))),
                **{f"float_{name}": float(float_prob[index, i]) for i, name in enumerate(class_names)},
                **{f"int8_{name}": float(int8_prob[index, i]) for i, name in enumerate(class_names)},
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions_int8.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_count": len(inputs),
        "float_accuracy": float(np.mean(float_pred == labels)),
        "int8_accuracy": float(np.mean(int8_pred == labels)),
        "prediction_agreement": float(np.mean(float_pred == int8_pred)),
        "probability_max_abs": float(np.max(np.abs(float_prob - int8_prob))),
        "probability_mean_abs": float(np.mean(np.abs(float_prob - int8_prob))),
        "quantization": quantization,
    }
    (args.output_dir / "summary_int8.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
