#!/usr/bin/env python3
"""Compare Python and C CMVN normalization, int8 input, and TFLite output."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from compare_logmel_python_c import (
    FRAME_COUNT,
    N_MELS,
    error_metrics,
    prepare_pcm16,
    python_stages,
)


def build_reference(repo: Path, output: Path) -> None:
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            str(repo / "main"),
            str(repo / "tools" / "logmel_c_reference_cli_int8.c"),
            str(repo / "main" / "logmel_features_int8.c"),
            "-lm",
            "-o",
            str(output),
        ],
        cwd=repo,
        check=True,
    )


def model_details(model_path: Path):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details()[0], interpreter.get_output_details()[0]


def quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero = detail["quantization"]
    return np.clip(np.rint(values / scale + zero), -128, 127).astype(np.int8)


def predict(interpreter, input_detail: dict, output_detail: dict, tensor: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_detail["index"], tensor[np.newaxis].astype(np.int8))
    interpreter.invoke()
    raw = interpreter.get_tensor(output_detail["index"])[0]
    scale, zero = output_detail["quantization"]
    return (raw.astype(np.float32) - zero) * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("cnn_output/logmel_all_cmvn_grouped/logmel_cnn_classifier_int8.tflite"),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("logmel_int8_c_comparison"))
    parser.add_argument("--normalized-max-abs", type=float, default=0.02)
    parser.add_argument("--quantized-max-diff", type=int, default=1)
    parser.add_argument("--probability-max-abs", type=float, default=0.02)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    model_path = (repo / args.model).resolve() if not args.model.is_absolute() else args.model
    config_path = args.config or model_path.with_name("preprocess_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pcm = prepare_pcm16(args.wav.resolve())
    pcm_path = output_dir / "input_pcm16.bin"
    pcm.tofile(pcm_path)
    python = python_stages(pcm, config)

    executable = output_dir / "logmel_c_reference_int8.exe"
    build_reference(repo, executable)
    subprocess.run([str(executable), str(pcm_path), str(output_dir)], cwd=repo, check=True)

    c_normalized = np.fromfile(output_dir / "normalized.bin", dtype="<f4").reshape(
        N_MELS, FRAME_COUNT, 1
    )
    c_quantized = np.fromfile(output_dir / "quantized_int8.bin", dtype=np.int8).reshape(
        N_MELS, FRAME_COUNT, 1
    )
    interpreter, input_detail, output_detail = model_details(model_path)
    python_quantized = quantize(python["normalized"], input_detail)
    quantized_difference = c_quantized.astype(np.int16) - python_quantized.astype(np.int16)

    python_probability = predict(interpreter, input_detail, output_detail, python_quantized)
    c_probability = predict(interpreter, input_detail, output_detail, c_quantized)
    normalized_error = error_metrics(python["normalized"], c_normalized)
    probability_error = error_metrics(python_probability, c_probability)
    max_q_diff = int(np.max(np.abs(quantized_difference)))
    mismatch_count = int(np.count_nonzero(quantized_difference))
    class_names = list(config["class_names"])
    passed = (
        normalized_error["max_abs"] <= args.normalized_max_abs
        and max_q_diff <= args.quantized_max_diff
        and probability_error["max_abs"] <= args.probability_max_abs
        and int(np.argmax(python_probability)) == int(np.argmax(c_probability))
    )
    report = {
        "wav": str(args.wav.resolve()),
        "model": str(model_path),
        "normalized_error": normalized_error,
        "quantized_max_abs_difference": max_q_diff,
        "quantized_mismatch_count": mismatch_count,
        "quantized_value_count": int(c_quantized.size),
        "python_probability": python_probability.astype(float).tolist(),
        "c_probability": c_probability.astype(float).tolist(),
        "probability_error": probability_error,
        "python_prediction": class_names[int(np.argmax(python_probability))],
        "c_prediction": class_names[int(np.argmax(c_probability))],
        "passed": passed,
    }
    (output_dir / "comparison_report_int8.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
