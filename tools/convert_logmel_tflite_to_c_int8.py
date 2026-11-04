#!/usr/bin/env python3
"""Embed a full-int8 log-mel CMVN TFLite model without touching float files.

python tools\convert_logmel_tflite_to_c_int8.py `
  --model cnn_output\logmel_cmvn\ics_pdm\logmel_cnn_classifier_int8.tflite `
  --config cnn_output\logmel_cmvn\ics_pdm\preprocess_config.json `
  --out-dir main

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from logmel_mel_band import mel_band_c_block, validate_mel_band


EXPECTED_OPS = {
    "AVERAGE_POOL_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "MAX_POOL_2D",
    "MEAN",
    "SOFTMAX",
}


def format_bytes(data: bytes, width: int = 12) -> str:
    return "\n".join(
        "    " + ", ".join(f"0x{value:02x}" for value in data[start : start + width]) + ","
        for start in range(0, len(data), width)
    )


def format_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {value}")
    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def inspect_model(model_path: Path) -> dict:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    return {
        "input_shape": [int(value) for value in input_info["shape"]],
        "input_dtype": np.dtype(input_info["dtype"]).name,
        "input_scale": float(input_info["quantization"][0]),
        "input_zero_point": int(input_info["quantization"][1]),
        "output_shape": [int(value) for value in output_info["shape"]],
        "output_dtype": np.dtype(output_info["dtype"]).name,
        "output_scale": float(output_info["quantization"][0]),
        "output_zero_point": int(output_info["quantization"][1]),
        "ops": sorted(
            {
                str(item["op_name"])
                for item in interpreter._get_ops_details()
                if str(item["op_name"]) != "DELEGATE"
            }
        ),
    }


def validate(config: dict, model: dict) -> None:
    expected = {
        "target_sr": 16000,
        "segment_seconds": 2.0,
        "segment_samples": 32000,
        "n_fft": 512,
        "hop_length": 256,
        "n_mels": 32,
        "input_shape": [32, 124, 1],
        "input_channels": "cmvn",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"config {key}={config.get(key)!r}, expected {value!r}")
    if model["input_shape"] != [1, 32, 124, 1] or model["output_shape"] != [1, 3]:
        raise ValueError(f"unexpected model shapes: {model}")
    if model["input_dtype"] != "int8" or model["output_dtype"] != "int8":
        raise ValueError("firmware int8 model requires int8 input and output")
    if model["input_scale"] <= 0.0 or model["output_scale"] <= 0.0:
        raise ValueError("model quantization scales must be positive")
    extra_ops = set(model["ops"]) - EXPECTED_OPS
    if extra_ops:
        raise ValueError(f"unsupported operators: {sorted(extra_ops)}")
    cmvn = config.get("cmvn", {})
    if not cmvn.get("enabled") or float(cmvn.get("std_floor_db", 0.0)) <= 0.0:
        raise ValueError("valid CMVN metadata is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("main"))
    args = parser.parse_args()

    model_path = args.model.resolve()
    config_path = (args.config or model_path.with_name("preprocess_config.json")).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_info = inspect_model(model_path)
    validate(config, model_info)
    mel_f_min, mel_f_max, mel_bin_edges = validate_mel_band(config)

    means = [float(value) for value in config["normalization_mean"]]
    stds = [float(value) for value in config["normalization_std"]]
    if len(means) != 1 or len(stds) != 1 or stds[0] <= 0.0:
        raise ValueError("one-channel normalization statistics are required")
    inv_std = 1.0 / (np.float32(stds[0]) + np.float32(1e-6))
    class_names = [str(value) for value in config["class_names"]]
    if len(class_names) != 3:
        raise ValueError("exactly three classes are required")

    data = model_path.read_bytes()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "model_data_int8.h").write_text(
        """#pragma once

/**
 * @file model_data_int8.h
 * @brief Linkage contract for the generated full-INT8 TFLite flatbuffer.
 *
 * The const model array is stored in Flash. Regenerate this header,
 * model_data_int8.cc, and preprocess_config_int8.h as one model release.
 */

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Embedded full-INT8 TFLite flatbuffer and its exact byte length. */
extern const unsigned char g_audio_model_int8_tflite[];
extern const size_t g_audio_model_int8_tflite_len;
#ifdef __cplusplus
}
#endif
""",
        encoding="ascii",
        newline="\n",
    )
    (args.out_dir / "model_data_int8.cc").write_text(
        f"""/*
 * Auto-generated full-INT8 TFLite flatbuffer. Keep this file synchronized
 * with model_data_int8.h and preprocess_config_int8.h; do not hand-edit.
 */
#include "model_data_int8.h"
#include <stddef.h>
alignas(16) const unsigned char g_audio_model_int8_tflite[] = {{
{format_bytes(data)}
}};
const size_t g_audio_model_int8_tflite_len = {len(data)};
""",
        encoding="ascii",
        newline="\n",
    )
    classes = ",\n".join(f'    "{name}"' for name in class_names)
    (args.out_dir / "preprocess_config_int8.h").write_text(
        f"""#pragma once

/**
 * @file preprocess_config_int8.h
 * @brief Generated preprocessing, tensor, and INT8 quantization contract.
 *
 * Generated by tools/convert_logmel_tflite_to_c_int8.py. Do not hand-edit.
 * Dimensions, normalization, scale/zero-point values, and class order must
 * match the exact flatbuffer embedded in model_data_int8.cc.
 */
#include <stdint.h>

#define AUDIO_MODEL_CLASS_COUNT 3
#define AUDIO_MODEL_INPUT_HEIGHT 32
#define AUDIO_MODEL_INPUT_WIDTH 124
#define AUDIO_MODEL_INPUT_CHANNELS 1
#define AUDIO_MODEL_INPUT_VALUES 3968

#define AUDIO_MODEL_FEATURE_LOGMEL 1
#define AUDIO_MODEL_FEATURE_LOGMEL_DELTA 2
#define AUDIO_MODEL_FEATURE_CMVN 3
#define AUDIO_MODEL_FEATURE_MODE AUDIO_MODEL_FEATURE_CMVN
#define AUDIO_MODEL_CMVN_STD_FLOOR_DB {format_float(float(config['cmvn']['std_floor_db']))}

{mel_band_c_block(mel_f_min, mel_f_max, mel_bin_edges)}
/* Real value = scale * (quantized value - zero point). */
#define AUDIO_MODEL_INPUT_SCALE {format_float(model_info['input_scale'])}
#define AUDIO_MODEL_INPUT_ZERO_POINT {model_info['input_zero_point']}
#define AUDIO_MODEL_OUTPUT_SCALE {format_float(model_info['output_scale'])}
#define AUDIO_MODEL_OUTPUT_ZERO_POINT {model_info['output_zero_point']}

/* Float frontend normalization performed before final INT8 quantization. */
static const float g_audio_input_mean[AUDIO_MODEL_INPUT_CHANNELS] = {{
    {format_float(means[0])}
}};
static const float g_audio_input_inv_std[AUDIO_MODEL_INPUT_CHANNELS] = {{
    {format_float(float(inv_std))}
}};
/* Output tensor index-to-label mapping generated from training classes. */
static const char *const g_audio_class_names[AUDIO_MODEL_CLASS_COUNT] = {{
{classes}
}};
""",
        encoding="ascii",
        newline="\n",
    )
    summary = {
        "model": str(model_path),
        "config": str(config_path),
        "model_size_bytes": len(data),
        "model_sha256": hashlib.sha256(data).hexdigest(),
        "model_info": model_info,
        "mel_f_min_hz": mel_f_min,
        "mel_f_max_hz": mel_f_max,
        "mel_bin_edges": mel_bin_edges,
        "normalization_mean": means,
        "normalization_std": stds,
        "class_names": class_names,
    }
    (args.out_dir / "logmel_conversion_summary_int8.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
