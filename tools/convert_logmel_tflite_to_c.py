#!/usr/bin/env python3
"""Embed a float32 log-mel CNN and its preprocessing constants in firmware."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from logmel_mel_band import mel_band_c_block, validate_mel_band


EXPECTED_OUTPUT_SHAPE = [1, 3]
SUPPORTED_FEATURE_MODES = {"logmel": 1, "logmel-delta": 2, "cmvn": 1}
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
        raise ValueError(f"non-finite preprocessing value: {value}")
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
    ops = [
        str(item["op_name"])
        for item in interpreter._get_ops_details()
        if str(item["op_name"]) != "DELEGATE"
    ]
    return {
        "input_shape": [int(value) for value in input_info["shape"]],
        "input_dtype": np.dtype(input_info["dtype"]).name,
        "output_shape": [int(value) for value in output_info["shape"]],
        "output_dtype": np.dtype(output_info["dtype"]).name,
        "ops": sorted(set(ops)),
    }


def validate_contract(
    config: dict,
    model_info: dict,
) -> tuple[list[float], list[float], list[str], str, float]:
    expected_config = {
        "target_sr": 16000,
        "segment_seconds": 2.0,
        "segment_samples": 32000,
        "n_fft": 512,
        "hop_length": 256,
        "n_mels": 32,
    }
    for key, expected in expected_config.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"config {key}={actual!r}, expected {expected!r}")

    feature_mode = str(config.get("input_channels", ""))
    if feature_mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(
            f"firmware supports input_channels={sorted(SUPPORTED_FEATURE_MODES)}, "
            f"got {feature_mode!r}"
        )
    channel_count = SUPPORTED_FEATURE_MODES[feature_mode]
    expected_feature_shape = [32, 124, channel_count]
    if config.get("input_shape") != expected_feature_shape:
        raise ValueError(
            f"config input_shape={config.get('input_shape')!r}, expected {expected_feature_shape!r} "
            f"for {feature_mode}"
        )

    cmvn_std_floor_db = 0.0
    if feature_mode == "cmvn":
        cmvn = config.get("cmvn", {})
        expected_cmvn = {
            "enabled": True,
            "mean_axis": "per_sample_per_mel_over_time",
            "variance_axis": "per_sample_per_mel_over_time",
        }
        for key, expected in expected_cmvn.items():
            if cmvn.get(key) != expected:
                raise ValueError(f"config cmvn.{key}={cmvn.get(key)!r}, expected {expected!r}")
        cmvn_std_floor_db = float(cmvn.get("std_floor_db", 0.0))
        if not math.isfinite(cmvn_std_floor_db) or cmvn_std_floor_db <= 0.0:
            raise ValueError("config cmvn.std_floor_db must be finite and greater than zero")

    compression = config.get("log_compression", {})
    if compression.get("reference") != "sample_max" or compression.get("top_db") != 80.0:
        raise ValueError("firmware requires sample_max reference and top_db=80")
    if compression.get("log_floor") != 1e-12:
        raise ValueError("firmware requires log_floor=1e-12")
    if config.get("dc_removal") is not True or config.get("rms_normalization") is not False:
        raise ValueError("firmware requires DC removal and no RMS normalization")

    means = [float(value) for value in config.get("normalization_mean", [])]
    stds = [float(value) for value in config.get("normalization_std", [])]
    class_names = [str(value) for value in config.get("class_names", [])]
    if (
        len(means) != channel_count
        or len(stds) != channel_count
        or any(value <= 0.0 for value in stds)
    ):
        raise ValueError(
            f"{feature_mode} normalization must contain {channel_count} positive-std channels"
        )
    if len(class_names) != 3:
        raise ValueError("model must contain exactly three class names")

    expected_model_shape = [1, *expected_feature_shape]
    if model_info["input_shape"] != expected_model_shape:
        raise ValueError(f"model input {model_info['input_shape']} != {expected_model_shape}")
    if model_info["output_shape"] != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(f"model output {model_info['output_shape']} != {EXPECTED_OUTPUT_SHAPE}")
    if model_info["input_dtype"] != "float32" or model_info["output_dtype"] != "float32":
        raise ValueError("this firmware version requires float32 model input/output")
    extra_ops = set(model_info["ops"]) - EXPECTED_OPS
    if extra_ops:
        raise ValueError(f"unsupported TFLite operators: {sorted(extra_ops)}")
    return means, stds, class_names, feature_mode, cmvn_std_floor_db


def write_model_files(out_dir: Path, data: bytes) -> None:
    (out_dir / "model_data.h").write_text(
        """#pragma once

/**
 * @file model_data.h
 * @brief Linkage contract for the generated float32 TFLite flatbuffer.
 *
 * Generated model bytes reside in Flash as a const, 16-byte-aligned array.
 * Replace this header and model_data.cc together; never edit the byte array.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Embedded TFLite flatbuffer and its exact byte length. */
extern const unsigned char g_audio_model_tflite[];
extern const size_t g_audio_model_tflite_len;

#ifdef __cplusplus
}
#endif
""",
        encoding="ascii",
        newline="\n",
    )
    (out_dir / "model_data.cc").write_text(
        f"""/*
 * Auto-generated float32 TFLite flatbuffer. The const array is aligned for
 * FlatBuffer/TFLite Micro access and remains in Flash; do not hand-edit.
 */
#include "model_data.h"

#include <stddef.h>

alignas(16) const unsigned char g_audio_model_tflite[] = {{
{format_bytes(data)}
}};

const size_t g_audio_model_tflite_len = {len(data)};
""",
        encoding="ascii",
        newline="\n",
    )


def write_preprocess_header(
    out_dir: Path,
    means: list[float],
    stds: list[float],
    class_names: list[str],
    feature_mode: str,
    cmvn_std_floor_db: float,
    mel_band_block: str,
) -> None:
    channel_count = SUPPORTED_FEATURE_MODES[feature_mode]
    input_values = 32 * 124 * channel_count
    inv_stds = [1.0 / (np.float32(value) + np.float32(1e-6)) for value in stds]
    means_text = ", ".join(format_float(value) for value in means)
    inv_stds_text = ", ".join(format_float(float(value)) for value in inv_stds)
    classes = ",\n".join(f'    "{name}"' for name in class_names)
    feature_mode_macro = {
        "logmel": "AUDIO_MODEL_FEATURE_LOGMEL",
        "logmel-delta": "AUDIO_MODEL_FEATURE_LOGMEL_DELTA",
        "cmvn": "AUDIO_MODEL_FEATURE_CMVN",
    }[feature_mode]
    (out_dir / "preprocess_config.h").write_text(
        f"""#pragma once

/**
 * @file preprocess_config.h
 * @brief Generated float32 preprocessing and tensor contract.
 *
 * Generated by tools/convert_logmel_tflite_to_c.py. Do not hand-edit values:
 * feature mode, dimensions, normalization, and class order must match the
 * exact model embedded in model_data.cc.
 */
#include <stdint.h>

#define AUDIO_MODEL_CLASS_COUNT 3
#define AUDIO_MODEL_INPUT_HEIGHT 32
#define AUDIO_MODEL_INPUT_WIDTH 124
#define AUDIO_MODEL_INPUT_CHANNELS {channel_count}
#define AUDIO_MODEL_INPUT_VALUES {input_values}

#define AUDIO_MODEL_FEATURE_LOGMEL 1
#define AUDIO_MODEL_FEATURE_LOGMEL_DELTA 2
#define AUDIO_MODEL_FEATURE_CMVN 3
#define AUDIO_MODEL_FEATURE_MODE {feature_mode_macro}
#define AUDIO_MODEL_CMVN_STD_FLOOR_DB {format_float(cmvn_std_floor_db)}

{mel_band_block}
/* Dataset-level normalization applied after per-example feature processing. */
static const float g_audio_input_mean[AUDIO_MODEL_INPUT_CHANNELS] = {{
    {means_text}
}};

static const float g_audio_input_inv_std[AUDIO_MODEL_INPUT_CHANNELS] = {{
    {inv_stds_text}
}};

/* Output tensor index-to-label mapping; order is model metadata. */
static const char *const g_audio_class_names[AUDIO_MODEL_CLASS_COUNT] = {{
{classes}
}};
""",
        encoding="ascii",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.resolve()
    config_path = (args.config or model_path.with_name("preprocess_config.json")).resolve()
    out_dir = args.out_dir.resolve()
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"model={model_path}, config={config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_info = inspect_model(model_path)
    means, stds, class_names, feature_mode, cmvn_std_floor_db = validate_contract(
        config, model_info
    )
    mel_f_min, mel_f_max, mel_bin_edges = validate_mel_band(config)
    model_data = model_path.read_bytes()

    out_dir.mkdir(parents=True, exist_ok=True)
    write_model_files(out_dir, model_data)
    write_preprocess_header(
        out_dir,
        means,
        stds,
        class_names,
        feature_mode,
        cmvn_std_floor_db,
        mel_band_c_block(mel_f_min, mel_f_max, mel_bin_edges),
    )
    summary = {
        "model": str(model_path),
        "config": str(config_path),
        "model_size_bytes": len(model_data),
        "model_sha256": hashlib.sha256(model_data).hexdigest(),
        "model_info": model_info,
        "input_channels": feature_mode,
        "cmvn_std_floor_db": cmvn_std_floor_db,
        "mel_f_min_hz": mel_f_min,
        "mel_f_max_hz": mel_f_max,
        "mel_bin_edges": mel_bin_edges,
        "normalization_mean": means,
        "normalization_std": stds,
        "class_names": class_names,
    }
    (out_dir / "logmel_conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
