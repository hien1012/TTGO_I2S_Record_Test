#!/usr/bin/env python3
"""Compare Python log-mel stages with the exact C frontend used by firmware."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from audio_dataset import read_wav_mono
from log_mel_features import create_mel_filterbank


SAMPLE_RATE = 16000
SAMPLE_COUNT = 32000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 32
FRAME_COUNT = 124
FFT_BINS = 257


def build_reference_cli(repo: Path, output: Path) -> None:
    command = [
        "gcc",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-I",
        str(repo / "main"),
        str(repo / "tools" / "logmel_c_reference_cli.c"),
        str(repo / "main" / "logmel_features.c"),
        "-lm",
        "-o",
        str(output),
    ]
    subprocess.run(command, cwd=repo, check=True)


def prepare_pcm16(wav_path: Path) -> np.ndarray:
    sr, audio = read_wav_mono(wav_path, SAMPLE_RATE)
    if sr != SAMPLE_RATE:
        raise ValueError(f"unexpected sample rate after resampling: {sr}")
    if len(audio) < SAMPLE_COUNT:
        audio = np.pad(audio, (0, SAMPLE_COUNT - len(audio)))
    else:
        audio = audio[:SAMPLE_COUNT]
    return np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")


def python_stages(pcm: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    samples = pcm.astype(np.float32) / np.float32(32767.0)
    centered = (samples - np.mean(samples, dtype=np.float64)).astype(np.float32)
    window = np.hamming(N_FFT).astype(np.float32)
    power = np.empty((FRAME_COUNT, FFT_BINS), dtype=np.float32)
    for frame_index in range(FRAME_COUNT):
        start = frame_index * HOP_LENGTH
        spectrum = np.fft.rfft(centered[start : start + N_FFT] * window, n=N_FFT)
        power[frame_index] = (
            spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        ).astype(np.float32)

    filters = create_mel_filterbank(
        SAMPLE_RATE,
        N_FFT,
        N_MELS,
        f_min=float(config.get("f_min_hz", 0.0)),
        f_max=float(config.get("f_max_hz", SAMPLE_RATE / 2.0)),
    )
    mel_power = (filters @ power.T).astype(np.float32)
    mel_power = np.maximum(mel_power, 0.0).astype(np.float32)
    log_mel = 10.0 * np.log10(np.maximum(mel_power, np.float32(1e-12)))
    log_mel -= np.max(log_mel)
    log_mel = np.maximum(log_mel, -80.0).astype(np.float32)
    delta = np.zeros_like(log_mel)
    delta[:, 1:] = log_mel[:, 1:] - log_mel[:, :-1]
    feature_mode = str(config.get("input_channels"))
    if feature_mode == "logmel":
        channels = log_mel[..., np.newaxis]
    elif feature_mode == "logmel-delta":
        channels = np.stack([log_mel, delta], axis=-1)
    elif feature_mode == "cmvn":
        cmn = log_mel - np.mean(log_mel, axis=1, keepdims=True, dtype=np.float32)
        cmvn_std = np.std(log_mel, axis=1, keepdims=True, dtype=np.float32)
        std_floor = np.float32(config["cmvn"]["std_floor_db"])
        cmvn = cmn / np.maximum(cmvn_std, std_floor)
        channels = cmvn[..., np.newaxis]
    else:
        raise ValueError(f"C comparison does not support input_channels={feature_mode!r}")
    channels = channels.astype(np.float32)
    mean = np.asarray(config["normalization_mean"], dtype=np.float32)
    std = np.asarray(config["normalization_std"], dtype=np.float32)
    normalized = ((channels - mean) / (std + np.float32(1e-6))).astype(np.float32)
    stages = {
        "dc_removed": centered,
        "power": power,
        "mel_power": mel_power,
        "log_mel": log_mel,
        "delta": delta,
        "normalized": normalized,
    }
    if feature_mode == "cmvn":
        stages["cmvn"] = cmvn.astype(np.float32)
    return stages


def read_c_stages(
    directory: Path,
    channel_count: int,
    feature_mode: str,
) -> dict[str, np.ndarray]:
    stages = {
        "dc_removed": np.fromfile(directory / "dc_removed.bin", dtype="<f4"),
        "power": np.fromfile(directory / "power.bin", dtype="<f4").reshape(
            FRAME_COUNT, FFT_BINS
        ),
        "mel_power": np.fromfile(directory / "mel_power.bin", dtype="<f4").reshape(
            N_MELS, FRAME_COUNT
        ),
        "log_mel": np.fromfile(directory / "log_mel.bin", dtype="<f4").reshape(
            N_MELS, FRAME_COUNT
        ),
        "delta": np.fromfile(directory / "delta.bin", dtype="<f4").reshape(
            N_MELS, FRAME_COUNT
        ),
        "normalized": np.fromfile(directory / "normalized.bin", dtype="<f4").reshape(
            N_MELS, FRAME_COUNT, channel_count
        ),
    }
    if feature_mode == "cmvn":
        stages["cmvn"] = np.fromfile(directory / "cmvn.bin", dtype="<f4").reshape(
            N_MELS, FRAME_COUNT
        )
    return stages


def error_metrics(reference: np.ndarray, actual: np.ndarray) -> dict:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    scale = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    return {
        "shape": list(reference.shape),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "p99_abs": float(np.quantile(np.abs(difference), 0.99)),
        "max_relative": float(np.max(np.abs(difference) / scale)),
    }


def tflite_predict(model_path: Path, tensor: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_info["index"], tensor[np.newaxis].astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_info["index"])[0].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("cnn_output/logmel_pdm_grouped/logmel_cnn_classifier_float32.tflite"),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("logmel_c_comparison"))
    parser.add_argument("--normalized-max-abs", type=float, default=0.02)
    parser.add_argument("--probability-max-abs", type=float, default=0.01)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    model_path = (repo / args.model).resolve() if not args.model.is_absolute() else args.model
    config_path = args.config or model_path.with_name("preprocess_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    channel_count = int(config["input_shape"][-1])
    feature_mode = str(config["input_channels"])
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pcm = prepare_pcm16(args.wav.resolve())
    pcm.tofile(output_dir / "input_pcm16.bin")
    python = python_stages(pcm, config)

    executable = output_dir / "logmel_c_reference.exe"
    build_reference_cli(repo, executable)
    subprocess.run(
        [str(executable), str(output_dir / "input_pcm16.bin"), str(output_dir)],
        cwd=repo,
        check=True,
    )
    c_stages = read_c_stages(output_dir, channel_count, feature_mode)
    stages = {name: error_metrics(python[name], c_stages[name]) for name in python}

    python_probability = tflite_predict(model_path, python["normalized"])
    c_probability = tflite_predict(model_path, c_stages["normalized"])
    probability_error = error_metrics(python_probability, c_probability)
    class_names = list(config["class_names"])
    passed = (
        stages["normalized"]["max_abs"] <= args.normalized_max_abs
        and probability_error["max_abs"] <= args.probability_max_abs
        and int(np.argmax(python_probability)) == int(np.argmax(c_probability))
    )
    report = {
        "wav": str(args.wav.resolve()),
        "model": str(model_path),
        "stages": stages,
        "python_probability": python_probability.astype(float).tolist(),
        "c_probability": c_probability.astype(float).tolist(),
        "probability_error": probability_error,
        "python_prediction": class_names[int(np.argmax(python_probability))],
        "c_prediction": class_names[int(np.argmax(c_probability))],
        "thresholds": {
            "normalized_max_abs": args.normalized_max_abs,
            "probability_max_abs": args.probability_max_abs,
        },
        "passed": passed,
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    for name, metrics in stages.items():
        print(
            f"{name:12s} max_abs={metrics['max_abs']:.8g} "
            f"rmse={metrics['rmse']:.8g} p99={metrics['p99_abs']:.8g}"
        )
    print("Python probability:", python_probability)
    print("C-input probability:", c_probability)
    print("Probability max abs:", probability_error["max_abs"])
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
