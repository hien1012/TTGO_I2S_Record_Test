#!/usr/bin/env python3
"""Run the firmware C log-mel frontend + int8 CNN on WAV files from the PC.

The frontend is the firmware's own logmel_features_int8.c compiled with gcc;
the CNN is the exact int8 flatbuffer embedded in model_data_int8.cc, executed
with TFLite reference kernels (the closest host match to TFLM reference kernels
used when ESP-NN is disabled). WAVs that are already 16 kHz mono PCM16 are fed
without any conversion, so device-saved windows are reproduced bit for bit.

Example (deployed infer_node sources and model):
  python run_c_logmel_int8_inference.py `
    --input "C:\\path\\to\\wav_folder" `
    --c-source-dir "..\\DID_NODE\\infer_node\\main\\audio" `
    -o c_int8_inference_output
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from audio_dataset import read_wav_mono
from evaluate_trained_audio_folders import infer_label_from_prefix


REPO = Path(__file__).resolve().parent
SCALE_TOLERANCE = 1.0e-7


def read_define(text: str, name: str) -> str:
    match = re.search(rf"#define\s+{name}\s+(\S+)", text)
    if match is None:
        raise ValueError(f"{name} not found in generated headers")
    return match.group(1).rstrip("fF")


def parse_contract(c_source_dir: Path) -> dict:
    """Read the frontend/model contract the firmware is compiled with."""
    config = (c_source_dir / "preprocess_config_int8.h").read_text(encoding="utf-8")
    frontend = (c_source_dir / "logmel_features_int8.h").read_text(encoding="utf-8")
    names = re.search(r"g_audio_class_names\[[^\]]*\]\s*=\s*\{(.*?)\};", config, re.S)
    if names is None:
        raise ValueError("g_audio_class_names not found in preprocess_config_int8.h")
    return {
        "class_names": re.findall(r'"([^"]+)"', names.group(1)),
        "input_scale": float(read_define(config, "AUDIO_MODEL_INPUT_SCALE")),
        "input_zero_point": int(read_define(config, "AUDIO_MODEL_INPUT_ZERO_POINT")),
        "output_scale": float(read_define(config, "AUDIO_MODEL_OUTPUT_SCALE")),
        "output_zero_point": int(read_define(config, "AUDIO_MODEL_OUTPUT_ZERO_POINT")),
        "mel_f_min_hz": float(read_define(config, "AUDIO_MODEL_MEL_F_MIN_HZ")),
        "mel_f_max_hz": float(read_define(config, "AUDIO_MODEL_MEL_F_MAX_HZ")),
        "sample_rate": int(read_define(frontend, "AUDIO_LOGMEL_SAMPLE_RATE_HZ")),
        "window_samples": int(read_define(frontend, "AUDIO_LOGMEL_INPUT_SAMPLES")),
        "input_shape": (
            int(read_define(frontend, "AUDIO_LOGMEL_N_MELS")),
            int(read_define(frontend, "AUDIO_LOGMEL_FRAME_COUNT")),
            int(read_define(config, "AUDIO_MODEL_INPUT_CHANNELS")),
        ),
    }


def load_model_bytes(model_path: Path) -> bytes:
    """Return the flatbuffer from a .tflite file or a generated model_data_int8.cc."""
    if model_path.suffix.lower() != ".cc":
        return model_path.read_bytes()
    text = model_path.read_text(encoding="ascii")
    body = re.search(r"\[\]\s*=\s*\{(.*?)\};", text, re.S)
    length = re.search(r"_len\s*=\s*(\d+)\s*;", text)
    if body is None or length is None:
        raise ValueError(f"no embedded model array found in {model_path}")
    data = bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", body.group(1)))
    if len(data) != int(length.group(1)):
        raise ValueError(f"{model_path}: array has {len(data)} bytes, _len says {length.group(1)}")
    return data


def build_frontend(c_source_dir: Path, executable: Path) -> None:
    subprocess.run(
        [
            "gcc", "-std=c11", "-O2", "-Wall", "-Wextra",
            "-I", str(c_source_dir),
            str(REPO / "tools" / "logmel_c_batch_cli_int8.c"),
            str(c_source_dir / "logmel_features_int8.c"),
            "-lm", "-o", str(executable),
        ],
        check=True,
    )


def collect_wavs(inputs: list[Path]) -> list[Path]:
    wavs: list[Path] = []
    for item in inputs:
        if item.is_dir():
            wavs.extend(sorted(p for p in item.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"))
        elif item.is_file() and item.suffix.lower() == ".wav":
            wavs.append(item)
        else:
            raise FileNotFoundError(f"not a WAV file or folder: {item}")
    return list(dict.fromkeys(path.resolve() for path in wavs))


def load_pcm16(path: Path, sample_rate: int) -> tuple[np.ndarray, bool]:
    """Return mono PCM16 at the model rate and whether no conversion was needed."""
    file_rate, data = wavfile.read(path)
    if data.dtype == np.int16 and data.ndim == 1 and file_rate == sample_rate:
        return data, True
    _, audio = read_wav_mono(path, sample_rate)
    return np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16), False


def split_windows(pcm: np.ndarray, window: int, hop: int, tail_policy: str):
    start = 0
    index = 0
    while start + window <= len(pcm):
        yield index, start, pcm[start : start + window]
        index += 1
        start += hop
    if tail_policy == "pad" and start < len(pcm):
        padded = np.zeros(window, dtype=np.int16)
        padded[: len(pcm) - start] = pcm[start:]
        yield index, start, padded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="WAV files and/or folders (recursive).")
    parser.add_argument(
        "--c-source-dir", type=Path, default=REPO / "main",
        help="Folder with logmel_features_int8.c/.h and preprocess_config_int8.h (e.g. infer_node/main/audio).",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="model_data_int8.cc or .tflite; defaults to model_data_int8.cc in --c-source-dir.",
    )
    parser.add_argument("--hop-seconds", type=float, default=None, help="Window step; default non-overlapping.")
    parser.add_argument("--tail-policy", choices=("drop", "pad"), default="drop")
    parser.add_argument("--save-features", action="store_true", help="Also save the int8 model inputs as .npy.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("c_int8_inference_output"))
    args = parser.parse_args()

    c_source_dir = args.c_source_dir.resolve()
    model_path = (args.model or c_source_dir / "model_data_int8.cc").resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = parse_contract(c_source_dir)
    class_names = contract["class_names"]
    sample_rate = contract["sample_rate"]
    window = contract["window_samples"]
    hop = window if args.hop_seconds is None else int(round(args.hop_seconds * sample_rate))
    if not 0 < hop <= window:
        raise ValueError("--hop-seconds must be > 0 and no longer than the model window")

    import tensorflow as tf

    model_data = load_model_bytes(model_path)
    interpreter = tf.lite.Interpreter(
        model_content=model_data,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    # Same guard as audio_model_init(): a stale header would corrupt every prediction.
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise ValueError("model must have int8 input and output")
    if tuple(input_detail["shape"][1:]) != contract["input_shape"]:
        raise ValueError(f"model input {input_detail['shape']} != header {contract['input_shape']}")
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]
    if (abs(in_scale - contract["input_scale"]) > SCALE_TOLERANCE or in_zero != contract["input_zero_point"]
            or abs(out_scale - contract["output_scale"]) > SCALE_TOLERANCE
            or out_zero != contract["output_zero_point"]):
        raise ValueError("preprocess_config_int8.h quantization does not match the model tensors")
    if int(output_detail["shape"][-1]) != len(class_names):
        raise ValueError("model output size does not match g_audio_class_names")

    wavs = collect_wavs(args.input)
    if not wavs:
        raise ValueError("no WAV files found")
    windows: list[np.ndarray] = []
    meta: list[dict] = []
    skipped: list[dict] = []
    converted_files = 0
    for wav in wavs:
        try:
            pcm, exact = load_pcm16(wav, sample_rate)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            skipped.append({"file": str(wav), "reason": str(exc)})
            continue
        converted_files += int(not exact)
        count = 0
        for index, start, samples in split_windows(pcm, window, hop, args.tail_policy):
            windows.append(samples)
            meta.append({"file": wav, "segment_index": index, "start": start, "exact_pcm": exact})
            count += 1
        if count == 0:
            skipped.append({"file": str(wav), "reason": f"shorter than {window / sample_rate:g}s"})
    if not windows:
        raise ValueError("no complete model windows were produced")

    executable = output_dir / "logmel_c_batch_int8.exe"
    build_frontend(c_source_dir, executable)
    pcm_path = output_dir / "input_windows_pcm16.bin"
    tensor_path = output_dir / "c_model_input_int8.bin"
    np.stack(windows).astype("<i2").tofile(pcm_path)
    subprocess.run([str(executable), str(pcm_path), str(tensor_path)], check=True)
    tensors = np.fromfile(tensor_path, dtype=np.int8).reshape(len(windows), *contract["input_shape"])
    pcm_path.unlink()
    if args.save_features:
        np.save(output_dir / "c_model_input_int8.npy", tensors)
    tensor_path.unlink()

    rows: list[dict] = []
    for item, tensor in zip(meta, tensors):
        interpreter.set_tensor(input_detail["index"], tensor[np.newaxis])
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0]
        # Matches audio_model_predict(): dequantize, first maximum wins.
        probabilities = (raw.astype(np.int32) - out_zero).astype(np.float32) * np.float32(out_scale)
        predicted = class_names[int(np.argmax(probabilities))]
        wav = item["file"]
        truth = infer_label_from_prefix(wav, class_names)
        row = {
            "file": str(wav),
            "filename": wav.name,
            "segment_index": item["segment_index"],
            "segment_start_s": item["start"] / sample_rate,
            "true_label": truth or "",
            "predicted_label": predicted,
            "correct": "" if truth is None else int(predicted == truth),
            "confidence": float(np.max(probabilities)),
            "exact_pcm": int(item["exact_pcm"]),
        }
        for name, value, quantized in zip(class_names, probabilities, raw):
            row[f"prob_{name}"] = float(value)
            row[f"raw_int8_{name}"] = int(quantized)
        rows.append(row)

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labeled = [row for row in rows if row["true_label"]]
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for row in labeled:
        confusion[class_names.index(row["true_label"]), class_names.index(row["predicted_label"])] += 1
    per_class = {
        name: {
            "segments": int(confusion[i].sum()),
            "recall": float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None,
        }
        for i, name in enumerate(class_names)
    }
    summary = {
        "c_source_dir": str(c_source_dir),
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_data).hexdigest(),
        "kernels": "tflite BUILTIN_REF",
        "mel_band_hz": [contract["mel_f_min_hz"], contract["mel_f_max_hz"]],
        "wav_files": len(wavs),
        "files_needing_conversion": converted_files,
        "segments": len(rows),
        "skipped": skipped,
        "prediction_counts": dict(Counter(row["predicted_label"] for row in rows)),
        "labeled_segments": len(labeled),
        "accuracy": float(np.trace(confusion) / len(labeled)) if labeled else None,
        "per_class": per_class,
        "class_names": class_names,
        "confusion_matrix": confusion.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")

    print(f"Model : {model_path} (sha256 {summary['model_sha256'][:16]})")
    print(f"C src : {c_source_dir}")
    print(f"WAVs  : {len(wavs)} ({converted_files} needed resampling/conversion), segments={len(rows)}, skipped={len(skipped)}")
    print(f"Predictions: {summary['prediction_counts']}")
    if labeled:
        print(f"Accuracy ({len(labeled)} labeled segments): {summary['accuracy']:.4f}")
        print("Confusion (rows=true, cols=pred):", class_names)
        for name, line in zip(class_names, confusion):
            print(f"  {name:>11s} {line.tolist()}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
