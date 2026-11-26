#!/usr/bin/env python3
"""Compare an infer_node audio debug dump with the PC reference, stage by stage.

On the node (INFER_AUDIO_DEBUG_DUMP=1) the first live window(s) after boot are dumped
to /sdcard/debug/out/w<window>_<epoch_ms>/: the window PCM (input.wav), the int8
features the production path fed to the CNN, every frontend stage recomputed from
that PCM, and every CNN tensor. Copy that `out` folder to the PC and run:

  python compare_infer_node_debug_dump.py --dump-dir E:\\debug\\out `
    --c-source-dir "..\\DID_NODE\\infer_node\\main\\audio"

PC references:
  * frontend: the same logmel_features_int8.c compiled with gcc
    (tools/logmel_c_reference_cli_int8.c dumps the same stages);
  * CNN: the same int8 flatbuffer (model_data_int8.cc) in TFLite with reference
    kernels and preserve_all_tensors, fed with the node's own int8 input so each
    layer is compared in isolation from the frontend.
The report names the first stage / layer where the node departs from the PC.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from run_c_logmel_int8_inference import REPO, load_model_bytes, load_pcm16, parse_contract

DEFAULT_C_SOURCE = REPO.parent / "DID_NODE" / "infer_node" / "main" / "audio"
TYPE_DTYPES = {"INT8": np.int8, "UINT8": np.uint8, "INT16": np.int16, "INT32": np.int32,
               "INT64": np.int64, "FLOAT32": np.float32, "BOOL": np.bool_}

# (device file, PC reference file, dtype, tolerance kind, tolerance)
FRONTEND_STAGES = [
    ("fe_dc_removed.f32", "dc_removed.bin", np.float32, "abs", 1e-6),
    ("fe_power.f32", "power.bin", np.float32, "rel", 1e-4),
    ("fe_mel_power.f32", "mel_power.bin", np.float32, "rel", 1e-4),
    ("fe_log_mel.f32", "log_mel.bin", np.float32, "abs", 1e-2),
    ("fe_delta.f32", "delta.bin", np.float32, "abs", 2e-2),
    ("fe_cmvn.f32", "cmvn.bin", np.float32, "abs", 1e-3),
    ("fe_normalized.f32", "normalized.bin", np.float32, "abs", 1e-3),
    ("fe_quantized.i8", "quantized_int8.bin", np.int8, "lsb", 1),
]


def build_reference(c_source_dir: Path, executable: Path) -> None:
    subprocess.run(
        ["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-I", str(c_source_dir),
         str(REPO / "tools" / "logmel_c_reference_cli_int8.c"),
         str(c_source_dir / "logmel_features_int8.c"), "-lm", "-o", str(executable)],
        check=True,
    )


def compare_arrays(device: np.ndarray, pc: np.ndarray, kind: str, tolerance: float) -> dict:
    if device.shape != pc.shape:
        return {"ok": False, "error": f"shape device={device.shape} pc={pc.shape}"}
    d = device.astype(np.float64)
    p = pc.astype(np.float64)
    diff = np.abs(d - p)
    result = {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "mismatch_count": int(np.count_nonzero(diff)),
        "values": int(diff.size),
        "first_mismatch_index": int(np.flatnonzero(diff)[0]) if np.any(diff) else None,
    }
    if kind == "rel":
        scale = float(np.max(np.abs(p))) or 1.0
        result["max_rel_to_peak"] = result["max_abs"] / scale
        result["ok"] = result["max_rel_to_peak"] <= tolerance
    elif kind == "lsb":
        result["ok"] = result["max_abs"] <= tolerance and result["mismatch_count"] <= 0.01 * diff.size
    else:
        result["ok"] = result["max_abs"] <= tolerance
    result["tolerance"] = f"{kind}<={tolerance:g}"
    return result


def read_index(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8") as handle:
        return {int(row["index"]): row for row in csv.DictReader(handle)}


def find_wav(stem_dir: Path, wav_dir: Path | None) -> Path | None:
    """Live dumps carry their own input.wav; otherwise look for <stem>.wav in --wav-dir."""
    if (stem_dir / "input.wav").is_file():
        return stem_dir / "input.wav"
    if wav_dir is not None:
        for path in wav_dir.glob("*"):
            if path.stem == stem_dir.name and path.suffix.lower() == ".wav":
                return path
    return None


def compare_one(stem_dir: Path, wav_dir: Path | None, executable: Path, model_data: bytes, contract: dict,
                work_dir: Path) -> dict:
    import tensorflow as tf

    stem = stem_dir.name
    meta = json.loads((stem_dir / "meta.json").read_text(encoding="utf-8"))
    report: dict = {"stem": stem, "device_meta": meta}
    wav_path = find_wav(stem_dir, wav_dir)
    if wav_path is None:
        report["error"] = f"no input.wav in {stem_dir} and no {stem}.wav in --wav-dir"
        return report
    report["wav"] = str(wav_path)

    # ---- frontend: PC build of the same C source on the same 2 s ----
    pcm, exact = load_pcm16(wav_path, contract["sample_rate"])
    if not exact:
        report["warning"] = "WAV needed conversion on the PC; the node only accepts 16 kHz mono PCM16"
    pc_dir = work_dir / stem
    pc_dir.mkdir(parents=True, exist_ok=True)
    pcm_path = pc_dir / "input_pcm16.bin"
    pcm[: contract["window_samples"]].astype("<i2").tofile(pcm_path)
    subprocess.run([str(executable), str(pcm_path), str(pc_dir)], check=True)

    stages = []
    first_frontend_failure = None
    for device_name, pc_name, dtype, kind, tolerance in FRONTEND_STAGES:
        device_file, pc_file = stem_dir / device_name, pc_dir / pc_name
        if not device_file.is_file() or not pc_file.is_file():
            continue
        result = compare_arrays(np.fromfile(device_file, dtype=dtype), np.fromfile(pc_file, dtype=dtype),
                                kind, tolerance)
        result["stage"] = device_name
        stages.append(result)
        if not result["ok"] and first_frontend_failure is None:
            first_frontend_failure = device_name
    prod_input = np.fromfile(stem_dir / "fe_quantized_prod.i8", dtype=np.int8)
    debug_input = np.fromfile(stem_dir / "fe_quantized.i8", dtype=np.int8)
    report["frontend"] = stages
    report["frontend_first_failure"] = first_frontend_failure
    report["device_prod_vs_debug_input_identical"] = bool(np.array_equal(prod_input, debug_input))

    # ---- CNN: same flatbuffer, fed with the node's own int8 input ----
    interpreter = tf.lite.Interpreter(
        model_content=model_data,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    out_scale, out_zero = output_detail["quantization"]

    def run(tensor: np.ndarray) -> np.ndarray:
        interpreter.set_tensor(input_detail["index"], tensor.reshape(input_detail["shape"]))
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"])[0]
        return (raw.astype(np.int32) - out_zero).astype(np.float32) * np.float32(out_scale)

    pc_input = np.fromfile(pc_dir / "quantized_int8.bin", dtype=np.int8)
    pc_probs_from_pc_input = run(pc_input)
    pc_probs_from_device_input = run(prod_input)  # leaves every tensor of this run readable

    index = read_index(stem_dir / "nn_index.csv")
    layers = []
    first_layer_failure = None
    for op in interpreter._get_ops_details():
        for tensor_index in op["outputs"]:
            row = index.get(int(tensor_index))
            entry = {"op_index": int(op["index"]), "op": op["op_name"], "tensor": int(tensor_index),
                     "name": row["name"] if row else ""}
            if row is None or not row["file"] or row["file"] == "const":
                entry.update({"ok": False, "error": "tensor not dumped by the node"})
            else:
                dtype = TYPE_DTYPES[row["type"]]
                device = np.fromfile(stem_dir / row["file"], dtype=dtype)
                pc = interpreter.get_tensor(int(tensor_index)).reshape(-1)
                entry.update(compare_arrays(device, pc, "lsb" if np.issubdtype(dtype, np.integer) else "abs",
                                            1 if np.issubdtype(dtype, np.integer) else 1e-4))
                entry["shape"] = row["shape"]
            layers.append(entry)
            if not entry["ok"] and first_layer_failure is None:
                first_layer_failure = f"op {entry['op_index']} {entry['op']} -> tensor {entry['tensor']}"
    report["cnn_layers"] = layers
    report["cnn_first_failure"] = first_layer_failure

    names = contract["class_names"]
    report["probabilities"] = {
        "device_live" if meta.get("source") == "live_capture" else "device_prod_interpreter": meta.get("prod_probs"),
        "device_dump_interpreter": meta.get("dump_probs"),
        "pc_cnn_on_device_input": [round(float(v), 6) for v in pc_probs_from_device_input],
        "pc_end_to_end": [round(float(v), 6) for v in pc_probs_from_pc_input],
    }
    live_meta = wav_path.with_suffix(".json")
    if wav_path.name != "input.wav" and live_meta.is_file():
        try:
            report["probabilities"]["device_live_capture"] = json.loads(live_meta.read_text(encoding="utf-8"))["probs"]
        except (KeyError, json.JSONDecodeError):
            pass
    report["predictions"] = {k: names[int(np.argmax(v))] for k, v in report["probabilities"].items() if v}

    if not report["device_prod_vs_debug_input_identical"]:
        mismatch = int(np.count_nonzero(prod_input != debug_input))
        verdict = (f"the int8 features the production path fed to the CNN differ from the node's own "
                   f"recomputation on the same PCM ({mismatch}/{prod_input.size} values): the live path "
                   f"corrupts or mixes up feature/PCM buffers")
        if first_frontend_failure:
            verdict += f"; recomputed frontend vs PC differs first at {first_frontend_failure}"
    elif first_frontend_failure:
        verdict = f"frontend differs first at {first_frontend_failure}"
    elif first_layer_failure:
        verdict = f"frontend matches; CNN differs first at {first_layer_failure}"
    elif meta.get("prod_probs") and np.max(np.abs(np.array(meta["prod_probs"]) - pc_probs_from_device_input)) > 0.005:
        verdict = "all dumped layers match, but the node's production interpreter output differs (memory planning?)"
    else:
        verdict = "node matches the PC at every stage and layer"
    if "device_live_capture" in report["probabilities"]:
        live = np.array(report["probabilities"]["device_live_capture"], dtype=float)
        prod = np.array(meta.get("prod_probs") or [np.nan] * len(names), dtype=float)
        if np.nanmax(np.abs(live - prod)) > 0.005:
            verdict += "; the live-capture result for this WAV does NOT reproduce in debug mode"
    report["verdict"] = verdict
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", type=Path, required=True,
                        help="Copy of /sdcard/debug/out (folder of <stem>/ dumps) or a single <stem>/ folder.")
    parser.add_argument("--wav-dir", type=Path, default=None,
                        help="Only for dumps without input.wav: folder holding <stem>.wav files.")
    parser.add_argument("--c-source-dir", type=Path, default=DEFAULT_C_SOURCE,
                        help="logmel_features_int8.c + headers used by the firmware.")
    parser.add_argument("--model", type=Path, default=None,
                        help="model_data_int8.cc or .tflite; defaults to model_data_int8.cc in --c-source-dir.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("infer_node_debug_comparison"))
    args = parser.parse_args()

    c_source_dir = args.c_source_dir.resolve()
    contract = parse_contract(c_source_dir)
    model_data = load_model_bytes((args.model or c_source_dir / "model_data_int8.cc").resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = output_dir / "logmel_c_reference_int8.exe"
    build_reference(c_source_dir, executable)

    dump_dir = args.dump_dir.resolve()
    stem_dirs = [dump_dir] if (dump_dir / "meta.json").is_file() else sorted(
        p for p in dump_dir.iterdir() if (p / "meta.json").is_file())
    if not stem_dirs:
        raise SystemExit(f"no <stem>/meta.json found under {dump_dir}")

    reports = []
    for stem_dir in stem_dirs:
        report = compare_one(stem_dir, args.wav_dir.resolve() if args.wav_dir else None, executable,
                             model_data, contract, output_dir / "pc_reference")
        reports.append(report)
        print(f"\n=== {report['stem']}")
        if "error" in report:
            print("  ERROR:", report["error"])
            continue
        for stage in report["frontend"]:
            extra = f" rel={stage['max_rel_to_peak']:.2e}" if "max_rel_to_peak" in stage else ""
            print(f"  {'OK ' if stage['ok'] else 'BAD'} {stage['stage']:22s} max_abs={stage['max_abs']:.3e}{extra} "
                  f"mismatch={stage['mismatch_count']}/{stage['values']} ({stage['tolerance']})")
        print(f"  node prod vs debug int8 input identical: {report['device_prod_vs_debug_input_identical']}")
        for layer in report["cnn_layers"]:
            if "error" in layer:
                print(f"  BAD op{layer['op_index']:02d} {layer['op']:18s} t{layer['tensor']:03d}: {layer['error']}")
            else:
                print(f"  {'OK ' if layer['ok'] else 'BAD'} op{layer['op_index']:02d} {layer['op']:18s} "
                      f"t{layer['tensor']:03d} {layer['shape']:>12s} max_diff={layer['max_abs']:g} "
                      f"mismatch={layer['mismatch_count']}/{layer['values']}")
        for key, probs in report["probabilities"].items():
            print(f"  {key:26s} {probs}  -> {report['predictions'].get(key)}")
        print("  VERDICT:", report["verdict"])

    (output_dir / "debug_dump_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nReport: {output_dir / 'debug_dump_report.json'}")


if __name__ == "__main__":
    main()
