#!/usr/bin/env python3
"""Cross-microphone comparison: Xu's MFCC-DNN vs log-mel CMVN CNN.

For each architecture and each training mic M in {ICS, INMP, PDM}:
  train on set A of M  (raw_audio/new_16kHz/<m>, 80/20 original-file train/val split)
  test  on set B of every mic (raw_audio/new_16kHz/test/{ICS,INMP,PDM})
-> 3 training mics x 3 test mics = 9 results per architecture.

Both architectures use the same seed, so they get the same train/val file split.

  python run_mic_loo_comparison.py                       # both architectures, seed 42
  python run_mic_loo_comparison.py --arch xu --seeds 1 2 3
  python run_mic_loo_comparison.py --summary-only        # re-collect existing runs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent
MICS = ["ICS", "INMP", "PDM"]
ARCH_LABELS = {"xu": "Xu MFCC26-DNN", "logmel": "log-mel CMVN CNN"}

# Same recipe as cnn_output/logmel_cmvn/0909_outdoor/*.
LOGMEL_RECIPE = [
    "--val-split", "0.2",
    "--max-duration", "2",
    "--augmentation-copies", "2",
    "--augmentation-gain-min-db", "-35",
    "--augmentation-gain-max-db", "6",
    "--augmentation-background-snr-min-db", "-10",
    "--augmentation-background-snr-max-db", "10",
    "--f-min", "250",
    "--f-max", "2500",
    "--epochs", "60",
    "--skip-tflite",
]


def run_dir(root: Path, arch: str, seed: int, train_mic: str) -> Path:
    return root / arch / f"seed{seed}" / f"train_{train_mic.lower()}"


def command(arch: str, args, seed: int, train_mic: str, out_dir: Path) -> List[str]:
    data = Path(args.data_root)
    train_dir = str(data / train_mic.lower())
    test_dirs = [str(data / "test" / mic) for mic in MICS]
    if arch == "xu":
        return [
            sys.executable, str(REPO_ROOT / "train_xu_mfcc_dnn.py"),
            "--train-dir", train_dir, "--test-dir", *test_dirs,
            "--random-state", str(seed), "--output-dir", str(out_dir),
            *args.xu_extra,
        ]
    return [
        sys.executable, str(REPO_ROOT / "train_cnn_from_wav_logmel_cmvn.py"),
        "--train-dir", train_dir, "--test-dir", *test_dirs,
        *LOGMEL_RECIPE,
        "--random-state", str(seed), "--output-dir", str(out_dir),
        *args.logmel_extra,
    ]


def macro_f1(result: Dict) -> float:
    return float(result["classification_report"]["macro avg"]["f1-score"])


def collect_rows(root: Path, archs: List[str], seeds: List[int], test_mics: Dict[str, str] | None = None) -> List[Dict]:
    """test_mics maps MICS names to test_by_folder keys when the test folders are named differently."""
    test_mics = test_mics or {mic: mic for mic in MICS}
    rows = []
    for arch in archs:
        for seed in seeds:
            for train_mic in MICS:
                metrics_path = run_dir(root, arch, seed, train_mic) / "metrics.json"
                if not metrics_path.exists():
                    print(f"missing: {metrics_path}")
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                # Both report the best-val-loss checkpoint; log-mel also keeps its last epoch.
                selections = {"best_val_loss": metrics["test_by_folder"]} if arch.startswith("xu") else {
                    "best_val_loss": metrics["best_val_loss_model"]["test_by_folder"],
                    "last_epoch": metrics["test_by_folder"],
                }
                for selection, by_folder in selections.items():
                    for test_mic in MICS:
                        result = by_folder[test_mics[test_mic]]
                        rows.append(
                            {
                                "arch": arch,
                                "selection": selection,
                                "seed": seed,
                                "train_mic": train_mic,
                                "test_mic": test_mic,
                                "in_domain": train_mic == test_mic,
                                "segments": result["total_count"],
                                "accuracy": result["accuracy"],
                                "balanced_accuracy": result["balanced_accuracy"],
                                "macro_f1": macro_f1(result),
                                "confusion_matrix": json.dumps(result["confusion_matrix"]),
                            }
                        )
    return rows


def fmt(values: List[float]) -> str:
    if len(values) == 1:
        return f"{values[0] * 100:.1f}"
    return f"{mean(values) * 100:.1f} ± {stdev(values) * 100:.1f}"


def write_summary(
    root: Path,
    rows: List[Dict],
    title: str = "Cross-microphone comparison (train on A of one mic, test on B of each mic)",
    arch_order: List[str] | None = None,
    extra_lines: List[str] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "mic_loo_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {title}",
        "",
        "Values are percentages; with several seeds, mean ± std over seeds.",
        "",
        *(extra_lines or []),
    ]
    order = arch_order or ["xu", "logmel"]
    groups = sorted(
        {(row["arch"], row["selection"]) for row in rows},
        key=lambda key: (order.index(key[0]) if key[0] in order else len(order), key[1]),
    )
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        lines.append(f"## {metric}")
        lines.append("")
        for arch, selection in groups:
            subset = [row for row in rows if row["arch"] == arch and row["selection"] == selection]
            seeds = sorted({row["seed"] for row in subset})
            lines.append(f"### {ARCH_LABELS.get(arch, arch)} ({selection}, seeds={seeds})")
            lines.append("")
            lines.append("| train \\ test | " + " | ".join(MICS) + " | cross-mic mean |")
            lines.append("|---|" + "---|" * (len(MICS) + 1))
            for train_mic in MICS:
                cells = []
                for test_mic in MICS:
                    values = [r[metric] for r in subset if r["train_mic"] == train_mic and r["test_mic"] == test_mic]
                    cells.append(fmt(values) if values else "-")
                cross = [r[metric] for r in subset if r["train_mic"] == train_mic and not r["in_domain"]]
                lines.append(f"| {train_mic} | " + " | ".join(cells) + f" | {fmt(cross) if cross else '-'} |")
            same = [r[metric] for r in subset if r["in_domain"]]
            cross = [r[metric] for r in subset if not r["in_domain"]]
            if same and cross:
                lines.append("")
                lines.append(
                    f"Same-mic mean {mean(same) * 100:.1f}, cross-mic mean {mean(cross) * 100:.1f}, "
                    f"drop {(mean(same) - mean(cross)) * 100:.1f} points"
                )
            lines.append("")
    md_path = root / "summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {csv_path}\nSaved: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=["xu", "logmel", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--data-root", default=str(REPO_ROOT / "raw_audio" / "new_16kHz"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "cnn_output" / "mic_loo"))
    parser.add_argument("--train-mics", nargs="+", choices=MICS, default=MICS)
    parser.add_argument("--rerun", action="store_true", help="Retrain even if metrics.json exists.")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--xu-extra", nargs=argparse.REMAINDER, default=[], help="Extra args for train_xu_mfcc_dnn.py (put last).")
    parser.add_argument("--logmel-extra", nargs=argparse.REMAINDER, default=[], help="Extra args for the log-mel trainer (put last).")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    archs = ["xu", "logmel"] if args.arch == "both" else [args.arch]
    if not args.summary_only:
        for arch in archs:
            for seed in args.seeds:
                for train_mic in args.train_mics:
                    out_dir = run_dir(root, arch, seed, train_mic)
                    if (out_dir / "metrics.json").exists() and not args.rerun:
                        print(f"skip existing: {out_dir}")
                        continue
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cmd = command(arch, args, seed, train_mic, out_dir)
                    print(f"\n>>> [{arch}] seed={seed} train={train_mic}\n" + " ".join(cmd))
                    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
                    with open(out_dir / "train_log.txt", "w", encoding="utf-8") as log:
                        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                    if result.returncode != 0:
                        raise SystemExit(f"{arch} train={train_mic} seed={seed} failed; see {out_dir / 'train_log.txt'}")
    rows = collect_rows(root, archs, args.seeds)
    if rows:
        write_summary(root, rows)


if __name__ == "__main__":
    main()
