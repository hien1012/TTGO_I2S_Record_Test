#!/usr/bin/env python3
"""Does moving Xu's MFCC-DNN from 44.1 kHz to 16 kHz change its accuracy?

The same 44.1 kHz recordings are run twice with Xu's pipeline; nothing else differs
(same files, same train/val split per seed, same model, augmentation and test segments):
  xu44k : native 44.1 kHz, notebook settings (mel 0-22.05 kHz, band-pass 50-13500 Hz)
  xu16k : the same WAVs resampled to 16 kHz (resample_poly), the 16 kHz port used in
          run_mic_loo_comparison.py (mel 0-8 kHz, band-pass 50-7600 Hz)

Data (44.1 kHz, three mics, A/B sessions like raw_audio/new_16kHz):
  A = raw_audio/pc_train/{ics_2,inmp_2,pdm_2}   (42 x 6 s per mic)
  B = raw_audio/pc_test/{ics_1,inmp_1,pdm_1}    (43 x 10 s per mic)
Train on A of one mic, test on B of every mic -> 9 cells x seeds per sample rate.

The report pairs every (seed, train mic, test mic) cell across the two rates and gives
the mean difference with a 95 % t-interval, a Wilcoxon signed-rank test, and the
segment-level prediction agreement.

  python run_xu_sample_rate_check.py --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np
from scipy import stats

from run_mic_loo_comparison import ARCH_LABELS, MICS, collect_rows, run_dir, write_summary

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_FOLDERS = {"ICS": "pc_train/ics_2", "INMP": "pc_train/inmp_2", "PDM": "pc_train/pdm_2"}
TEST_FOLDERS = {"ICS": "pc_test/ics_1", "INMP": "pc_test/inmp_1", "PDM": "pc_test/pdm_1"}
VARIANTS = {"xu44k": 44100, "xu16k": 16000}
ARCH_LABELS.update({"xu44k": "Xu MFCC26-DNN @ 44.1 kHz (original)", "xu16k": "Xu MFCC26-DNN @ 16 kHz (port)"})


def train(args, variant: str, seed: int, train_mic: str, out_dir: Path) -> None:
    data = Path(args.data_root)
    cmd = [
        sys.executable, str(REPO_ROOT / "train_xu_mfcc_dnn.py"),
        "--train-dir", str(data / TRAIN_FOLDERS[train_mic]),
        "--test-dir", *[str(data / TEST_FOLDERS[mic]) for mic in MICS],
        "--sample-rate", str(VARIANTS[variant]),
        "--random-state", str(seed),
        "--output-dir", str(out_dir),
    ]
    print(f"\n>>> [{variant}] seed={seed} train={train_mic}\n" + " ".join(cmd), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with open(out_dir / "train_log.txt", "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"{variant} train={train_mic} seed={seed} failed; see {out_dir / 'train_log.txt'}")


def test_predictions(out_dir: Path, test_key: str) -> Dict[str, int]:
    """Segment -> predicted class of the best-val-loss model (train_xu_mfcc_dnn test_predictions.csv)."""
    with open(out_dir / "test_predictions.csv", encoding="utf-8") as handle:
        return {
            row["relative_path"]: int(row["pred"])
            for row in csv.DictReader(handle)
            if row["relative_path"].split("/")[0] == test_key
        }


def paired_report(root: Path, rows: List[Dict], seeds: List[int], test_keys: Dict[str, str]) -> List[str]:
    lines = ["## Paired 44.1 kHz vs 16 kHz comparison", ""]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        index = {
            (r["arch"], r["seed"], r["train_mic"], r["test_mic"]): r[metric]
            for r in rows
            if r["selection"] == "best_val_loss"
        }
        keys = [
            (seed, tr, te)
            for seed in seeds
            for tr in MICS
            for te in MICS
            if ("xu44k", seed, tr, te) in index and ("xu16k", seed, tr, te) in index
        ]
        if len(keys) < 2:
            continue
        a = np.array([index[("xu44k", *key)] for key in keys]) * 100
        b = np.array([index[("xu16k", *key)] for key in keys]) * 100
        diff = b - a
        ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff)) if diff.std() > 0 else (0.0, 0.0)
        wilcoxon_p = stats.wilcoxon(b, a).pvalue if np.any(diff != 0) else 1.0
        lines += [
            f"### {metric}",
            "",
            f"- pairs (seed x train mic x test mic): {len(keys)}",
            f"- 44.1 kHz mean {a.mean():.1f}, 16 kHz mean {b.mean():.1f}",
            f"- 16k - 44.1k: mean {diff.mean():+.2f} points, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], "
            f"median {np.median(diff):+.2f}, mean |diff| {np.abs(diff).mean():.2f}",
            f"- Wilcoxon signed-rank p = {wilcoxon_p:.3f}",
            "",
            "| train \\ test | " + " | ".join(MICS) + " |",
            "|---|" + "---|" * len(MICS),
        ]
        for tr in MICS:
            cells = []
            for te in MICS:
                d = [index[("xu16k", s, tr, te)] * 100 - index[("xu44k", s, tr, te)] * 100 for s in seeds
                     if ("xu16k", s, tr, te) in index and ("xu44k", s, tr, te) in index]
                cells.append(f"{mean(d):+.1f}" if d else "-")
            lines.append(f"| {tr} | " + " | ".join(cells) + " |")
        lines += ["", "(cells: 16k - 44.1k, mean over seeds)", ""]

    # Segment-level agreement: do the two sample rates predict the same class per segment?
    agree, total = 0, 0
    for seed in seeds:
        for tr in MICS:
            dirs = [run_dir(root, v, seed, tr) for v in VARIANTS]
            if not all((d / "metrics.json").exists() for d in dirs):
                continue
            for te in MICS:
                p44 = test_predictions(dirs[0], test_keys[te])
                p16 = test_predictions(dirs[1], test_keys[te])
                common = p44.keys() & p16.keys()
                agree += sum(p44[k] == p16[k] for k in common)
                total += len(common)
    if total:
        lines += [f"Segment-level prediction agreement 44.1k vs 16k: {agree}/{total} = {agree / total * 100:.1f}%", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--data-root", default=str(REPO_ROOT / "raw_audio"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "cnn_output" / "xu_sample_rate_check"))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    if not args.summary_only:
        for seed in args.seeds:
            for variant in VARIANTS:
                for train_mic in MICS:
                    out_dir = run_dir(root, variant, seed, train_mic)
                    if (out_dir / "metrics.json").exists() and not args.rerun:
                        print(f"skip existing: {out_dir}")
                        continue
                    train(args, variant, seed, train_mic, out_dir)

    # Test folders are named ics_1 / inmp_1 / pdm_1; test_by_folder keys use the folder name.
    test_keys = {mic: Path(TEST_FOLDERS[mic]).name for mic in MICS}
    rows = collect_rows(root, list(VARIANTS), args.seeds, test_mics=test_keys)
    if rows:
        write_summary(
            root,
            rows,
            title="Xu MFCC26-DNN: 44.1 kHz original vs 16 kHz port on the same recordings",
            arch_order=list(VARIANTS),
            extra_lines=paired_report(root, rows, args.seeds, test_keys),
        )


if __name__ == "__main__":
    main()
