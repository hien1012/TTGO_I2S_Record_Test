#!/usr/bin/env python3
"""Cross-microphone comparison: per-clip CMVN vs long-term CMN inputs for the log-mel CNN.

Same protocol, recipe, seeds and train/val file split as run_mic_loo_comparison.py
(train on set A of one mic, test on set B of every mic); only --input-channels changes:
  cmvn          : baseline, per-clip per-band mean/std (re-used from cnn_output/mic_loo/logmel)
  ltcmn_bg      : clip level removed, minus the mic's long-term band profile from background clips
  ltcmn_all     : same, profile from all clips (no labels)
  cmvn_ltcmn_bg : two channels, cmvn + ltcmn_bg

The report pairs every (seed, train mic, test mic) cell with the baseline and gives the
mean difference with a 95 % t-interval and a Wilcoxon signed-rank test, overall and
separately for same-mic and cross-mic cells.

  python run_ltcmn_mic_loo_comparison.py --seeds 1 2 3 4 5
  python run_ltcmn_mic_loo_comparison.py --summary-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
from scipy import stats

from run_mic_loo_comparison import ARCH_LABELS, MICS, collect_rows, command, run_dir, write_summary

REPO_ROOT = Path(__file__).resolve().parent
BASELINE = "cmvn"
VARIANTS = {
    "ltcmn_bg": ["--input-channels", "ltcmn", "--ltcmn-reference", "background"],
    "ltcmn_all": ["--input-channels", "ltcmn", "--ltcmn-reference", "all"],
    "cmvn_ltcmn_bg": ["--input-channels", "cmvn-ltcmn", "--ltcmn-reference", "background"],
}
ARCH_LABELS.update(
    {
        BASELINE: "per-clip CMVN (baseline)",
        "ltcmn_bg": "long-term CMN, background profile",
        "ltcmn_all": "long-term CMN, all-clip profile",
        "cmvn_ltcmn_bg": "CMVN + long-term CMN (2 ch), background profile",
    }
)


def train(args, variant: str, seed: int, train_mic: str, out_dir: Path) -> None:
    runner_args = SimpleNamespace(data_root=args.data_root, logmel_extra=VARIANTS[variant])
    cmd = command("logmel", runner_args, seed, train_mic, out_dir)
    print(f"\n>>> [{variant}] seed={seed} train={train_mic}\n" + " ".join(cmd), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with open(out_dir / "train_log.txt", "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"{variant} train={train_mic} seed={seed} failed; see {out_dir / 'train_log.txt'}")


def paired_lines(label: str, a: np.ndarray, b: np.ndarray) -> List[str]:
    diff = b - a
    if len(diff) < 2:
        return [f"- {label}: too few pairs"]
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff)) if diff.std() > 0 else (0.0, 0.0)
    p = stats.wilcoxon(b, a).pvalue if np.any(diff != 0) else 1.0
    return [
        f"- {label} (n={len(diff)}): baseline {a.mean():.1f} -> {b.mean():.1f}, "
        f"diff {diff.mean():+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}], Wilcoxon p = {p:.3f}"
    ]


def paired_report(rows: List[Dict], seeds: List[int], variants: List[str]) -> List[str]:
    lines = [f"## Paired comparison against {ARCH_LABELS[BASELINE]} (best_val_loss)", ""]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        index = {
            (r["arch"], r["seed"], r["train_mic"], r["test_mic"]): r[metric] * 100
            for r in rows
            if r["selection"] == "best_val_loss"
        }
        lines += [f"### {metric}", ""]
        for variant in variants:
            keys = [
                (seed, tr, te)
                for seed in seeds
                for tr in MICS
                for te in MICS
                if (BASELINE, seed, tr, te) in index and (variant, seed, tr, te) in index
            ]
            if not keys:
                continue
            lines.append(f"**{ARCH_LABELS[variant]}**")
            lines.append("")
            for label, subset in (
                ("all cells", keys),
                ("same-mic", [k for k in keys if k[1] == k[2]]),
                ("cross-mic", [k for k in keys if k[1] != k[2]]),
            ):
                a = np.array([index[(BASELINE, *k)] for k in subset])
                b = np.array([index[(variant, *k)] for k in subset])
                lines += paired_lines(label, a, b)
            lines += ["", "| train \\ test | " + " | ".join(MICS) + " |", "|---|" + "---|" * len(MICS)]
            for tr in MICS:
                cells = []
                for te in MICS:
                    d = [index[(variant, s, tr, te)] - index[(BASELINE, s, tr, te)] for s in seeds
                         if (variant, s, tr, te) in index and (BASELINE, s, tr, te) in index]
                    cells.append(f"{mean(d):+.1f}" if d else "-")
                lines.append(f"| {tr} | " + " | ".join(cells) + " |")
            lines += ["", "(cells: variant - baseline, mean over seeds)", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "raw_audio" / "new_16kHz"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "cnn_output" / "mic_loo_ltcmn"))
    parser.add_argument(
        "--baseline-root",
        default=str(REPO_ROOT / "cnn_output" / "mic_loo"),
        help="run_mic_loo_comparison.py output root holding the CMVN baseline under logmel/.",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    if not args.summary_only:
        for seed in args.seeds:
            for variant in args.variants:
                for train_mic in MICS:
                    out_dir = run_dir(root, variant, seed, train_mic)
                    if (out_dir / "metrics.json").exists() and not args.rerun:
                        print(f"skip existing: {out_dir}")
                        continue
                    train(args, variant, seed, train_mic, out_dir)

    baseline_rows = collect_rows(Path(args.baseline_root).resolve(), ["logmel"], args.seeds)
    for row in baseline_rows:
        row["arch"] = BASELINE
    rows = baseline_rows + collect_rows(root, args.variants, args.seeds)
    if rows:
        write_summary(
            root,
            rows,
            title="Log-mel CNN: per-clip CMVN vs long-term CMN (train on A of one mic, test on B of each mic)",
            arch_order=[BASELINE, *args.variants],
            extra_lines=paired_report(rows, args.seeds, args.variants),
        )


if __name__ == "__main__":
    main()
