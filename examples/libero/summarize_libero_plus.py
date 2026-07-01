#!/usr/bin/env python3
"""Summarize the pi0.5-LIBERO LIBERO-Plus perturbation eval (Phase 2).

Reads the per-task JSONL shards produced by main_libero_plus.py (one record per
perturbed task: {suite, category, level, task, success, steps, ...}), dedups by
task name (last write wins, so resumes/overlaps are safe), and prints:
  * per-category success (n, successes, rate) and the delta vs pi0.5's OWN clean
    base success on this suite (from the Phase 1 base-gate) -- a within-model,
    within-harness brittleness measure;
  * a per-(category x level) breakdown;
  * an aggregate over all evaluated categories.

CROSS-HARNESS CAVEAT: a separate, clearly-labelled block prints the GR00T-RGB
LIBERO-Plus.Spatial reference numbers for context ONLY. pi0.5 ran in openpi;
GR00T ran in our GR00T harness. These are DIFFERENT harnesses -- do NOT read the
pi0.5-vs-GR00T gap as a controlled delta. Compare each model to its own clean base.

Usage:
  python summarize_libero_plus.py <results_dir_with_shard_jsonls> [--suite libero_spatial]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

# pi0.5 OWN clean-base success per suite, from the Phase 1 base-gate on S03
# (500 episodes/suite, seed=7): Spatial 98.6 / Object 97.4 / Goal 96.8 / Long 90.8.
CLEAN_BASE = {
    "libero_spatial": 98.6,
    "libero_object": 97.4,
    "libero_goal": 96.8,
    "libero_10": 90.8,
}

# GR00T-RGB LIBERO-Plus.Spatial reference (OUR GR00T harness) -- CONTEXT ONLY,
# cross-harness, NOT a controlled comparison. Fill in as more are confirmed.
GROOT_RGB_SPATIAL_CONTEXT = {
    "Camera Viewpoints": 65,
    "Robot Initial States": 42,
}


def load_records(results_dir):
    by_task = {}
    files = sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))
    if not files:
        # also allow pointing directly at results/ subdir
        files = sorted(glob.glob(os.path.join(results_dir, "results", "*.jsonl")))
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "task" in rec:
                    by_task[rec["task"]] = rec  # last wins
    return list(by_task.values()), files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--suite", default="libero_spatial")
    args = ap.parse_args()

    records, files = load_records(args.results_dir)
    print(f"Loaded {len(records)} unique-task records from {len(files)} shard file(s)")
    print(f"Suite: {args.suite}   pi0.5 clean base (this suite, own harness): "
          f"{CLEAN_BASE.get(args.suite, float('nan'))}%\n")

    cat_tot = defaultdict(int)
    cat_suc = defaultdict(int)
    cl_tot = defaultdict(int)
    cl_suc = defaultdict(int)
    errors = 0
    for rec in records:
        cat = rec.get("category", "?")
        lvl = str(rec.get("level", "?"))
        suc = 1 if rec.get("success") else 0
        cat_tot[cat] += 1
        cat_suc[cat] += suc
        cl_tot[(cat, lvl)] += 1
        cl_suc[(cat, lvl)] += suc
        if rec.get("error"):
            errors += 1

    base = CLEAN_BASE.get(args.suite)
    print("=== PER-CATEGORY (pi0.5, this harness) ===")
    print(f"{'category':<26}{'n':>6}{'succ':>6}{'rate%':>8}{'vs clean':>10}")
    print("-" * 56)
    all_tot = all_suc = 0
    for cat in sorted(cat_tot):
        n = cat_tot[cat]
        s = cat_suc[cat]
        rate = 100.0 * s / n if n else float("nan")
        all_tot += n
        all_suc += s
        delta = f"{rate - base:+.1f}" if base is not None else "--"
        print(f"{cat:<26}{n:>6}{s:>6}{rate:>8.1f}{delta:>10}")
    print("-" * 56)
    agg = 100.0 * all_suc / all_tot if all_tot else float("nan")
    agg_delta = f"{agg - base:+.1f}" if base is not None else "--"
    print(f"{'AGGREGATE (all axes)':<26}{all_tot:>6}{all_suc:>6}{agg:>8.1f}{agg_delta:>10}")
    if errors:
        print(f"\n[note] {errors} task(s) recorded an env/rollout error (counted as failures).")

    print("\n=== PER-CATEGORY x LEVEL ===")
    print(f"{'category':<26}{'lvl':>4}{'n':>6}{'succ':>6}{'rate%':>8}")
    print("-" * 50)
    for (cat, lvl) in sorted(cl_tot):
        n = cl_tot[(cat, lvl)]
        s = cl_suc[(cat, lvl)]
        rate = 100.0 * s / n if n else float("nan")
        print(f"{cat:<26}{lvl:>4}{n:>6}{s:>6}{rate:>8.1f}")

    if args.suite == "libero_spatial":
        print("\n=== CROSS-HARNESS CONTEXT (NOT a controlled comparison) ===")
        print("GR00T-RGB ran in OUR GR00T harness; pi0.5 above ran in openpi. Compare")
        print("each model to ITS OWN clean base -- the numbers below are context only.")
        print(f"{'category':<26}{'pi0.5%':>8}{'GR00T-RGB%':>12}")
        print("-" * 46)
        for cat in ["Camera Viewpoints", "Robot Initial States", "Objects Layout", "Language Instructions"]:
            if cat in cat_tot:
                p = 100.0 * cat_suc[cat] / cat_tot[cat]
                g = GROOT_RGB_SPATIAL_CONTEXT.get(cat)
                gs = f"{g}" if g is not None else "n/a"
                print(f"{cat:<26}{p:>8.1f}{gs:>12}")


if __name__ == "__main__":
    main()
