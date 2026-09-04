#!/usr/bin/env python3
"""
Per-summary extractive density and LongDocFACTScore, joined on paper_id.

Produces the per-summary values behind the density-vs-factuality analysis in
the paper (results/per_summary/). One CSV row per (system, paper) with columns

    system, scale, paper_id, domain, extractive_density, summary_length_words,
    longdocfactscore

where system is finetuned_<scale>, base_<scale> or teacher. Extractive density
follows Grusky et al. (2018) and matches the implementation in evaluate.py;
LongDocFACTScore is scored against the full paper text.

Run once per model scale and concatenate afterwards
(evaluation/run_extract_per_summary.slurm does both):

    python evaluation/extract_per_summary.py \
        --generations results/llama3.3-70b-qlora/eval/eval_generations.json \
        --scale 70B --output_dir results/per_summary

    python evaluation/extract_per_summary.py \
        --generations results/llama3.1-8b-lora/eval/eval_generations.json \
        --scale 8B --output_dir results/per_summary

--rename OLD=NEW relabels a system key in the generations file before
processing, for files that were written with a different --model_tag.
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Defaults are relative to the repository root so the script runs unchanged
# from a checkout; the SLURM script passes every path explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]


# ===================================================================== #
#                  DATA LOADING (same as in evaluate.py)                #
# ===================================================================== #

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_teacher_prefix(summary):
    if summary.startswith("## Summary"):
        summary = summary[len("## Summary"):].lstrip("\n").strip()
    return summary


def build_test_data(dataset_path, summaries_path, test_ids_path):
    test_meta = load_json(test_ids_path)
    test_id_to_domain = {item["paper_id"]: item["domain"] for item in test_meta}
    test_ids = set(test_id_to_domain.keys())
    log.info(f"Test set: {len(test_ids)} papers")

    dataset = load_jsonl(dataset_path)
    data_by_id = {r["paper_id"]: r for r in dataset}

    teacher_records = load_jsonl(summaries_path)
    teacher_by_id = {r["paper_id"]: r for r in teacher_records if r.get("success", False)}

    examples = []
    for pid in sorted(test_ids):
        if pid not in data_by_id or pid not in teacher_by_id:
            continue
        d = data_by_id[pid]
        t = teacher_by_id[pid]
        examples.append({
            "paper_id": pid,
            "title": d.get("title", ""),
            "input_text": d["input_text"],
            "abstract": d["reference_summary"],
            "teacher_summary": strip_teacher_prefix(t["generated_summary"]),
            "domain": test_id_to_domain[pid],
        })
    log.info(f"Loaded {len(examples)} test examples with all fields")
    return examples


# ===================================================================== #
#                  PER-SUMMARY EXTRACTIVENESS (Grusky et al. 2018)      #
# ===================================================================== #

def _tok(text):
    return re.findall(r'\b\w+\b', text.lower())


def _ext_frags(st, sr):
    """Greedy extractive fragments (Grusky et al. 2018)."""
    idx = defaultdict(list)
    for i, t in enumerate(sr):
        idx[t].append(i)
    frags, j = [], 0
    while j < len(st):
        bl = 0
        if st[j] in idx:
            for sp in idx[st[j]]:
                k = 0
                while j + k < len(st) and sp + k < len(sr) and st[j + k] == sr[sp + k]:
                    k += 1
                bl = max(bl, k)
        if bl > 0:
            frags.append(bl)
            j += bl
        else:
            j += 1
    return frags


def density_and_length(summary, source):
    """Return (extractive_density, summary_length_words) or (None, None)."""
    st, sr = _tok(summary), _tok(source)
    if not st or not sr:
        return None, None
    fr = _ext_frags(st, sr)
    density = sum(f * f for f in fr) / len(st)
    return density, len(st)


# ===================================================================== #
#                  PER-SUMMARY LONGDOCFACTSCORE                         #
# ===================================================================== #

def score_ldfact_per_summary(srcs, preds, device="cuda"):
    """Return list of float scores, one per (src, pred) pair. NaN on failure."""
    from longdocfactscore.ldfacts import LongDocFACTScore
    scorer = LongDocFACTScore(device=device)

    t0 = time.time()
    try:
        scores = scorer.score_src_hyp_long(srcs, preds)
        if not isinstance(scores, (list, np.ndarray)) or len(scores) != len(preds):
            raise ValueError(f"Unexpected return shape: {type(scores)}, "
                             f"len={len(scores) if hasattr(scores, '__len__') else '?'}")
        scores = [float(s) for s in scores]
    except Exception as e:
        log.warning(f"  Batch scoring failed: {e}. Falling back to per-example scoring.")
        scores = []
        for i, (s, p) in enumerate(zip(srcs, preds)):
            try:
                sc = scorer.score_src_hyp_long([s], [p])
                scores.append(float(sc[0]) if isinstance(sc, (list, np.ndarray)) else float(sc))
            except Exception as inner:
                log.error(f"    paper {i}: {inner}")
                scores.append(float("nan"))
            if (i + 1) % 25 == 0:
                log.info(f"    {i + 1}/{len(preds)} scored individually")
    log.info(f"  LongDocFACTScore complete in {time.time() - t0:.1f}s")
    return scores


# ===================================================================== #
#                              MAIN                                     #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True,
                        help="Path to eval_generations.json")
    parser.add_argument("--scale", required=True, choices=["8B", "70B"],
                        help="Model scale tag for the CSV output")
    parser.add_argument("--dataset",
                        default=str(REPO_ROOT / "data" / "dataset.jsonl"))
    parser.add_argument("--summaries",
                        default=str(REPO_ROOT / "data" / "summaries.jsonl"))
    parser.add_argument("--test_ids",
                        default=str(REPO_ROOT / "results" / "llama3.1-8b-lora"
                                    / "train" / "test_paper_ids.json"),
                        help="test_paper_ids.json from a training run "
                             "(identical for both reported runs)")
    parser.add_argument("--output_dir",
                        default=str(REPO_ROOT / "results" / "per_summary"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rename", action="append", default=[],
                        help="Rename a system key in the generations JSON before "
                             "processing. Format: OLD=NEW. Repeat the flag for "
                             "multiple renames. Example: "
                             "--rename finetuned_70B=finetuned_8B "
                             "--rename base_70B=base_8B")
    args = parser.parse_args()

    # Parse rename pairs into a dict
    renames = {}
    for r in args.rename:
        if "=" not in r:
            log.error(f"Invalid --rename value: {r!r} (expected OLD=NEW)")
            sys.exit(2)
        old, new = r.split("=", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            log.error(f"Invalid --rename value: {r!r}")
            sys.exit(2)
        renames[old] = new

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading test data...")
    examples = build_test_data(args.dataset, args.summaries, args.test_ids)
    if not examples:
        log.error("No test examples found.")
        sys.exit(1)
    eby = {e["paper_id"]: e for e in examples}

    log.info(f"Loading generations from {args.generations}")
    generations = load_json(args.generations)

    if renames:
        log.info(f"Applying renames: {renames}")
        for old, new in renames.items():
            if old not in generations:
                log.warning(f"  --rename {old}={new}: key '{old}' not found in "
                            f"generations. Available keys: {list(generations.keys())}")
                continue
            if new in generations:
                log.error(f"  --rename {old}={new}: target key '{new}' already "
                          f"exists in generations. Refusing to overwrite.")
                sys.exit(2)
            generations[new] = generations.pop(old)
            log.info(f"  Renamed '{old}' -> '{new}'")

    # Add teacher as a system, matching evaluate.py behavior
    generations["teacher"] = {ex["paper_id"]: ex["teacher_summary"] for ex in examples}

    log.info(f"Systems to process: {list(generations.keys())}")

    # ----- Compute per-summary density and length for every (system, paper) -----
    log.info("\n" + "=" * 60)
    log.info("Computing per-summary extractive density and length")
    log.info("=" * 60)

    rows = []  # accumulator: list of dicts, one per (system, paper_id)

    for system_label, sums in generations.items():
        n_valid = 0
        for pid in sorted(sums):
            summary = sums.get(pid)
            if not summary or pid not in eby:
                continue
            density, length = density_and_length(summary, eby[pid]["input_text"])
            if density is None:
                continue
            rows.append({
                "system": system_label,
                "scale": args.scale,
                "paper_id": pid,
                "domain": eby[pid]["domain"],
                "extractive_density": density,
                "summary_length_words": length,
                "longdocfactscore": float("nan"),  # filled in below
            })
            n_valid += 1
        log.info(f"  {system_label}: {n_valid} valid summaries")

    # ----- Compute LongDocFACTScore per summary, system by system -----
    log.info("\n" + "=" * 60)
    log.info("Computing per-summary LongDocFACTScore")
    log.info("=" * 60)

    # Index rows by (system, paper_id) so we can write LDFS back into them
    row_index = {(r["system"], r["paper_id"]): r for r in rows}

    for system_label, sums in generations.items():
        log.info(f"  Scoring system: {system_label}")
        pids, srcs, preds = [], [], []
        for pid in sorted(sums):
            summary = sums.get(pid)
            if not summary or pid not in eby:
                continue
            # Only score rows we kept (in case density returned None for any)
            if (system_label, pid) not in row_index:
                continue
            pids.append(pid)
            srcs.append(eby[pid]["input_text"])
            preds.append(summary)
        if not preds:
            log.warning(f"    No summaries to score for {system_label}")
            continue

        scores = score_ldfact_per_summary(srcs, preds, device=args.device)

        n_valid_scores = 0
        for pid, sc in zip(pids, scores):
            row_index[(system_label, pid)]["longdocfactscore"] = sc
            if not np.isnan(sc):
                n_valid_scores += 1
        log.info(f"    Wrote {n_valid_scores}/{len(scores)} valid scores")

    # ----- Save CSV -----
    out_path = output_dir / f"per_summary_{args.scale}.csv"
    log.info(f"\nWriting {len(rows)} rows to {out_path}")

    fieldnames = ["system", "scale", "paper_id", "domain",
                  "extractive_density", "summary_length_words", "longdocfactscore"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # ----- Quick sanity summary -----
    log.info("\n" + "=" * 60)
    log.info("Per-system sanity check (means)")
    log.info("=" * 60)
    by_system = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)
    for sys_label in sorted(by_system):
        rs = by_system[sys_label]
        d = np.array([r["extractive_density"] for r in rs])
        ldfs = np.array([r["longdocfactscore"] for r in rs])
        ldfs_valid = ldfs[~np.isnan(ldfs)]
        log.info(f"  {sys_label}: n={len(rs)}, "
                 f"density mean={d.mean():.2f}, "
                 f"LDFS mean={ldfs_valid.mean():.3f} (n_valid={len(ldfs_valid)})")

    log.info(f"\nDone. CSV: {out_path}")


if __name__ == "__main__":
    main()