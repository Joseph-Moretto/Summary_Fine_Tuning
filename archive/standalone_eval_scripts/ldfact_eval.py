#!/usr/bin/env python3
"""
LongDocFACTScore evaluation for scientific paper summarization.

Uses retrieval-based factual consistency scoring that handles documents
of any length — unlike SummaC/AlignScore/QuestEval which truncate to 512 tokens.

How it works:
  1. Splits each summary into sentences (claims)
  2. Uses sentence embeddings to find the most relevant passages in the source
  3. Applies BARTScore to each matched claim-passage pair
  4. Averages across claims for a per-summary score

This means the metric actually checks facts from the full paper, not just
the introduction.

Usage:
    python ldfact_eval.py \
        --generations /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/eval_generations.json \
        --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
        --test_ids /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/test_paper_ids.json \
        --output /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/ldfact_results.json
"""

import argparse
import json
import sys
import time
import logging
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# LongDocFACTScore computation
# ---------------------------------------------------------------------------
def compute_ldfactscore(sources, predictions, device="cuda"):
    """
    Compute LongDocFACTScore for each (source, prediction) pair.
    Returns list of per-example scores.
    """
    from longdocfactscore.ldfacts import LongDocFACTScore

    log.info("  Loading LongDocFACTScore model...")
    scorer = LongDocFACTScore(device=device)

    log.info(f"  Scoring {len(predictions)} examples...")
    t0 = time.time()

    # score_src_hyp_long expects lists
    try:
        scores = scorer.score_src_hyp_long(sources, predictions)
    except Exception as e:
        log.error(f"  Batch scoring failed: {e}")
        log.info("  Falling back to per-example scoring...")
        scores = []
        for i, (src, pred) in enumerate(zip(sources, predictions)):
            try:
                s = scorer.score_src_hyp_long([src], [pred])
                scores.append(s[0] if isinstance(s, list) else float(s))
            except Exception as ex:
                log.warning(f"  Example {i} failed: {ex}")
                scores.append(float("nan"))

            if (i + 1) % 10 == 0 or (i + 1) == len(predictions):
                elapsed = time.time() - t0
                log.info(f"    {i+1}/{len(predictions)} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    log.info(f"  LongDocFACTScore done in {elapsed:.1f}s")

    # Ensure we have a flat list of floats
    if isinstance(scores, (list, np.ndarray)):
        scores = [float(s) if not isinstance(s, float) else s for s in scores]
    else:
        scores = [float(scores)]

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LongDocFACTScore evaluation")
    parser.add_argument("--generations", required=True,
                        help="Path to eval_generations.json")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset.jsonl (for source texts)")
    parser.add_argument("--summaries", default=None,
                        help="Path to summaries.jsonl (for teacher summaries)")
    parser.add_argument("--test_ids", required=True,
                        help="Path to test_paper_ids.json")
    parser.add_argument("--output", required=True,
                        help="Path to save results JSON")
    parser.add_argument("--device", default="cuda",
                        help="Device for scoring model (cuda or cpu)")
    args = parser.parse_args()

    # Load data
    log.info("Loading data...")
    generations = load_json(args.generations)
    test_meta = load_json(args.test_ids)
    test_id_to_domain = {item["paper_id"]: item["domain"] for item in test_meta}

    dataset = load_jsonl(args.dataset)
    dataset_by_id = {r["paper_id"]: r for r in dataset}

    # Optionally add teacher summaries as a model to evaluate
    if args.summaries:
        teacher_records = load_jsonl(args.summaries)
        teacher_by_id = {}
        for r in teacher_records:
            if r.get("success", False):
                summary = r["generated_summary"]
                teacher_by_id[r["paper_id"]] = strip_teacher_prefix(summary)
        # Add teacher to generations if not already present
        if "teacher" not in generations:
            teacher_gen = {}
            for pid in test_id_to_domain:
                if pid in teacher_by_id:
                    teacher_gen[pid] = teacher_by_id[pid]
            if teacher_gen:
                generations["teacher"] = teacher_gen
                log.info(f"  Added teacher summaries ({len(teacher_gen)} examples)")

    log.info(f"Models in generations: {list(generations.keys())}")

    # Evaluate each model
    all_results = {}

    for label, summaries in generations.items():
        log.info(f"\n{'='*60}")
        log.info(f"LongDocFACTScore: {label}")
        log.info(f"{'='*60}")

        # Build aligned lists
        paper_ids = []
        sources = []
        preds = []
        domains = []

        for pid in sorted(summaries.keys()):
            if not summaries[pid]:
                continue
            if pid not in dataset_by_id:
                continue
            if pid not in test_id_to_domain:
                continue

            paper_ids.append(pid)
            sources.append(dataset_by_id[pid]["input_text"])
            preds.append(summaries[pid])
            domains.append(test_id_to_domain[pid])

        if not preds:
            log.warning(f"  No examples for {label}")
            continue

        # Compute scores
        scores = compute_ldfactscore(sources, preds, device=args.device)

        # Filter NaN
        valid = [(pid, s, d) for pid, s, d in zip(paper_ids, scores, domains)
                 if not np.isnan(s)]
        log.info(f"  Valid scores: {len(valid)}/{len(scores)}")

        if not valid:
            continue

        # Overall stats
        all_scores = np.array([s for _, s, _ in valid])
        model_results = {
            "overall": {
                "mean": float(all_scores.mean()),
                "std": float(all_scores.std()),
                "median": float(np.median(all_scores)),
                "min": float(all_scores.min()),
                "max": float(all_scores.max()),
                "n": len(all_scores),
            },
            "per_example": {pid: float(s) for pid, s, _ in valid},
        }

        # Per-domain
        domain_groups = defaultdict(list)
        for pid, s, d in valid:
            domain_groups[d].append(s)

        for domain in ["cs", "physics", "math"]:
            if domain in domain_groups:
                d_scores = np.array(domain_groups[domain])
                model_results[domain] = {
                    "mean": float(d_scores.mean()),
                    "std": float(d_scores.std()),
                    "median": float(np.median(d_scores)),
                    "n": len(d_scores),
                }

        all_results[label] = model_results

        log.info(f"  Overall: {model_results['overall']['mean']:.4f} "
                 f"(±{model_results['overall']['std']:.4f}, "
                 f"n={model_results['overall']['n']})")
        for domain in ["cs", "physics", "math"]:
            if domain in model_results:
                log.info(f"  {domain:>8}: {model_results[domain]['mean']:.4f} "
                         f"(±{model_results[domain]['std']:.4f}, "
                         f"n={model_results[domain]['n']})")

    # Print comparison table
    labels = list(all_results.keys())
    if labels:
        print(f"\n{'='*80}")
        print(f"  LONGDOCFACTSCORE — FACTUAL CONSISTENCY (retrieval-based)")
        print(f"  (Higher = more factually consistent with full source document)")
        print(f"{'='*80}")

        header = f"  {'Scope':<12}"
        for label in labels:
            header += f" {label:>18}"
        print(header)
        print(f"  {'─'*76}")

        for scope in ["overall", "cs", "physics", "math"]:
            row = f"  {scope:<12}"
            for label in labels:
                if scope in all_results.get(label, {}):
                    val = all_results[label][scope]["mean"]
                    row += f" {val:>18.4f}"
                else:
                    row += f" {'—':>18}"
            print(row)

    # Significance test
    ft_key = [k for k in all_results if "finetuned" in k]
    base_key = [k for k in all_results if "base" in k]
    if ft_key and base_key:
        ft_key, base_key = ft_key[0], base_key[0]
        ft_per = all_results[ft_key].get("per_example", {})
        base_per = all_results[base_key].get("per_example", {})
        common = sorted(set(ft_per.keys()) & set(base_per.keys()))

        if len(common) >= 10:
            from scipy import stats
            ft_arr = np.array([ft_per[pid] for pid in common])
            base_arr = np.array([base_per[pid] for pid in common])
            mask = ~(np.isnan(ft_arr) | np.isnan(base_arr))
            ft_arr, base_arr = ft_arr[mask], base_arr[mask]

            if len(ft_arr) >= 10:
                stat, p_val = stats.wilcoxon(ft_arr, base_arr)
                diff = ft_arr.mean() - base_arr.mean()
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                print(f"\n  Paired comparison ({ft_key} vs {base_key}):")
                print(f"    Mean diff: {diff:+.4f}")
                print(f"    Wilcoxon p-value: {p_val:.6f} ({sig})")

    print()

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
