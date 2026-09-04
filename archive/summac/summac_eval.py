#!/usr/bin/env python3
"""
Standalone SummaC factual consistency evaluation.

Runs in a separate conda environment with pinned transformers==4.35.2
to avoid compatibility issues with the main eval environment.

Loads pre-generated summaries from eval_generations.json and computes
SummaC scores (source document vs. generated summary).

Usage:
    python summac_eval.py \
        --generations /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/eval_generations.json \
        --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
        --test_ids /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/test_paper_ids.json \
        --output /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/summac_results.json
"""

import argparse
import json
import sys
import logging
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser(description="Standalone SummaC evaluation")
    parser.add_argument("--generations", required=True,
                        help="Path to eval_generations.json")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset.jsonl (for source texts)")
    parser.add_argument("--test_ids", required=True,
                        help="Path to test_paper_ids.json")
    parser.add_argument("--output", required=True,
                        help="Path to save SummaC results JSON")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for SummaC scoring")
    parser.add_argument("--max_source_words", type=int, default=2000,
                        help="Truncate source documents to this many words "
                             "(SummaC's NLI model has a 512-token window, "
                             "but sentence splitting works better with more context)")
    args = parser.parse_args()

    # Load data
    log.info("Loading data...")
    generations = load_json(args.generations)
    test_meta = load_json(args.test_ids)
    test_id_to_domain = {item["paper_id"]: item["domain"] for item in test_meta}

    dataset = load_jsonl(args.dataset)
    dataset_by_id = {r["paper_id"]: r for r in dataset}

    log.info(f"Models in generations: {list(generations.keys())}")

    # Load SummaC
    log.info("Loading SummaC model...")
    try:
        from summac.model_summac import SummaCZS
    except ImportError:
        log.error("SummaC not installed. Run: pip install summac")
        sys.exit(1)

    model = SummaCZS(
        model_name="vitc",
        granularity="sentence",
        device="cuda",
    )
    log.info("SummaC model loaded.")

    # Evaluate each model
    all_results = {}

    for label, summaries in generations.items():
        log.info(f"\n{'='*60}")
        log.info(f"SummaC evaluation: {label}")
        log.info(f"{'='*60}")

        # Build aligned lists of (source, summary) pairs
        paper_ids = []
        sources = []
        preds = []
        domains = []

        for pid in sorted(summaries.keys()):
            if not summaries[pid]:  # skip empty
                continue
            if pid not in dataset_by_id:
                continue
            if pid not in test_id_to_domain:
                continue

            source_text = dataset_by_id[pid]["input_text"]
            # Truncate source to manageable length
            words = source_text.split()
            if len(words) > args.max_source_words:
                source_text = " ".join(words[:args.max_source_words])

            paper_ids.append(pid)
            sources.append(source_text)
            preds.append(summaries[pid])
            domains.append(test_id_to_domain[pid])

        log.info(f"  Scoring {len(preds)} examples...")

        # Score in batches to avoid OOM
        all_scores = []
        bs = args.batch_size
        for i in range(0, len(preds), bs):
            batch_sources = sources[i:i+bs]
            batch_preds = preds[i:i+bs]
            try:
                result = model.score(batch_sources, batch_preds)
                all_scores.extend(result["scores"])
            except Exception as e:
                log.warning(f"  Batch {i//bs} failed: {e}")
                all_scores.extend([float('nan')] * len(batch_preds))

            if (i + bs) % 20 == 0 or (i + bs) >= len(preds):
                log.info(f"  {min(i+bs, len(preds))}/{len(preds)} scored")

        # Filter out NaN scores
        valid_scores = [(pid, s, d) for pid, s, d in zip(paper_ids, all_scores, domains)
                        if not np.isnan(s)]
        log.info(f"  Valid scores: {len(valid_scores)}/{len(all_scores)}")

        if not valid_scores:
            log.warning(f"  No valid scores for {label}")
            continue

        # Overall
        scores_array = np.array([s for _, s, _ in valid_scores])
        model_results = {
            "overall": {
                "mean": float(scores_array.mean()),
                "std": float(scores_array.std()),
                "median": float(np.median(scores_array)),
                "n": len(scores_array),
            },
            "per_example": {pid: float(s) for pid, s, _ in valid_scores},
        }

        # Per-domain
        domain_groups = defaultdict(list)
        for pid, s, d in valid_scores:
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

        # Print summary
        log.info(f"  Overall: {model_results['overall']['mean']:.4f} "
                 f"(+-{model_results['overall']['std']:.4f}, "
                 f"n={model_results['overall']['n']})")
        for domain in ["cs", "physics", "math"]:
            if domain in model_results:
                log.info(f"  {domain:>8}: {model_results[domain]['mean']:.4f} "
                         f"(+-{model_results[domain]['std']:.4f}, "
                         f"n={model_results[domain]['n']})")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  SUMMAC FACTUAL CONSISTENCY SCORES")
    print(f"{'='*80}")
    labels = list(all_results.keys())
    header = f"  {'Scope':<12}"
    for label in labels:
        header += f" {label:>18}"
    print(header)
    print(f"  {'-'*76}")

    for scope in ["overall", "cs", "physics", "math"]:
        row = f"  {scope:<12}"
        for label in labels:
            if scope in all_results.get(label, {}):
                val = all_results[label][scope]["mean"]
                row += f" {val:>18.4f}"
            else:
                row += f" {'-':>18}"
        print(row)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {args.output}")

    # Significance test if both finetuned and base are present
    ft_key = [k for k in all_results if "finetuned" in k]
    base_key = [k for k in all_results if "base" in k]
    if ft_key and base_key:
        ft_key, base_key = ft_key[0], base_key[0]
        ft_scores = all_results[ft_key]["per_example"]
        base_scores = all_results[base_key]["per_example"]
        common = sorted(set(ft_scores.keys()) & set(base_scores.keys()))
        if common:
            from scipy import stats as sp_stats
            ft_arr = np.array([ft_scores[pid] for pid in common])
            base_arr = np.array([base_scores[pid] for pid in common])
            stat, p_val = sp_stats.wilcoxon(ft_arr, base_arr)
            diff = ft_arr.mean() - base_arr.mean()
            print(f"\n  Paired comparison ({ft_key} vs {base_key}):")
            print(f"    Mean diff: {diff:+.4f}")
            print(f"    Wilcoxon p-value: {p_val:.6f}")
            print(f"    {'Significant (p<0.05)' if p_val < 0.05 else 'Not significant'}")
    print()


if __name__ == "__main__":
    main()
