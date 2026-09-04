#!/usr/bin/env python3
"""
Extractiveness vs. Abstractiveness analysis.

Measures how much each model copies from the source vs. generates novel text.
This explains why BARTScore-based factuality metrics favor more extractive models.

Metrics computed:
  1. Novel n-gram %: fraction of n-grams in the summary NOT found in source
     (higher = more abstractive)
  2. Extractive Fragment Coverage: fraction of summary words that are part of
     extractive fragments (longest common subsequences with source)
     (higher = more extractive / copy-paste)
  3. Extractive Fragment Density: average length of extractive fragments
     (higher = longer copied spans)
  4. Compression Ratio: source length / summary length
  5. Vocabulary diversity: unique words / total words (type-token ratio)

References:
  - Grusky et al. 2018 "Newsroom: A Dataset of 1.3M Summaries with Diverse
    Extractive Strategies" (defines coverage and density)
  - See et al. 2017 "Get To The Point" (novel n-gram analysis)

Usage:
    python abstractiveness_analysis.py \
        --generations /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/eval_generations.json \
        --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
        --summaries /scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl \
        --test_ids /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/test_paper_ids.json \
        --output /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/abstractiveness_results.json
"""

import argparse
import json
import logging
import re
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


def tokenize(text):
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r'\b\w+\b', text.lower())


# ---------------------------------------------------------------------------
# Novel n-gram percentage (See et al. 2017)
# ---------------------------------------------------------------------------
def novel_ngram_pct(summary_tokens, source_tokens, n):
    """
    Fraction of n-grams in the summary that do NOT appear in the source.
    Higher = more abstractive.
    """
    if len(summary_tokens) < n:
        return 0.0

    source_ngrams = set()
    for i in range(len(source_tokens) - n + 1):
        source_ngrams.add(tuple(source_tokens[i:i+n]))

    summary_ngrams = []
    for i in range(len(summary_tokens) - n + 1):
        summary_ngrams.append(tuple(summary_tokens[i:i+n]))

    if not summary_ngrams:
        return 0.0

    novel = sum(1 for ng in summary_ngrams if ng not in source_ngrams)
    return novel / len(summary_ngrams)


# ---------------------------------------------------------------------------
# Extractive fragments (Grusky et al. 2018)
# ---------------------------------------------------------------------------
def extractive_fragments(summary_tokens, source_tokens):
    """
    Find greedy extractive fragments — longest common substrings between
    summary and source, greedily extracted.

    Returns list of fragment lengths.
    """
    source_set = {}
    for i, token in enumerate(source_tokens):
        if token not in source_set:
            source_set[token] = []
        source_set[token].append(i)

    fragments = []
    j = 0  # position in summary
    while j < len(summary_tokens):
        best_len = 0
        token = summary_tokens[j]
        if token in source_set:
            for src_pos in source_set[token]:
                # Try to extend match
                k = 0
                while (j + k < len(summary_tokens) and
                       src_pos + k < len(source_tokens) and
                       summary_tokens[j + k] == source_tokens[src_pos + k]):
                    k += 1
                best_len = max(best_len, k)

        if best_len > 0:
            fragments.append(best_len)
            j += best_len
        else:
            j += 1

    return fragments


def coverage(fragments, summary_len):
    """
    Extractive fragment coverage: fraction of summary covered by
    extractive fragments. Higher = more extractive.
    """
    if summary_len == 0:
        return 0.0
    return sum(fragments) / summary_len


def density(fragments, summary_len):
    """
    Extractive fragment density: average squared fragment length
    normalized by summary length. Higher = longer copied spans.
    """
    if summary_len == 0:
        return 0.0
    return sum(f * f for f in fragments) / summary_len


# ---------------------------------------------------------------------------
# Per-example analysis
# ---------------------------------------------------------------------------
def analyze_example(summary_text, source_text):
    """Compute all abstractiveness metrics for one example."""
    sum_tokens = tokenize(summary_text)
    src_tokens = tokenize(source_text)

    if not sum_tokens or not src_tokens:
        return None

    # Novel n-grams
    novel_1 = novel_ngram_pct(sum_tokens, src_tokens, 1)
    novel_2 = novel_ngram_pct(sum_tokens, src_tokens, 2)
    novel_3 = novel_ngram_pct(sum_tokens, src_tokens, 3)
    novel_4 = novel_ngram_pct(sum_tokens, src_tokens, 4)

    # Extractive fragments
    frags = extractive_fragments(sum_tokens, src_tokens)
    cov = coverage(frags, len(sum_tokens))
    dens = density(frags, len(sum_tokens))

    # Compression ratio
    comp = len(src_tokens) / len(sum_tokens) if len(sum_tokens) > 0 else 0

    # Type-token ratio (vocabulary diversity)
    ttr = len(set(sum_tokens)) / len(sum_tokens) if sum_tokens else 0

    # Average sentence length (structural complexity proxy)
    sentences = [s.strip() for s in re.split(r'[.!?]+', summary_text) if s.strip()]
    avg_sent_len = np.mean([len(tokenize(s)) for s in sentences]) if sentences else 0

    return {
        "novel_1gram": novel_1,
        "novel_2gram": novel_2,
        "novel_3gram": novel_3,
        "novel_4gram": novel_4,
        "coverage": cov,
        "density": dens,
        "compression_ratio": comp,
        "type_token_ratio": ttr,
        "avg_sentence_length": float(avg_sent_len),
        "num_sentences": len(sentences),
        "summary_length": len(sum_tokens),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extractiveness vs. Abstractiveness analysis"
    )
    parser.add_argument("--generations", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--summaries", default=None,
                        help="Path to summaries.jsonl (teacher summaries)")
    parser.add_argument("--test_ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load data
    log.info("Loading data...")
    generations = load_json(args.generations)
    test_meta = load_json(args.test_ids)
    test_id_to_domain = {item["paper_id"]: item["domain"] for item in test_meta}

    dataset = load_jsonl(args.dataset)
    dataset_by_id = {r["paper_id"]: r for r in dataset}

    # Add teacher if available
    if args.summaries:
        teacher_records = load_jsonl(args.summaries)
        teacher_by_id = {}
        for r in teacher_records:
            if r.get("success", False):
                teacher_by_id[r["paper_id"]] = strip_teacher_prefix(
                    r["generated_summary"]
                )
        if "teacher" not in generations:
            teacher_gen = {
                pid: teacher_by_id[pid]
                for pid in test_id_to_domain
                if pid in teacher_by_id
            }
            if teacher_gen:
                generations["teacher"] = teacher_gen

    log.info(f"Models: {list(generations.keys())}")

    # Analyze each model
    all_results = {}

    for label, summaries in generations.items():
        log.info(f"\nAnalyzing: {label}")

        per_example = {}
        for pid in sorted(summaries.keys()):
            if not summaries[pid] or pid not in dataset_by_id:
                continue
            if pid not in test_id_to_domain:
                continue

            result = analyze_example(
                summaries[pid], dataset_by_id[pid]["input_text"]
            )
            if result:
                result["domain"] = test_id_to_domain[pid]
                per_example[pid] = result

        if not per_example:
            continue

        # Aggregate
        metrics = list(next(iter(per_example.values())).keys())
        metrics = [m for m in metrics if m != "domain"]

        agg = {"overall": {}, "per_example": per_example}
        values = {m: [per_example[pid][m] for pid in per_example] for m in metrics}

        for m in metrics:
            agg["overall"][m] = {
                "mean": float(np.mean(values[m])),
                "std": float(np.std(values[m])),
                "median": float(np.median(values[m])),
            }

        # Per-domain
        for domain in ["cs", "physics", "math"]:
            domain_pids = [
                pid for pid in per_example
                if per_example[pid]["domain"] == domain
            ]
            if domain_pids:
                agg[domain] = {}
                for m in metrics:
                    d_vals = [per_example[pid][m] for pid in domain_pids]
                    agg[domain][m] = {
                        "mean": float(np.mean(d_vals)),
                        "n": len(d_vals),
                    }

        all_results[label] = agg

    # Print comparison table
    labels = list(all_results.keys())
    key_metrics = [
        ("novel_1gram", "Novel unigrams %", True),
        ("novel_2gram", "Novel bigrams %", True),
        ("novel_3gram", "Novel trigrams %", True),
        ("novel_4gram", "Novel 4-grams %", True),
        ("coverage", "Extractive coverage", False),
        ("density", "Extractive density", False),
        ("compression_ratio", "Compression ratio", None),
        ("type_token_ratio", "Vocabulary diversity", True),
        ("avg_sentence_length", "Avg sentence length", None),
        ("summary_length", "Summary length (words)", None),
    ]

    print(f"\n{'='*88}")
    print(f"  EXTRACTIVENESS vs. ABSTRACTIVENESS ANALYSIS")
    print(f"  (Novel n-gram % = fraction NOT copied from source; higher = more abstractive)")
    print(f"  (Coverage/Density = extractive fragment metrics; lower = more abstractive)")
    print(f"{'='*88}")

    header = f"  {'Metric':<24}"
    for label in labels:
        header += f" {label:>16}"
    print(header)
    print(f"  {'─'*84}")

    for metric_key, display_name, higher_better in key_metrics:
        row = f"  {display_name:<24}"
        values_for_comparison = []
        for label in labels:
            val = all_results[label]["overall"][metric_key]["mean"]
            values_for_comparison.append((label, val))
            if metric_key in ("novel_1gram", "novel_2gram", "novel_3gram",
                              "novel_4gram", "coverage"):
                row += f" {val:>15.1%}"
            else:
                row += f" {val:>16.2f}"
        print(row)

    # Per-domain breakdown for key metrics
    for domain in ["cs", "physics", "math"]:
        print(f"\n  {domain.upper()}:")
        for metric_key, display_name, _ in key_metrics[:6]:
            row = f"    {display_name:<22}"
            for label in labels:
                if domain in all_results[label]:
                    val = all_results[label][domain][metric_key]["mean"]
                    if metric_key in ("novel_1gram", "novel_2gram", "novel_3gram",
                                      "novel_4gram", "coverage"):
                        row += f" {val:>15.1%}"
                    else:
                        row += f" {val:>16.2f}"
                else:
                    row += f" {'—':>16}"
            print(row)

    # Statistical significance for novel n-grams
    print(f"\n{'='*88}")
    print(f"  SIGNIFICANCE TESTS (Wilcoxon signed-rank, paired)")
    print(f"{'='*88}")

    ft_key = [k for k in all_results if "finetuned" in k]
    base_key = [k for k in all_results if "base" in k]

    if ft_key and base_key:
        ft_key, base_key = ft_key[0], base_key[0]
        ft_per = all_results[ft_key]["per_example"]
        base_per = all_results[base_key]["per_example"]
        common = sorted(set(ft_per.keys()) & set(base_per.keys()))

        from scipy import stats

        print(f"  {'Metric':<24} {'FT Mean':>9} {'Base Mean':>10} {'Diff':>8} {'p-value':>10} {'Sig':>6}")
        print(f"  {'─'*70}")

        for metric_key, display_name, higher_better in key_metrics[:8]:
            ft_vals = np.array([ft_per[pid][metric_key] for pid in common])
            base_vals = np.array([base_per[pid][metric_key] for pid in common])
            diff = ft_vals.mean() - base_vals.mean()
            try:
                _, p_val = stats.wilcoxon(ft_vals, base_vals)
            except ValueError:
                p_val = 1.0
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {display_name:<24} {ft_vals.mean():>9.4f} {base_vals.mean():>10.4f} "
                  f"{diff:>+8.4f} {p_val:>10.6f} {sig:>6}")

    print()

    # Save
    # Remove per_example for cleaner output file
    save_results = {}
    for label in all_results:
        save_results[label] = {
            k: v for k, v in all_results[label].items() if k != "per_example"
        }
    with open(args.output, "w") as f:
        json.dump(save_results, f, indent=2)
    log.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
