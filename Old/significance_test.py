#!/usr/bin/env python3
"""
Statistical significance testing for evaluation results.

Computes paired bootstrap tests and Wilcoxon signed-rank tests to determine
whether differences between fine-tuned and base model are significant.

Usage:
    python significance_test.py \
        --eval_results /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/eval_results.json \
        --generations /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/eval_generations.json \
        --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
        --summaries /scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl \
        --test_ids /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/test_paper_ids.json
"""

import argparse
import json
import sys
import numpy as np
from collections import defaultdict
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading (mirrors evaluate.py)
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
# Per-example metric computation
# ---------------------------------------------------------------------------
def compute_rouge_per_example(predictions, references):
    """Returns dict of {metric_name: [score_per_example]}."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in results:
            results[key].append(score[key].fmeasure)
    return results


def compute_bertscore_per_example(predictions, references):
    """Returns dict of {metric_name: [score_per_example]}."""
    # Patch the overflow bug
    import bert_score.utils as bsu
    def _safe_sent_encode(tokenizer, text, *args, **kwargs):
        return tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    bsu.sent_encode = _safe_sent_encode

    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(
        predictions, references, lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False, batch_size=16,
    )
    return {
        "bertscore_precision": P.tolist(),
        "bertscore_recall": R.tolist(),
        "bertscore_f1": F1.tolist(),
    }


def compute_bleu_per_example(predictions, references):
    """Returns dict of {metric_name: [score_per_example]}."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tok = [ref.split()]
        pred_tok = pred.split()
        scores.append(sentence_bleu(ref_tok, pred_tok, smoothing_function=smooth))
    return {"bleu_sentence": scores}


def compute_meteor_per_example(predictions, references):
    """Returns dict of {metric_name: [score_per_example]}."""
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    from nltk.tokenize import word_tokenize
    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = word_tokenize(ref)
        pred_tokens = word_tokenize(pred)
        scores.append(nltk_meteor([ref_tokens], pred_tokens))
    return {"meteor": scores}


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """
    Paired bootstrap test: what fraction of bootstrap samples show A > B?
    Returns p-value (probability that the observed difference is due to chance).
    """
    rng = np.random.RandomState(seed)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)
    observed_diff = scores_a.mean() - scores_b.mean()

    count = 0
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        boot_diff = scores_a[indices].mean() - scores_b[indices].mean()
        # Two-sided test: count when bootstrap diff has opposite sign
        if observed_diff >= 0 and boot_diff <= 0:
            count += 1
        elif observed_diff < 0 and boot_diff >= 0:
            count += 1

    p_value = (count + 1) / (n_bootstrap + 1)  # +1 for continuity correction
    return p_value, observed_diff


def run_significance_tests(scores_ft, scores_base):
    """Run paired bootstrap and Wilcoxon tests, return results dict."""
    scores_ft = np.array(scores_ft)
    scores_base = np.array(scores_base)

    mean_ft = float(scores_ft.mean())
    mean_base = float(scores_base.mean())
    diff = mean_ft - mean_base

    # Paired bootstrap
    p_bootstrap, _ = paired_bootstrap_test(scores_ft, scores_base)

    # Wilcoxon signed-rank test (non-parametric paired test)
    try:
        stat_w, p_wilcoxon = stats.wilcoxon(scores_ft, scores_base, alternative='two-sided')
    except ValueError:
        # All differences are zero
        p_wilcoxon = 1.0

    # Effect size: Cohen's d for paired samples
    diffs = scores_ft - scores_base
    cohens_d = float(diffs.mean() / (diffs.std(ddof=1) + 1e-10))

    return {
        "mean_finetuned": mean_ft,
        "mean_base": mean_base,
        "diff": diff,
        "p_bootstrap": p_bootstrap,
        "p_wilcoxon": p_wilcoxon,
        "cohens_d": cohens_d,
        "n": len(scores_ft),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Statistical significance testing")
    parser.add_argument("--generations", required=True, help="Path to eval_generations.json")
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--summaries", required=True, help="Path to summaries.jsonl")
    parser.add_argument("--test_ids", required=True, help="Path to test_paper_ids.json")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    generations = load_json(args.generations)
    test_meta = load_json(args.test_ids)
    test_id_to_domain = {item["paper_id"]: item["domain"] for item in test_meta}

    dataset = load_jsonl(args.dataset)
    dataset_by_id = {r["paper_id"]: r for r in dataset}

    teacher_records = load_jsonl(args.summaries)
    teacher_by_id = {}
    for r in teacher_records:
        if r.get("success", False):
            teacher_by_id[r["paper_id"]] = r

    # Find the fine-tuned and base model keys
    ft_key = [k for k in generations if "finetuned" in k]
    base_key = [k for k in generations if "base" in k]
    if not ft_key or not base_key:
        print(f"Need both finetuned and base in generations. Found: {list(generations.keys())}")
        sys.exit(1)
    ft_key, base_key = ft_key[0], base_key[0]

    # Build aligned lists (only papers present in both)
    common_ids = sorted(
        set(generations[ft_key].keys()) & set(generations[base_key].keys())
    )
    # Filter to non-empty outputs
    common_ids = [
        pid for pid in common_ids
        if generations[ft_key][pid] and generations[base_key][pid]
    ]
    print(f"Common non-empty papers: {len(common_ids)}")

    ft_preds = [generations[ft_key][pid] for pid in common_ids]
    base_preds = [generations[base_key][pid] for pid in common_ids]
    abstracts = [dataset_by_id[pid]["reference_summary"] for pid in common_ids]
    teacher_summs = [
        strip_teacher_prefix(teacher_by_id[pid]["generated_summary"])
        for pid in common_ids
    ]
    domains = [test_id_to_domain[pid] for pid in common_ids]

    # Compute per-example scores
    print("\nComputing per-example metrics...")

    for ref_label, references in [("vs_abstracts", abstracts), ("vs_teacher", teacher_summs)]:
        print(f"\n{'='*80}")
        print(f"  REFERENCE: {ref_label}")
        print(f"{'='*80}")

        print("  Computing ROUGE...")
        ft_rouge = compute_rouge_per_example(ft_preds, references)
        base_rouge = compute_rouge_per_example(base_preds, references)

        print("  Computing BERTScore...")
        ft_bert = compute_bertscore_per_example(ft_preds, references)
        base_bert = compute_bertscore_per_example(base_preds, references)

        print("  Computing BLEU...")
        ft_bleu = compute_bleu_per_example(ft_preds, references)
        base_bleu = compute_bleu_per_example(base_preds, references)

        print("  Computing METEOR...")
        ft_meteor = compute_meteor_per_example(ft_preds, references)
        base_meteor = compute_meteor_per_example(base_preds, references)

        # Combine all metrics
        all_ft = {**ft_rouge, **ft_bert, **ft_bleu, **ft_meteor}
        all_base = {**base_rouge, **base_bert, **base_bleu, **base_meteor}

        # Run significance tests — OVERALL
        print(f"\n  {'─'*78}")
        print(f"  OVERALL (n={len(common_ids)})")
        print(f"  {'─'*78}")
        print(f"  {'Metric':<22} {'FT Mean':>9} {'Base Mean':>10} {'Diff':>8} {'p(boot)':>9} {'p(wilcox)':>10} {'Cohen d':>9} {'Sig?':>6}")
        print(f"  {'─'*78}")

        for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1",
                        "bleu_sentence", "meteor"]:
            if metric not in all_ft:
                continue
            result = run_significance_tests(all_ft[metric], all_base[metric])
            sig = "***" if result["p_bootstrap"] < 0.001 else \
                  "**" if result["p_bootstrap"] < 0.01 else \
                  "*" if result["p_bootstrap"] < 0.05 else ""
            print(f"  {metric:<22} {result['mean_finetuned']:>9.4f} {result['mean_base']:>10.4f} "
                  f"{result['diff']:>+8.4f} {result['p_bootstrap']:>9.4f} "
                  f"{result['p_wilcoxon']:>10.4f} {result['cohens_d']:>9.3f} {sig:>6}")

        # Per-domain breakdown
        for domain in ["cs", "physics", "math"]:
            domain_mask = [i for i, d in enumerate(domains) if d == domain]
            if not domain_mask:
                continue

            print(f"\n  {'─'*78}")
            print(f"  {domain.upper()} (n={len(domain_mask)})")
            print(f"  {'─'*78}")
            print(f"  {'Metric':<22} {'FT Mean':>9} {'Base Mean':>10} {'Diff':>8} {'p(boot)':>9} {'p(wilcox)':>10} {'Cohen d':>9} {'Sig?':>6}")
            print(f"  {'─'*78}")

            for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1",
                            "bleu_sentence", "meteor"]:
                if metric not in all_ft:
                    continue
                ft_domain = [all_ft[metric][i] for i in domain_mask]
                base_domain = [all_base[metric][i] for i in domain_mask]
                result = run_significance_tests(ft_domain, base_domain)
                sig = "***" if result["p_bootstrap"] < 0.001 else \
                      "**" if result["p_bootstrap"] < 0.01 else \
                      "*" if result["p_bootstrap"] < 0.05 else ""
                print(f"  {metric:<22} {result['mean_finetuned']:>9.4f} {result['mean_base']:>10.4f} "
                      f"{result['diff']:>+8.4f} {result['p_bootstrap']:>9.4f} "
                      f"{result['p_wilcoxon']:>10.4f} {result['cohens_d']:>9.3f} {sig:>6}")

    # Summary length comparison
    print(f"\n{'='*80}")
    print(f"  SUMMARY LENGTH ANALYSIS")
    print(f"{'='*80}")
    ft_lengths = np.array([len(generations[ft_key][pid].split()) for pid in common_ids])
    base_lengths = np.array([len(generations[base_key][pid].split()) for pid in common_ids])
    stat, p_len = stats.wilcoxon(ft_lengths, base_lengths)
    print(f"  Fine-tuned avg: {ft_lengths.mean():.1f} words (std: {ft_lengths.std():.1f})")
    print(f"  Base avg:       {base_lengths.mean():.1f} words (std: {base_lengths.std():.1f})")
    print(f"  Teacher target: 150-250 words")
    print(f"  Fine-tuned in range: {np.sum((ft_lengths >= 150) & (ft_lengths <= 250))}/{len(ft_lengths)}")
    print(f"  Base in range:       {np.sum((base_lengths >= 150) & (base_lengths <= 250))}/{len(base_lengths)}")
    print(f"  Wilcoxon p-value for length difference: {p_len:.6f}")
    print()


if __name__ == "__main__":
    main()
