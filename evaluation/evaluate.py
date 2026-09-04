#!/usr/bin/env python3
"""
Evaluation pipeline for the summarization models.

Given a base model and a LoRA adapter, generates summaries for the held-out
test papers with the fine-tuned model and with the plain base model, then
scores every system (fine-tuned, base, and the teacher summaries themselves)
in four parts:

  1. Reference-based metrics   ROUGE-1/2/L, BERTScore, BLEU and METEOR against
                               the paper abstracts and against the teacher
                               summaries, overall and per domain
  2. Significance tests        fine-tuned vs. base on per-paper scores: paired
                               sign-flip permutation test, Wilcoxon signed-rank
                               test, Cohen's d
  3. LongDocFACTScore          retrieval-based factual consistency against the
                               full paper text (Bishop et al., 2023)
  4. Abstractiveness           novel n-gram rates, extractive fragment coverage
                               and density (Grusky et al., 2018)

Files written to --output_dir:
  eval_generations.json         {system: {paper_id: summary}}
  eval_results.json             all reference-metric scores
  eval_report.txt               the same numbers as a text table
  significance_results.txt      part 2
  ldfact_results.json           part 3
  abstractiveness_results.json  part 4

Usage:
    # Full pipeline: inference for both models, then all analyses
    python evaluation/evaluate.py \\
        --base_model meta-llama/Llama-3.1-8B-Instruct \\
        --adapter_path results/llama3.1-8b-lora/train/adapter \\
        --test_ids results/llama3.1-8b-lora/train/test_paper_ids.json \\
        --output_dir runs/eval-8b --model_tag 8B

    # 70B: load the base model in 4-bit
    python evaluation/evaluate.py ... --use_qlora --model_tag 70B

    # Re-score cached generations without loading a model
    python evaluation/evaluate.py --metrics_only --output_dir runs/eval-8b \\
        --test_ids results/llama3.1-8b-lora/train/test_paper_ids.json

    # Skip individual analyses
    python evaluation/evaluate.py ... --skip_ldfact --skip_significance --skip_abstractiveness

The system message and instruction below are the ones the reported results
were generated with. Their wording differs from the training prompt in
training/finetune_v2.py, and paper text is cut to the token budget from the
front only. evaluate_train_aligned.py is a later variant that reuses the
training prompt and the training-time head-and-tail truncation; it was not
used for the numbers in the paper.
"""

import argparse
import json
import re
import sys
import time
import logging
from io import StringIO
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Defaults are relative to the repository root so the script runs unchanged
# from a checkout; the SLURM scripts pass every path explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_nltk_data():
    """Download the NLTK resources BLEU and METEOR need, if they are missing."""
    import nltk
    for resource in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

# Prompt used for both the fine-tuned and the base model at evaluation time.
SYSTEM_MESSAGE = (
    "You are an expert scientific summarizer. You produce concise, accurate "
    "summaries of scientific papers that capture the key contributions, "
    "methodology, results, and significance."
)
INSTRUCTION = (
    "Summarize the following scientific paper in approximately 150-250 words. "
    "Focus on:\n"
    "1. The main contribution or finding\n"
    "2. The methodology or approach used\n"
    "3. Key results\n"
    "4. Significance or implications\n\n"
    "Write in clear, technical language appropriate for a scientific audience."
)

DOMAINS = ["cs", "physics", "math"]


# ===================================================================== #
#                              DATA LOADING                             #
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
    """Remove the '## Summary' header line some teacher summaries start with."""
    if summary.startswith("## Summary"):
        summary = summary[len("## Summary"):].lstrip("\n").strip()
    return summary


def build_test_data(dataset_path, summaries_path, test_ids_path):
    """Assemble the test examples: paper text, abstract, teacher summary, domain.

    test_ids_path is the test_paper_ids.json written by the training script.
    Papers without a successful teacher summary are skipped.
    """
    test_meta = load_json(test_ids_path)
    test_id_to_domain = {item["paper_id"]: item["domain"] for item in test_meta}
    test_ids = set(test_id_to_domain.keys())
    log.info(f"Test set: {len(test_ids)} papers")

    dataset = load_jsonl(dataset_path)
    data_by_id = {r["paper_id"]: r for r in dataset}

    teacher_records = load_jsonl(summaries_path)
    teacher_by_id = {}
    for r in teacher_records:
        if r.get("success", False):
            teacher_by_id[r["paper_id"]] = r

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
#                        MODEL LOADING & INFERENCE                       #
# ===================================================================== #

def _bnb_4bit_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")


def load_finetuned_model(base_model_name, adapter_path, use_qlora=False):
    """Load the base model (bf16, or 4-bit with use_qlora) and apply the adapter.

    The tokenizer is taken from the adapter directory, i.e. the one saved at
    training time.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    log.info(f"Loading base model: {base_model_name}")
    kwargs = {"device_map": "auto"}
    if use_qlora:
        log.info("  Using QLoRA 4-bit quantization")
        kwargs["quantization_config"] = _bnb_4bit_config()
    else:
        import torch
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    log.info(f"  Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def load_base_model(base_model_name, use_qlora=False):
    """Load the un-tuned base model with the same precision as the fine-tuned one."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info(f"Loading base (non-fine-tuned) model: {base_model_name}")
    kwargs = {"device_map": "auto"}
    if use_qlora:
        kwargs["quantization_config"] = _bnb_4bit_config()
    else:
        import torch
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.eval()
    return model, tokenizer


def _truncate_input_text(tokenizer, title, input_text, max_seq_length=8192):
    """Cut the paper text so that the whole prompt fits in max_seq_length tokens.

    The budget is measured by tokenizing the prompt with an empty paper body;
    the paper is then truncated from the end (head-only) and marked.
    """
    overhead_messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n"},
    ]
    overhead_prompt = tokenizer.apply_chat_template(
        overhead_messages, tokenize=False, add_generation_prompt=True)
    overhead_tokens = len(tokenizer.encode(overhead_prompt, add_special_tokens=False))
    content_budget = max_seq_length - overhead_tokens - 10
    if content_budget <= 100:
        log.warning(f"Very small content budget ({content_budget} tokens). Truncating aggressively.")
        content_budget = max(100, content_budget)
    content_ids = tokenizer.encode(input_text, add_special_tokens=False)
    if len(content_ids) > content_budget:
        content_ids = content_ids[:content_budget]
        input_text = tokenizer.decode(content_ids, skip_special_tokens=True) + "\n\n[Truncated]"
    return input_text


def generate_summary(model, tokenizer, title, input_text):
    """Generate one summary. Sampling parameters are fixed for all systems."""
    import torch
    input_text = _truncate_input_text(tokenizer, title, input_text)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n{input_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # The chat template already contains BOS; do not add it a second time.
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=512, do_sample=True,
            temperature=0.7, top_p=0.9, repetition_penalty=1.1,
            pad_token_id=pad_token_id)
    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_inference(model, tokenizer, examples, label):
    """Summarize every test paper; a failed generation is recorded as ''."""
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    results = {}
    n = len(examples)
    log.info(f"Running inference ({label}) on {n} examples...")
    t0 = time.time()
    for i, ex in enumerate(examples):
        try:
            results[ex["paper_id"]] = generate_summary(model, tokenizer, ex["title"], ex["input_text"])
        except Exception as e:
            log.error(f"[{label}] Error on {ex['paper_id']}: {e}")
            results[ex["paper_id"]] = ""
        if (i + 1) % 10 == 0 or (i + 1) == n:
            log.info(f"  [{label}] {i+1}/{n} done ({(i+1)/(time.time()-t0):.2f} ex/s)")
    log.info(f"Inference ({label}) complete: {time.time()-t0:.1f}s total")
    return results


# ===================================================================== #
#                     PART 1: REFERENCE METRICS                          #
# ===================================================================== #

def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)
    return {k: float(np.mean(v)) for k, v in scores.items()}


_bertscore_patched = False


def _patch_bertscore_overflow():
    """Truncate BERTScore inputs to 512 tokens.

    bert_score.utils.sent_encode does not truncate, and a summary or abstract
    longer than DeBERTa's 512-token window raises an indexing error inside the
    model. Truncating at encode time is the smallest fix.
    """
    global _bertscore_patched
    if _bertscore_patched:
        return
    import bert_score.utils as bsu

    def _safe(tokenizer, text, *a, **kw):
        return tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)

    bsu.sent_encode = _safe
    _bertscore_patched = True


def compute_bertscore(predictions, references):
    _patch_bertscore_overflow()
    from bert_score import score as bert_score
    P, R, F1 = bert_score(predictions, references, lang="en",
                          model_type="microsoft/deberta-xlarge-mnli", verbose=False, batch_size=16)
    return {"bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean())}


def compute_bleu(predictions, references):
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    ref_tokens = [[ref.split()] for ref in references]
    pred_tokens = [pred.split() for pred in predictions]
    corpus = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
    sentence = [sentence_bleu(r, p, smoothing_function=smooth) for p, r in zip(pred_tokens, ref_tokens)]
    return {"bleu_corpus": float(corpus), "bleu_sentence_avg": float(np.mean(sentence))}


def compute_meteor(predictions, references):
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    scores = [meteor_score([word_tokenize(ref)], word_tokenize(pred))
              for pred, ref in zip(predictions, references)]
    return {"meteor": float(np.mean(scores))}


def compute_all_metrics(predictions, references):
    metrics = {}
    log.info("    ROUGE...")
    metrics.update(compute_rouge(predictions, references))
    log.info("    BERTScore...")
    metrics.update(compute_bertscore(predictions, references))
    log.info("    BLEU...")
    metrics.update(compute_bleu(predictions, references))
    log.info("    METEOR...")
    metrics.update(compute_meteor(predictions, references))
    return metrics


def evaluate_model_summaries(examples, summaries, label):
    """Score one system against the abstracts and against the teacher summaries.

    Returns {"summary_stats": ..., "vs_abstract": {scope: metrics},
    "vs_teacher": {scope: metrics}} with scopes "overall" and each domain.
    The teacher is not scored against itself.
    """
    valid = [ex for ex in examples if ex["paper_id"] in summaries and summaries[ex["paper_id"]]]
    log.info(f"Evaluating [{label}]: {len(valid)} examples")
    by_domain = defaultdict(list)
    for ex in valid:
        by_domain[ex["domain"]].append(ex)

    word_counts = [len(summaries[ex["paper_id"]].split()) for ex in valid]
    results = {"summary_stats": {
        "avg_word_count": float(np.mean(word_counts)),
        "median_word_count": float(np.median(word_counts)),
        "min_word_count": int(np.min(word_counts)),
        "max_word_count": int(np.max(word_counts)),
        "total_examples": len(valid),
    }}

    references = [("vs_abstract", "abstract", "abstracts"),
                  ("vs_teacher", "teacher_summary", "teacher summaries")]
    for ref_key, ref_field, ref_desc in references:
        if label == "teacher" and ref_key == "vs_teacher":
            log.info(f"  [{label}] vs {ref_desc} — SKIPPED (self-reference)")
            continue
        log.info(f"  [{label}] vs {ref_desc}")
        ref_results = {}
        preds = [summaries[ex["paper_id"]] for ex in valid]
        refs = [ex[ref_field] for ex in valid]
        ref_results["overall"] = compute_all_metrics(preds, refs)
        ref_results["overall"]["n"] = len(preds)
        for domain in DOMAINS:
            domain_examples = by_domain.get(domain, [])
            if not domain_examples:
                continue
            domain_preds = [summaries[ex["paper_id"]] for ex in domain_examples]
            domain_refs = [ex[ref_field] for ex in domain_examples]
            ref_results[domain] = compute_all_metrics(domain_preds, domain_refs)
            ref_results[domain]["n"] = len(domain_preds)
        results[ref_key] = ref_results
    return results


def format_metrics_table(results, systems):
    """Render the reference metrics of all systems as the eval_report.txt table."""
    lines = []
    sep = "-" * 120
    sample = None
    for system in systems:
        if system in results and "vs_abstract" in results[system]:
            sample = results[system]["vs_abstract"]["overall"]
            break
    if not sample:
        return "No results to display."
    metric_keys = [k for k in sample if k != "n"]

    for ref_key in ["vs_abstract", "vs_teacher"]:
        ref_desc = ("vs. Abstracts (human reference)" if ref_key == "vs_abstract"
                    else "vs. Teacher Summaries (Claude Opus)")
        lines.append(f"\n{'='*120}\n  REFERENCE: {ref_desc}\n{'='*120}")
        for scope in ["overall"] + DOMAINS:
            scope_title = scope.upper() if scope != "overall" else "OVERALL"
            lines.append(f"\n  {scope_title}\n  {sep}")
            header = f"  {'Metric':<25}"
            for system in systems:
                header += f" {system:>20}"
            lines.append(header)
            lines.append(f"  {sep}")
            for metric in metric_keys:
                row = f"  {metric:<25}"
                for system in systems:
                    value = results.get(system, {}).get(ref_key, {}).get(scope, {}).get(metric)
                    row += f" {value:>20.4f}" if value is not None else f" {'—':>20}"
                lines.append(row)
            row = f"  {'n':<25}"
            for system in systems:
                n = results.get(system, {}).get(ref_key, {}).get(scope, {}).get("n", "—")
                row += f" {str(n):>20}"
            lines.append(row)

    lines.append(f"\n{'='*120}\n  SUMMARY STATISTICS (word counts)\n{'='*120}")
    header = f"  {'Stat':<25}"
    for system in systems:
        header += f" {system:>20}"
    lines.append(header)
    lines.append(f"  {sep}")
    for stat in ["avg_word_count", "median_word_count", "min_word_count", "max_word_count", "total_examples"]:
        row = f"  {stat:<25}"
        for system in systems:
            value = results.get(system, {}).get("summary_stats", {}).get(stat)
            row += f" {value:>20.1f}" if isinstance(value, float) else f" {value:>20}" if value is not None else f" {'—':>20}"
        lines.append(row)
    return "\n".join(lines)


# ===================================================================== #
#                  PART 2: SIGNIFICANCE TESTING                          #
# ===================================================================== #
# Per-example scores for the fine-tuned and base model, tested as paired
# samples (same paper, same reference).

def _rouge_per_example(preds, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(preds, refs):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)
    return scores


def _bertscore_per_example(preds, refs):
    _patch_bertscore_overflow()
    from bert_score import score as bert_score
    _, _, F1 = bert_score(preds, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli",
                          verbose=False, batch_size=16)
    return {"bertscore_f1": F1.tolist()}


def _bleu_per_example(preds, refs):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    return {"bleu_sentence": [sentence_bleu([r.split()], p.split(), smoothing_function=smooth)
                              for p, r in zip(preds, refs)]}


def _meteor_per_example(preds, refs):
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    return {"meteor": [meteor_score([word_tokenize(r)], word_tokenize(p)) for p, r in zip(preds, refs)]}


def _all_per_example(preds, refs):
    return {**_rouge_per_example(preds, refs), **_bertscore_per_example(preds, refs),
            **_bleu_per_example(preds, refs), **_meteor_per_example(preds, refs)}


def _paired_permutation_test(a, b, n=10000, seed=42):
    """Two-sided paired sign-flip permutation test.

    Under the null hypothesis of no systematic difference, the sign of each
    paired difference is arbitrary. Each iteration flips every difference with
    probability 1/2 and records whether the resulting absolute mean is at
    least the observed one. Returns (p-value, observed mean difference).
    """
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    diffs = a - b
    observed = abs(diffs.mean())
    count = 0
    for _ in range(n):
        signs = rng.choice([-1, 1], size=len(diffs))
        if abs((diffs * signs).mean()) >= observed:
            count += 1
    return (count + 1) / (n + 1), float(diffs.mean())


def _paired_tests(ft, base):
    """Permutation p-value, Wilcoxon p-value and Cohen's d for paired scores."""
    from scipy import stats
    ft, base = np.array(ft), np.array(base)
    p_perm, _ = _paired_permutation_test(ft, base)
    try:
        _, p_wilcoxon = stats.wilcoxon(ft, base, alternative='two-sided')
    except ValueError:  # all differences are zero
        p_wilcoxon = 1.0
    diffs = ft - base
    cohens_d = float(diffs.mean() / (diffs.std(ddof=1) + 1e-10))
    return {"mean_ft": float(ft.mean()), "mean_base": float(base.mean()),
            "diff": float(ft.mean() - base.mean()), "p_boot": p_perm,
            "p_wilcox": p_wilcoxon, "cohens_d": cohens_d, "n": len(ft)}


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _find_ft_and_base(keys):
    """Pick the fine-tuned and base system keys ('finetuned_*', 'base_*')."""
    ft_keys = [k for k in keys if "finetuned" in k]
    base_keys = [k for k in keys if "base" in k]
    if not ft_keys or not base_keys:
        return None, None
    return ft_keys[0], base_keys[0]


def run_full_significance(generations, examples, output_dir):
    """Part 2: fine-tuned vs. base on every metric, overall and per domain."""
    from scipy import stats
    log.info("\n" + "="*80 + "\nPART 2: STATISTICAL SIGNIFICANCE TESTING\n" + "="*80)
    ft_key, base_key = _find_ft_and_base(generations)
    if ft_key is None:
        log.warning("Need finetuned + base for significance tests")
        return

    examples_by_id = {e["paper_id"]: e for e in examples}
    common_ids = sorted(pid for pid in set(generations[ft_key]) & set(generations[base_key])
                        if generations[ft_key][pid] and generations[base_key][pid] and pid in examples_by_id)
    log.info(f"Common non-empty papers: {len(common_ids)}")
    ft_preds = [generations[ft_key][p] for p in common_ids]
    base_preds = [generations[base_key][p] for p in common_ids]
    abstracts = [examples_by_id[p]["abstract"] for p in common_ids]
    teacher_summs = [examples_by_id[p]["teacher_summary"] for p in common_ids]
    domains = [examples_by_id[p]["domain"] for p in common_ids]

    out = StringIO()
    for ref_label, refs in [("vs_abstracts", abstracts), ("vs_teacher", teacher_summs)]:
        out.write(f"\n{'='*80}\n  REFERENCE: {ref_label}\n{'='*80}\n")
        log.info(f"  Significance: {ref_label}")
        log.info("    Computing per-example metrics...")
        ft_metrics = _all_per_example(ft_preds, refs)
        base_metrics = _all_per_example(base_preds, refs)

        scopes = [("OVERALL", list(range(len(common_ids))))]
        scopes += [(d.upper(), [i for i, dom in enumerate(domains) if dom == d]) for d in DOMAINS]
        for scope_label, scope_idx in scopes:
            if not scope_idx:
                continue
            out.write(f"\n  {'─'*78}\n  {scope_label} (n={len(scope_idx)})\n  {'─'*78}\n")
            out.write(f"  {'Metric':<22} {'FT Mean':>9} {'Base Mean':>10} {'Diff':>8} {'p(boot)':>9} {'p(wilcox)':>10} {'Cohen d':>9} {'Sig?':>6}\n  {'─'*78}\n")
            for metric in ["rouge1", "rouge2", "rougeL", "bertscore_f1", "bleu_sentence", "meteor"]:
                if metric not in ft_metrics:
                    continue
                res = _paired_tests([ft_metrics[metric][i] for i in scope_idx],
                                    [base_metrics[metric][i] for i in scope_idx])
                out.write(f"  {metric:<22} {res['mean_ft']:>9.4f} {res['mean_base']:>10.4f} {res['diff']:>+8.4f} "
                          f"{res['p_boot']:>9.4f} {res['p_wilcox']:>10.4f} {res['cohens_d']:>9.3f} {_stars(res['p_boot']):>6}\n")

    ft_lens = np.array([len(generations[ft_key][p].split()) for p in common_ids])
    base_lens = np.array([len(generations[base_key][p].split()) for p in common_ids])
    _, p_len = stats.wilcoxon(ft_lens, base_lens)
    out.write(f"\n{'='*80}\n  SUMMARY LENGTH ANALYSIS\n{'='*80}\n")
    out.write(f"  Fine-tuned avg: {ft_lens.mean():.1f} words (std: {ft_lens.std():.1f})\n")
    out.write(f"  Base avg:       {base_lens.mean():.1f} words (std: {base_lens.std():.1f})\n")
    out.write(f"  FT in 150-250: {np.sum((ft_lens>=150)&(ft_lens<=250))}/{len(ft_lens)}\n")
    out.write(f"  Base in 150-250: {np.sum((base_lens>=150)&(base_lens<=250))}/{len(base_lens)}\n")
    out.write(f"  Wilcoxon p: {p_len:.6f}\n")

    report = out.getvalue()
    print(report)
    with open(output_dir / "significance_results.txt", "w") as f:
        f.write(report)
    log.info(f"Significance results saved to {output_dir / 'significance_results.txt'}")


# ===================================================================== #
#                  PART 3: LONGDOCFACTSCORE                              #
# ===================================================================== #

def run_ldfact_eval(generations, examples, output_dir, device="cuda"):
    """Part 3: LongDocFACTScore of every system against the full paper text.

    Scores are averaged overall and per domain. If batch scoring fails, papers
    are scored one at a time and failures are dropped as NaN.
    """
    log.info("\n" + "="*80 + "\nPART 3: LONGDOCFACTSCORE\n" + "="*80)
    try:
        from longdocfactscore.ldfacts import LongDocFACTScore
    except ImportError:
        log.warning("longdocfactscore not installed — skipping. pip install longdocfactscore")
        return

    examples_by_id = {e["paper_id"]: e for e in examples}
    all_results = {}
    for label, summaries in generations.items():
        log.info(f"  LongDocFACTScore: {label}")
        pids, srcs, preds, domains = [], [], [], []
        for pid in sorted(summaries):
            if not summaries[pid] or pid not in examples_by_id:
                continue
            pids.append(pid)
            srcs.append(examples_by_id[pid]["input_text"])
            preds.append(summaries[pid])
            domains.append(examples_by_id[pid]["domain"])
        if not preds:
            continue

        scorer = LongDocFACTScore(device=device)
        t0 = time.time()
        try:
            scores = scorer.score_src_hyp_long(srcs, preds)
            if not isinstance(scores, (list, np.ndarray)):
                log.warning(f"  Unexpected return type: {type(scores)}. Falling back.")
                raise ValueError(f"Unexpected return type: {type(scores)}")
            elif len(scores) != len(preds):
                log.warning(f"  Score count mismatch: {len(scores)} vs {len(preds)}. Falling back.")
                raise ValueError("Score count mismatch")
        except Exception as e:
            log.error(f"  Batch failed: {e}. Scoring individually...")
            scores = []
            for i, (src, pred) in enumerate(zip(srcs, preds)):
                try:
                    sc = scorer.score_src_hyp_long([src], [pred])
                    scores.append(float(sc[0]) if isinstance(sc, (list, np.ndarray)) else float(sc))
                except Exception:
                    scores.append(float("nan"))
                if (i + 1) % 25 == 0:
                    log.info(f"    {i+1}/{len(preds)} scored individually")
        log.info(f"  Done in {time.time()-t0:.1f}s")

        scores = [float(s) for s in scores]
        valid = [(p, s, d) for p, s, d in zip(pids, scores, domains) if not np.isnan(s)]
        all_scores = np.array([s for _, s, _ in valid])
        model_results = {
            "overall": {"mean": float(all_scores.mean()), "std": float(all_scores.std()), "n": len(all_scores)},
            "per_example": {p: float(s) for p, s, _ in valid},
        }
        by_domain = defaultdict(list)
        for p, s, d in valid:
            by_domain[d].append(s)
        for domain in DOMAINS:
            if domain in by_domain:
                domain_scores = np.array(by_domain[domain])
                model_results[domain] = {"mean": float(domain_scores.mean()),
                                         "std": float(domain_scores.std()), "n": len(domain_scores)}
        all_results[label] = model_results
        log.info(f"    Overall: {model_results['overall']['mean']:.4f} (+-{model_results['overall']['std']:.4f})")

    labels = list(all_results)
    if labels:
        print(f"\n{'='*80}\n  LONGDOCFACTSCORE (higher = more factually consistent)\n{'='*80}")
        header = f"  {'Scope':<12}"
        for label in labels:
            header += f" {label:>18}"
        print(header + "\n  " + "─"*76)
        for scope in ["overall"] + DOMAINS:
            row = f"  {scope:<12}"
            for label in labels:
                row += (f" {all_results[label][scope]['mean']:>18.4f}"
                        if scope in all_results.get(label, {}) else f" {'—':>18}")
            print(row)

    to_save = {label: {k: v for k, v in res.items() if k != "per_example"}
               for label, res in all_results.items()}
    with open(output_dir / "ldfact_results.json", "w") as f:
        json.dump(to_save, f, indent=2)
    log.info(f"LongDocFACTScore saved to {output_dir / 'ldfact_results.json'}")


# ===================================================================== #
#              PART 4: ABSTRACTIVENESS ANALYSIS                          #
# ===================================================================== #

def _tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def _novel_ngram_fraction(summ_tokens, src_tokens, n):
    """Fraction of the summary's n-grams that do not occur in the source."""
    if len(summ_tokens) < n:
        return 0.0
    src_ngrams = set(tuple(src_tokens[i:i+n]) for i in range(len(src_tokens)-n+1))
    summ_ngrams = [tuple(summ_tokens[i:i+n]) for i in range(len(summ_tokens)-n+1)]
    return sum(1 for g in summ_ngrams if g not in src_ngrams) / len(summ_ngrams) if summ_ngrams else 0.0


def _extractive_fragments(summ_tokens, src_tokens):
    """Greedy extractive fragments of Grusky et al. (2018); returns their lengths."""
    positions = defaultdict(list)
    for i, tok in enumerate(src_tokens):
        positions[tok].append(i)
    fragments, j = [], 0
    while j < len(summ_tokens):
        best = 0
        if summ_tokens[j] in positions:
            for start in positions[summ_tokens[j]]:
                k = 0
                while (j+k < len(summ_tokens) and start+k < len(src_tokens)
                       and summ_tokens[j+k] == src_tokens[start+k]):
                    k += 1
                best = max(best, k)
        if best > 0:
            fragments.append(best)
            j += best
        else:
            j += 1
    return fragments


def _analyze_example(summary, source):
    summ_tokens, src_tokens = _tokenize(summary), _tokenize(source)
    if not summ_tokens or not src_tokens:
        return None
    fragments = _extractive_fragments(summ_tokens, src_tokens)
    return {"novel_1gram": _novel_ngram_fraction(summ_tokens, src_tokens, 1),
            "novel_2gram": _novel_ngram_fraction(summ_tokens, src_tokens, 2),
            "novel_3gram": _novel_ngram_fraction(summ_tokens, src_tokens, 3),
            "novel_4gram": _novel_ngram_fraction(summ_tokens, src_tokens, 4),
            "coverage": sum(fragments)/len(summ_tokens),
            "density": sum(f*f for f in fragments)/len(summ_tokens),
            "type_token_ratio": len(set(summ_tokens))/len(summ_tokens),
            "summary_length": len(summ_tokens)}


def run_abstractiveness_analysis(generations, examples, output_dir):
    """Part 4: how much each system copies from the source."""
    log.info("\n" + "="*80 + "\nPART 4: ABSTRACTIVENESS ANALYSIS\n" + "="*80)
    examples_by_id = {e["paper_id"]: e for e in examples}
    all_results = {}
    for label, summaries in generations.items():
        log.info(f"  Analyzing: {label}")
        per_example = {}
        for pid in sorted(summaries):
            if not summaries[pid] or pid not in examples_by_id:
                continue
            res = _analyze_example(summaries[pid], examples_by_id[pid]["input_text"])
            if res:
                res["domain"] = examples_by_id[pid]["domain"]
                per_example[pid] = res
        if not per_example:
            continue
        metric_keys = [m for m in next(iter(per_example.values())) if m != "domain"]
        agg = {"overall": {}}
        for metric in metric_keys:
            values = [per_example[p][metric] for p in per_example]
            agg["overall"][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for domain in DOMAINS:
            domain_pids = [p for p in per_example if per_example[p]["domain"] == domain]
            if domain_pids:
                agg[domain] = {metric: {"mean": float(np.mean([per_example[p][metric] for p in domain_pids])),
                                        "n": len(domain_pids)} for metric in metric_keys}
        all_results[label] = {"agg": agg, "per_example": per_example}

    labels = list(all_results)
    key_metrics = [("novel_1gram", "Novel unigrams %"), ("novel_2gram", "Novel bigrams %"),
                   ("novel_3gram", "Novel trigrams %"), ("novel_4gram", "Novel 4-grams %"),
                   ("coverage", "Extractive coverage"), ("density", "Extractive density"),
                   ("type_token_ratio", "Vocab diversity"), ("summary_length", "Summary length")]
    print(f"\n{'='*88}\n  EXTRACTIVENESS vs. ABSTRACTIVENESS\n{'='*88}")
    header = f"  {'Metric':<24}"
    for label in labels:
        header += f" {label:>16}"
    print(header + "\n  " + "─"*84)
    for metric, name in key_metrics:
        row = f"  {name:<24}"
        for label in labels:
            value = all_results[label]["agg"]["overall"][metric]["mean"]
            row += f" {value:>15.1%}" if metric.startswith("novel") or metric == "coverage" else f" {value:>16.2f}"
        print(row)

    ft_key, base_key = _find_ft_and_base(all_results)
    if ft_key is not None:
        from scipy import stats
        ft_per, base_per = all_results[ft_key]["per_example"], all_results[base_key]["per_example"]
        common = sorted(set(ft_per) & set(base_per))
        if len(common) >= 10:
            print(f"\n  {'─'*84}\n  SIGNIFICANCE ({ft_key} vs {base_key}, n={len(common)})\n  {'─'*84}")
            print(f"  {'Metric':<24} {'FT Mean':>9} {'Base Mean':>10} {'Diff':>8} {'p-value':>10} {'Sig':>6}\n  {'─'*84}")
            for metric, name in key_metrics:
                ft_vals = np.array([ft_per[p][metric] for p in common])
                base_vals = np.array([base_per[p][metric] for p in common])
                try:
                    _, p_value = stats.wilcoxon(ft_vals, base_vals)
                except ValueError:
                    p_value = 1.0
                print(f"  {name:<24} {ft_vals.mean():>9.4f} {base_vals.mean():>10.4f} "
                      f"{ft_vals.mean()-base_vals.mean():>+8.4f} {p_value:>10.6f} {_stars(p_value):>6}")
    print()

    to_save = {label: res["agg"] for label, res in all_results.items()}
    with open(output_dir / "abstractiveness_results.json", "w") as f:
        json.dump(to_save, f, indent=2)
    log.info(f"Abstractiveness results saved to {output_dir / 'abstractiveness_results.json'}")


# ===================================================================== #
#                               MAIN                                     #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Evaluation pipeline: inference, reference metrics, "
                                                 "significance tests, LongDocFACTScore, abstractiveness")
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "dataset.jsonl"))
    parser.add_argument("--summaries", default=str(REPO_ROOT / "data" / "summaries.jsonl"))
    parser.add_argument("--test_ids", default=None,
                        help="test_paper_ids.json from the training run "
                             "(default: <output_dir>/test_paper_ids.json)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", default=None,
                        help="LoRA adapter directory; without it only the base model is run")
    parser.add_argument("--model_tag", default="8B",
                        help="suffix for the system labels, e.g. finetuned_8B / base_8B")
    parser.add_argument("--use_qlora", action="store_true",
                        help="load the base model in 4-bit NF4")
    parser.add_argument("--metrics_only", action="store_true",
                        help="skip inference and score cached generations")
    parser.add_argument("--generations_file", default=None,
                        help="generations to score with --metrics_only "
                             "(default: <output_dir>/eval_generations.json)")
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--skip_significance", action="store_true")
    parser.add_argument("--skip_ldfact", action="store_true")
    parser.add_argument("--skip_abstractiveness", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_ids_path = args.test_ids or str(output_dir / "test_paper_ids.json")

    log.info("Loading test data...")
    examples = build_test_data(args.dataset, args.summaries, test_ids_path)
    if not examples:
        log.error("No test examples found.")
        sys.exit(1)

    # PART 0: inference (or load cached generations)
    all_generations = {}
    if args.metrics_only:
        gen_path = args.generations_file or str(output_dir / "eval_generations.json")
        log.info(f"Loading cached generations from {gen_path}")
        all_generations = load_json(gen_path)
    else:
        import torch
        if args.adapter_path:
            model, tokenizer = load_finetuned_model(args.base_model, args.adapter_path, args.use_qlora)
            ft_label = f"finetuned_{args.model_tag}"
            all_generations[ft_label] = run_inference(model, tokenizer, examples, ft_label)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not args.skip_base:
            model, tokenizer = load_base_model(args.base_model, args.use_qlora)
            base_label = f"base_{args.model_tag}"
            all_generations[base_label] = run_inference(model, tokenizer, examples, base_label)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with open(output_dir / "eval_generations.json", "w") as f:
            json.dump(all_generations, f, indent=2)
        log.info(f"Saved generations to {output_dir / 'eval_generations.json'}")

    # The teacher summaries are scored like any other system.
    all_generations["teacher"] = {ex["paper_id"]: ex["teacher_summary"] for ex in examples}

    # PART 1: reference metrics
    log.info("\n" + "="*80 + "\nPART 1: REFERENCE-BASED METRICS\n" + "="*80)
    all_results = {}
    systems = []
    for label, summaries in all_generations.items():
        log.info(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")
        all_results[label] = evaluate_model_summaries(examples, summaries, label)
        systems.append(label)
    report = format_metrics_table(all_results, systems)
    print(report)
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(output_dir / "eval_report.txt", "w") as f:
        f.write(report)

    ft_label = f"finetuned_{args.model_tag}"
    base_label = f"base_{args.model_tag}"

    def _overall(label, metric="rougeL", ref="vs_abstract"):
        return all_results.get(label, {}).get(ref, {}).get("overall", {}).get(metric)

    print(f"\n{'='*80}\n  KEY COMPARISONS (ROUGE-L, vs. abstracts)\n{'='*80}")
    for desc, a, b in [("Fine-tuned vs Teacher", ft_label, "teacher"), ("Fine-tuned vs Base", ft_label, base_label)]:
        value_a, value_b = _overall(a), _overall(b)
        if value_a is not None and value_b is not None:
            print(f"  {desc}: {value_a:.4f} vs {value_b:.4f} (diff = {value_a-value_b:+.4f})")

    # PARTS 2-4
    if not args.skip_significance:
        run_full_significance(all_generations, examples, output_dir)
    if not args.skip_ldfact:
        run_ldfact_eval(all_generations, examples, output_dir)
    if not args.skip_abstractiveness:
        run_abstractiveness_analysis(all_generations, examples, output_dir)

    log.info(f"\n{'='*80}\nALL RESULTS SAVED TO: {output_dir}\n{'='*80}\n")


if __name__ == "__main__":
    main()
