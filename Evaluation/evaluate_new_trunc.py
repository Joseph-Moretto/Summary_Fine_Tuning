#!/usr/bin/env python3
"""
Comprehensive evaluation pipeline for Scientific Paper Summarization
via Knowledge Distillation.

Combines:
  1. Reference metrics  — ROUGE-1/2/L, BERTScore, BLEU, METEOR
  2. Significance tests  — paired bootstrap + Wilcoxon signed-rank
  3. LongDocFACTScore    — retrieval-based factual consistency
  4. Abstractiveness     — novel n-gram %, extractive coverage/density

All results saved to --output_dir:
  eval_generations.json        — raw generated summaries
  eval_results.json            — per-metric scores
  eval_report.txt              — human-readable metrics table
  significance_results.txt     — paired significance tests
  ldfact_results.json          — LongDocFACTScore scores
  abstractiveness_results.json — extractiveness vs abstractiveness

Usage:
    # Full pipeline (inference + all analysis)
    python evaluate.py --base_model meta-llama/Llama-3.1-8B-Instruct \\
        --adapter_path /path/to/adapter --output_dir /path/to/output --model_tag 8B

    # Metrics only (skip inference, use cached generations)
    python evaluate.py --metrics_only --output_dir /path/to/output

    # Skip specific analyses
    python evaluate.py --metrics_only --output_dir /path/to/output \\
        --skip_ldfact --skip_significance --skip_abstractiveness
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

# Ensure NLTK data is available
def _ensure_nltk_data():
    import nltk
    for resource in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
_ensure_nltk_data()

SYSTEM_MESSAGE = (
    "You are an expert scientific paper summarizer. Generate concise, accurate "
    "summaries of scientific papers. Write in third person, present tense, using "
    "flowing prose without bullet points."
)
INSTRUCTION = (
    "Summarize the following scientific paper in 150-250 words, covering the "
    "main contribution, methodology, key results, and significance."
)


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

def load_finetuned_model(base_model_name, adapter_path, use_qlora=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    log.info(f"Loading base model: {base_model_name}")
    kwargs = {"device_map": "auto"}
    if use_qlora:
        from transformers import BitsAndBytesConfig
        log.info("  Using QLoRA 4-bit quantization")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info(f"Loading base (non-fine-tuned) model: {base_model_name}")
    kwargs = {"device_map": "auto"}
    if use_qlora:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    else:
        import torch
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.eval()
    return model, tokenizer

## ---------------------------------------------------------------------------
## Smart truncation — must match training script (Fine_Tuning_v2.py)
## ---------------------------------------------------------------------------

TRUNCATION_MARKER = "\n\n[...TRUNCATED...]\n\n"

# Section header patterns for smart truncation
SECTION_PATTERNS = [
    r'^\s*\\(?:section|subsection|subsubsection)\{',
    r'^\s*(?:\d+\.?\s+|[IVX]+\.?\s+)[A-Z]',
    r'^\s*(?:Abstract|Introduction|Background|Related Work|Methodology|Methods|'
    r'Approach|Model|Framework|Experiments?|Results?|Discussion|Conclusion|'
    r'Summary|Acknowledgments?|References|Appendix|Supplementary)',
    r'^\s*#{1,4}\s+',
]
SECTION_RE = re.compile('|'.join(SECTION_PATTERNS), re.MULTILINE | re.IGNORECASE)

def find_section_boundaries(text):
    """Find character positions of section headers in the text."""
    return sorted(m.start() for m in SECTION_RE.finditer(text))

def smart_truncate_paper(paper_text, tokenizer, token_budget, head_ratio=0.7):
    """Truncate paper text to fit within token budget, preserving key sections.

    Strategy: keep first ~70% of budget from the beginning (abstract, intro,
    methodology) and last ~30% from the end (results, conclusion). Try to cut
    at section boundaries. Insert a [TRUNCATED] marker at the cut point.

    This MUST match the training script's truncation exactly.
    """
    paper_tokens = tokenizer.encode(paper_text, add_special_tokens=False)
    if len(paper_tokens) <= token_budget:
        return paper_text

    head_tokens = int(token_budget * head_ratio)
    tail_tokens = token_budget - head_tokens - 5

    head_text = tokenizer.decode(paper_tokens[:head_tokens], skip_special_tokens=True)
    tail_text = tokenizer.decode(paper_tokens[-tail_tokens:], skip_special_tokens=True)

    # Snap head_text to a section/paragraph/sentence boundary
    search_start = int(len(head_text) * 0.8)
    best_break = len(head_text)
    section_bounds = find_section_boundaries(head_text[search_start:])
    if section_bounds:
        best_break = search_start + section_bounds[-1]
    else:
        last_para = head_text.rfind('\n\n', search_start)
        if last_para > search_start:
            best_break = last_para
        else:
            last_sentence = head_text.rfind('. ', search_start)
            if last_sentence > search_start:
                best_break = last_sentence + 1
    head_text = head_text[:best_break].rstrip()

    # Snap tail_text to start at a section/paragraph/sentence boundary
    search_end = int(len(tail_text) * 0.2)
    best_start = 0
    section_bounds = find_section_boundaries(tail_text[:search_end])
    if section_bounds:
        best_start = section_bounds[0]
    else:
        first_para = tail_text.find('\n\n', 0, search_end)
        if first_para >= 0:
            best_start = first_para + 2
        else:
            first_sentence = tail_text.find('. ', 0, search_end)
            if first_sentence >= 0:
                best_start = first_sentence + 2
    tail_text = tail_text[best_start:].lstrip()

    truncated = f"{head_text}{TRUNCATION_MARKER}{tail_text}"

    # Final safety check
    final_tokens = tokenizer.encode(truncated, add_special_tokens=False)
    if len(final_tokens) > token_budget:
        truncated = tokenizer.decode(
            tokenizer.encode(truncated, add_special_tokens=False)[:token_budget],
            skip_special_tokens=True
        )

    return truncated

def _truncate_input_text(tokenizer, title, input_text, max_seq_length=8192):
    """Compute content budget and apply smart truncation matching training."""
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
    return smart_truncate_paper(input_text, tokenizer, content_budget)

def generate_summary(model, tokenizer, title, input_text):
    import torch
    input_text = _truncate_input_text(tokenizer, title, input_text)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n{input_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    from bert_score import score as bsf
    P, R, F1 = bsf(predictions, references, lang="en",
                    model_type="microsoft/deberta-xlarge-mnli", verbose=False, batch_size=16)
    return {"bertscore_precision": float(P.mean()), "bertscore_recall": float(R.mean()), "bertscore_f1": float(F1.mean())}

def compute_bleu(predictions, references):
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    rt = [[ref.split()] for ref in references]
    pt = [pred.split() for pred in predictions]
    cb = corpus_bleu(rt, pt, smoothing_function=smooth)
    sb = [sentence_bleu(r, p, smoothing_function=smooth) for p, r in zip(pt, rt)]
    return {"bleu_corpus": float(cb), "bleu_sentence_avg": float(np.mean(sb))}

def compute_meteor(predictions, references):
    from nltk.translate.meteor_score import meteor_score as ms
    from nltk.tokenize import word_tokenize
    scores = [ms([word_tokenize(ref)], word_tokenize(pred)) for pred, ref in zip(predictions, references)]
    return {"meteor": float(np.mean(scores))}

def compute_all_metrics(predictions, references, sources):
    m = {}
    log.info("    ROUGE..."); m.update(compute_rouge(predictions, references))
    log.info("    BERTScore..."); m.update(compute_bertscore(predictions, references))
    log.info("    BLEU..."); m.update(compute_bleu(predictions, references))
    log.info("    METEOR..."); m.update(compute_meteor(predictions, references))
    return m

def evaluate_model_summaries(examples, summaries, label):
    valid = [ex for ex in examples if ex["paper_id"] in summaries and summaries[ex["paper_id"]]]
    log.info(f"Evaluating [{label}]: {len(valid)} examples")
    dg = defaultdict(list)
    for ex in valid:
        dg[ex["domain"]].append(ex)
    wc = [len(summaries[ex["paper_id"]].split()) for ex in valid]
    results = {"summary_stats": {"avg_word_count": float(np.mean(wc)), "median_word_count": float(np.median(wc)),
               "min_word_count": int(np.min(wc)), "max_word_count": int(np.max(wc)), "total_examples": len(valid)}}
    for rk, rf, rd in [("vs_abstract", "abstract", "abstracts"), ("vs_teacher", "teacher_summary", "teacher summaries")]:
        if label == "teacher" and rk == "vs_teacher":
            log.info(f"  [{label}] vs {rd} — SKIPPED (self-reference)")
            continue
        log.info(f"  [{label}] vs {rd}")
        rr = {}
        preds = [summaries[ex["paper_id"]] for ex in valid]
        refs = [ex[rf] for ex in valid]
        srcs = [ex["input_text"] for ex in valid]
        rr["overall"] = compute_all_metrics(preds, refs, srcs)
        rr["overall"]["n"] = len(preds)
        for dom in ["cs", "physics", "math"]:
            de = dg.get(dom, [])
            if not de:
                continue
            dp = [summaries[ex["paper_id"]] for ex in de]
            dr = [ex[rf] for ex in de]
            ds = [ex["input_text"] for ex in de]
            rr[dom] = compute_all_metrics(dp, dr, ds)
            rr[dom]["n"] = len(dp)
        results[rk] = rr
    return results

def format_metrics_table(results, comparisons):
    lines = []
    sep = "-" * 120
    sample = None
    for c in comparisons:
        if c in results and "vs_abstract" in results[c]:
            sample = results[c]["vs_abstract"]["overall"]
            break
    if not sample:
        return "No results to display."
    mks = [k for k in sample if k != "n"]
    for rl in ["vs_abstract", "vs_teacher"]:
        rd = "vs. Abstracts (human reference)" if rl == "vs_abstract" else "vs. Teacher Summaries (Claude Opus)"
        lines.append(f"\n{'='*120}\n  REFERENCE: {rd}\n{'='*120}")
        for scope in ["overall", "cs", "physics", "math"]:
            sd = scope.upper() if scope != "overall" else "OVERALL"
            lines.append(f"\n  {sd}\n  {sep}")
            h = f"  {'Metric':<25}"
            for c in comparisons:
                h += f" {c:>20}"
            lines.append(h)
            lines.append(f"  {sep}")
            for mk in mks:
                r = f"  {mk:<25}"
                for c in comparisons:
                    v = results.get(c, {}).get(rl, {}).get(scope, {}).get(mk)
                    r += f" {v:>20.4f}" if v is not None else f" {'—':>20}"
                lines.append(r)
            r = f"  {'n':<25}"
            for c in comparisons:
                n = results.get(c, {}).get(rl, {}).get(scope, {}).get("n", "—")
                r += f" {str(n):>20}"
            lines.append(r)
    lines.append(f"\n{'='*120}\n  SUMMARY STATISTICS (word counts)\n{'='*120}")
    h = f"  {'Stat':<25}"
    for c in comparisons:
        h += f" {c:>20}"
    lines.append(h)
    lines.append(f"  {sep}")
    for s in ["avg_word_count", "median_word_count", "min_word_count", "max_word_count", "total_examples"]:
        r = f"  {s:<25}"
        for c in comparisons:
            v = results.get(c, {}).get("summary_stats", {}).get(s)
            r += f" {v:>20.1f}" if isinstance(v, float) else f" {v:>20}" if v is not None else f" {'—':>20}"
        lines.append(r)
    return "\n".join(lines)


# ===================================================================== #
#                  PART 2: SIGNIFICANCE TESTING                          #
# ===================================================================== #

def _rouge_per_ex(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, rf in zip(preds, refs):
        s = sc.score(rf, p)
        for k in r:
            r[k].append(s[k].fmeasure)
    return r

def _bert_per_ex(preds, refs):
    _patch_bertscore_overflow()
    from bert_score import score as bsf
    _, _, F1 = bsf(preds, refs, lang="en", model_type="microsoft/deberta-xlarge-mnli", verbose=False, batch_size=16)
    return {"bertscore_f1": F1.tolist()}

def _bleu_per_ex(preds, refs):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    sm = SmoothingFunction().method1
    return {"bleu_sentence": [sentence_bleu([r.split()], p.split(), smoothing_function=sm) for p, r in zip(preds, refs)]}

def _meteor_per_ex(preds, refs):
    from nltk.translate.meteor_score import meteor_score as ms
    from nltk.tokenize import word_tokenize
    return {"meteor": [ms([word_tokenize(r)], word_tokenize(p)) for p, r in zip(preds, refs)]}

def _bootstrap(a, b, n=10000, seed=42):
    """Paired permutation bootstrap test (two-sided).
    
    For each bootstrap iteration, randomly swap the two scores within each
    pair with 50% probability. This simulates the null hypothesis that
    there is no systematic difference between A and B. Then count how
    often the permuted absolute mean difference exceeds the observed one.
    """
    rng = np.random.RandomState(seed)
    a, b = np.array(a), np.array(b)
    diffs = a - b
    obs = abs(diffs.mean())
    cnt = 0
    for _ in range(n):
        # Randomly flip the sign of each paired difference (swap A and B)
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = abs((diffs * signs).mean())
        if perm_diff >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1), float(diffs.mean())

def _sig_tests(ft, base):
    from scipy import stats
    ft, base = np.array(ft), np.array(base)
    pb, _ = _bootstrap(ft, base)
    try:
        _, pw = stats.wilcoxon(ft, base, alternative='two-sided')
    except ValueError:
        pw = 1.0
    d = ft - base
    cd = float(d.mean() / (d.std(ddof=1) + 1e-10))
    return {"mean_ft": float(ft.mean()), "mean_base": float(base.mean()),
            "diff": float(ft.mean() - base.mean()), "p_boot": pb, "p_wilcox": pw, "cohens_d": cd, "n": len(ft)}

def run_full_significance(gens, examples, output_dir):
    from scipy import stats
    log.info("\n" + "="*80 + "\nPART 2: STATISTICAL SIGNIFICANCE TESTING\n" + "="*80)
    ftk = [k for k in gens if "finetuned" in k]
    bsk = [k for k in gens if "base" in k]
    if not ftk or not bsk:
        log.warning("Need finetuned + base for significance tests"); return
    ftk, bsk = ftk[0], bsk[0]
    eby = {e["paper_id"]: e for e in examples}
    cids = sorted(pid for pid in set(gens[ftk]) & set(gens[bsk]) if gens[ftk][pid] and gens[bsk][pid] and pid in eby)
    log.info(f"Common non-empty papers: {len(cids)}")
    ft_preds = [gens[ftk][p] for p in cids]
    base_preds = [gens[bsk][p] for p in cids]
    abstracts = [eby[p]["abstract"] for p in cids]
    teacher_summs = [eby[p]["teacher_summary"] for p in cids]
    doms = [eby[p]["domain"] for p in cids]
    out = StringIO()
    for rl, refs in [("vs_abstracts", abstracts), ("vs_teacher", teacher_summs)]:
        out.write(f"\n{'='*80}\n  REFERENCE: {rl}\n{'='*80}\n")
        log.info(f"  Significance: {rl}")
        log.info("    Computing per-example metrics...")
        ft_metrics = {**_rouge_per_ex(ft_preds, refs), **_bert_per_ex(ft_preds, refs), **_bleu_per_ex(ft_preds, refs), **_meteor_per_ex(ft_preds, refs)}
        base_metrics = {**_rouge_per_ex(base_preds, refs), **_bert_per_ex(base_preds, refs), **_bleu_per_ex(base_preds, refs), **_meteor_per_ex(base_preds, refs)}
        for sl, sm in [("OVERALL", list(range(len(cids))))] + [(d.upper(), [i for i, dm in enumerate(doms) if dm == d]) for d in ["cs", "physics", "math"]]:
            if not sm:
                continue
            out.write(f"\n  {'─'*78}\n  {sl} (n={len(sm)})\n  {'─'*78}\n")
            out.write(f"  {'Metric':<22} {'FT Mean':>9} {'Base Mean':>10} {'Diff':>8} {'p(boot)':>9} {'p(wilcox)':>10} {'Cohen d':>9} {'Sig?':>6}\n  {'─'*78}\n")
            for m in ["rouge1", "rouge2", "rougeL", "bertscore_f1", "bleu_sentence", "meteor"]:
                if m not in ft_metrics:
                    continue
                r = _sig_tests([ft_metrics[m][i] for i in sm], [base_metrics[m][i] for i in sm])
                sig = "***" if r["p_boot"] < 0.001 else "**" if r["p_boot"] < 0.01 else "*" if r["p_boot"] < 0.05 else ""
                out.write(f"  {m:<22} {r['mean_ft']:>9.4f} {r['mean_base']:>10.4f} {r['diff']:>+8.4f} {r['p_boot']:>9.4f} {r['p_wilcox']:>10.4f} {r['cohens_d']:>9.3f} {sig:>6}\n")
    ft_lens = np.array([len(gens[ftk][p].split()) for p in cids])
    base_lens = np.array([len(gens[bsk][p].split()) for p in cids])
    _, pl = stats.wilcoxon(ft_lens, base_lens)
    out.write(f"\n{'='*80}\n  SUMMARY LENGTH ANALYSIS\n{'='*80}\n")
    out.write(f"  Fine-tuned avg: {ft_lens.mean():.1f} words (std: {ft_lens.std():.1f})\n")
    out.write(f"  Base avg:       {base_lens.mean():.1f} words (std: {base_lens.std():.1f})\n")
    out.write(f"  FT in 150-250: {np.sum((ft_lens>=150)&(ft_lens<=250))}/{len(ft_lens)}\n")
    out.write(f"  Base in 150-250: {np.sum((base_lens>=150)&(base_lens<=250))}/{len(base_lens)}\n")
    out.write(f"  Wilcoxon p: {pl:.6f}\n")
    rpt = out.getvalue()
    print(rpt)
    with open(output_dir / "significance_results.txt", "w") as f:
        f.write(rpt)
    log.info(f"Significance results saved to {output_dir / 'significance_results.txt'}")


# ===================================================================== #
#                  PART 3: LONGDOCFACTSCORE                              #
# ===================================================================== #

def run_ldfact_eval(gens, examples, output_dir, device="cuda"):
    log.info("\n" + "="*80 + "\nPART 3: LONGDOCFACTSCORE\n" + "="*80)
    try:
        from longdocfactscore.ldfacts import LongDocFACTScore
    except ImportError:
        log.warning("longdocfactscore not installed — skipping. pip install longdocfactscore")
        return
    eby = {e["paper_id"]: e for e in examples}
    all_results = {}
    for label, sums in gens.items():
        log.info(f"  LongDocFACTScore: {label}")
        pids, srcs, preds, doms = [], [], [], []
        for pid in sorted(sums):
            if not sums[pid] or pid not in eby:
                continue
            pids.append(pid); srcs.append(eby[pid]["input_text"]); preds.append(sums[pid]); doms.append(eby[pid]["domain"])
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
            for i, (s, p) in enumerate(zip(srcs, preds)):
                try:
                    sc = scorer.score_src_hyp_long([s], [p])
                    scores.append(float(sc[0]) if isinstance(sc, (list, np.ndarray)) else float(sc))
                except Exception:
                    scores.append(float("nan"))
                if (i + 1) % 25 == 0:
                    log.info(f"    {i+1}/{len(preds)} scored individually")
        log.info(f"  Done in {time.time()-t0:.1f}s")
        scores = [float(s) for s in scores]
        valid = [(p, s, d) for p, s, d in zip(pids, scores, doms) if not np.isnan(s)]
        asc = np.array([s for _, s, _ in valid])
        mr = {"overall": {"mean": float(asc.mean()), "std": float(asc.std()), "n": len(asc)},
              "per_example": {p: float(s) for p, s, _ in valid}}
        dg = defaultdict(list)
        for p, s, d in valid:
            dg[d].append(s)
        for dom in ["cs", "physics", "math"]:
            if dom in dg:
                ds = np.array(dg[dom])
                mr[dom] = {"mean": float(ds.mean()), "std": float(ds.std()), "n": len(ds)}
        all_results[label] = mr
        log.info(f"    Overall: {mr['overall']['mean']:.4f} (+-{mr['overall']['std']:.4f})")
    labels = list(all_results)
    if labels:
        print(f"\n{'='*80}\n  LONGDOCFACTSCORE (higher = more factually consistent)\n{'='*80}")
        h = f"  {'Scope':<12}"
        for l in labels:
            h += f" {l:>18}"
        print(h + "\n  " + "─"*76)
        for sc in ["overall", "cs", "physics", "math"]:
            r = f"  {sc:<12}"
            for l in labels:
                r += f" {all_results[l][sc]['mean']:>18.4f}" if sc in all_results.get(l, {}) else f" {'—':>18}"
            print(r)
    sv = {l: {k: v for k, v in r.items() if k != "per_example"} for l, r in all_results.items()}
    with open(output_dir / "ldfact_results.json", "w") as f:
        json.dump(sv, f, indent=2)
    log.info(f"LongDocFACTScore saved to {output_dir / 'ldfact_results.json'}")


# ===================================================================== #
#              PART 4: ABSTRACTIVENESS ANALYSIS                          #
# ===================================================================== #

def _tok(text):
    return re.findall(r'\b\w+\b', text.lower())

def _novel_pct(st, sr, n):
    if len(st) < n:
        return 0.0
    sng = set(tuple(sr[i:i+n]) for i in range(len(sr)-n+1))
    sg = [tuple(st[i:i+n]) for i in range(len(st)-n+1)]
    return sum(1 for g in sg if g not in sng) / len(sg) if sg else 0.0

def _ext_frags(st, sr):
    idx = defaultdict(list)
    for i, t in enumerate(sr):
        idx[t].append(i)
    frags, j = [], 0
    while j < len(st):
        bl = 0
        if st[j] in idx:
            for sp in idx[st[j]]:
                k = 0
                while j+k < len(st) and sp+k < len(sr) and st[j+k] == sr[sp+k]:
                    k += 1
                bl = max(bl, k)
        if bl > 0:
            frags.append(bl); j += bl
        else:
            j += 1
    return frags

def _analyze_ex(summ, src):
    st, sr = _tok(summ), _tok(src)
    if not st or not sr:
        return None
    fr = _ext_frags(st, sr)
    return {"novel_1gram": _novel_pct(st, sr, 1), "novel_2gram": _novel_pct(st, sr, 2),
            "novel_3gram": _novel_pct(st, sr, 3), "novel_4gram": _novel_pct(st, sr, 4),
            "coverage": sum(fr)/len(st), "density": sum(f*f for f in fr)/len(st),
            "type_token_ratio": len(set(st))/len(st), "summary_length": len(st)}

def run_abstractiveness_analysis(gens, examples, output_dir):
    log.info("\n" + "="*80 + "\nPART 4: ABSTRACTIVENESS ANALYSIS\n" + "="*80)
    eby = {e["paper_id"]: e for e in examples}
    all_results = {}
    for label, sums in gens.items():
        log.info(f"  Analyzing: {label}")
        pe = {}
        for pid in sorted(sums):
            if not sums[pid] or pid not in eby:
                continue
            r = _analyze_ex(sums[pid], eby[pid]["input_text"])
            if r:
                r["domain"] = eby[pid]["domain"]; pe[pid] = r
        if not pe:
            continue
        mks = [m for m in next(iter(pe.values())) if m != "domain"]
        agg = {"overall": {}}
        for m in mks:
            vs = [pe[p][m] for p in pe]
            agg["overall"][m] = {"mean": float(np.mean(vs)), "std": float(np.std(vs))}
        for dom in ["cs", "physics", "math"]:
            dp = [p for p in pe if pe[p]["domain"] == dom]
            if dp:
                agg[dom] = {m: {"mean": float(np.mean([pe[p][m] for p in dp])), "n": len(dp)} for m in mks}
        all_results[label] = {"agg": agg, "per_example": pe}
    labels = list(all_results)
    km = [("novel_1gram", "Novel unigrams %"), ("novel_2gram", "Novel bigrams %"),
          ("novel_3gram", "Novel trigrams %"), ("novel_4gram", "Novel 4-grams %"),
          ("coverage", "Extractive coverage"), ("density", "Extractive density"),
          ("type_token_ratio", "Vocab diversity"), ("summary_length", "Summary length")]
    print(f"\n{'='*88}\n  EXTRACTIVENESS vs. ABSTRACTIVENESS\n{'='*88}")
    h = f"  {'Metric':<24}"
    for l in labels:
        h += f" {l:>16}"
    print(h + "\n  " + "─"*84)
    for mk, dn in km:
        r = f"  {dn:<24}"
        for l in labels:
            v = all_results[l]["agg"]["overall"][mk]["mean"]
            r += f" {v:>15.1%}" if mk.startswith("novel") or mk == "coverage" else f" {v:>16.2f}"
        print(r)
    ftk = [k for k in all_results if "finetuned" in k]
    bsk = [k for k in all_results if "base" in k]
    if ftk and bsk:
        from scipy import stats
        ftk, bsk = ftk[0], bsk[0]
        fp, bp = all_results[ftk]["per_example"], all_results[bsk]["per_example"]
        com = sorted(set(fp) & set(bp))
        if len(com) >= 10:
            print(f"\n  {'─'*84}\n  SIGNIFICANCE ({ftk} vs {bsk}, n={len(com)})\n  {'─'*84}")
            print(f"  {'Metric':<24} {'FT Mean':>9} {'Base Mean':>10} {'Diff':>8} {'p-value':>10} {'Sig':>6}\n  {'─'*84}")
            for mk, dn in km:
                fv = np.array([fp[p][mk] for p in com])
                bv = np.array([bp[p][mk] for p in com])
                try:
                    _, pv = stats.wilcoxon(fv, bv)
                except ValueError:
                    pv = 1.0
                sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                print(f"  {dn:<24} {fv.mean():>9.4f} {bv.mean():>10.4f} {fv.mean()-bv.mean():>+8.4f} {pv:>10.6f} {sig:>6}")
    print()
    sv = {l: r["agg"] for l, r in all_results.items()}
    with open(output_dir / "abstractiveness_results.json", "w") as f:
        json.dump(sv, f, indent=2)
    log.info(f"Abstractiveness results saved to {output_dir / 'abstractiveness_results.json'}")


# ===================================================================== #
#                               MAIN                                     #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation pipeline")
    parser.add_argument("--dataset", default="/scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl")
    parser.add_argument("--summaries", default="/scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl")
    parser.add_argument("--test_ids", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--model_tag", default="8B")
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--metrics_only", action="store_true")
    parser.add_argument("--generations_file", default=None)
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
        log.error("No test examples found."); sys.exit(1)

    # PART 0: Inference
    all_generations = {}
    if args.metrics_only:
        gp = args.generations_file or str(output_dir / "eval_generations.json")
        log.info(f"Loading cached generations from {gp}")
        all_generations = load_json(gp)
    else:
        import torch
        if args.adapter_path:
            model, tok = load_finetuned_model(args.base_model, args.adapter_path, args.use_qlora)
            fl = f"finetuned_{args.model_tag}"
            all_generations[fl] = run_inference(model, tok, examples, fl)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        if not args.skip_base:
            model, tok = load_base_model(args.base_model, args.use_qlora)
            bl = f"base_{args.model_tag}"
            all_generations[bl] = run_inference(model, tok, examples, bl)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        with open(output_dir / "eval_generations.json", "w") as f:
            json.dump(all_generations, f, indent=2)
        log.info(f"Saved generations to {output_dir / 'eval_generations.json'}")

    all_generations["teacher"] = {ex["paper_id"]: ex["teacher_summary"] for ex in examples}

    # PART 1: Reference metrics
    log.info("\n" + "="*80 + "\nPART 1: REFERENCE-BASED METRICS\n" + "="*80)
    all_results = {}
    comp_labels = []
    for label, sums in all_generations.items():
        log.info(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")
        all_results[label] = evaluate_model_summaries(examples, sums, label)
        comp_labels.append(label)
    report = format_metrics_table(all_results, comp_labels)
    print(report)
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(output_dir / "eval_report.txt", "w") as f:
        f.write(report)

    fl = f"finetuned_{args.model_tag}"
    bl = f"base_{args.model_tag}"
    gm = lambda l, m="rougeL", r="vs_abstract": all_results.get(l, {}).get(r, {}).get("overall", {}).get(m)
    print(f"\n{'='*80}\n  KEY COMPARISONS (ROUGE-L, vs. abstracts)\n{'='*80}")
    for desc, a, b in [(f"Fine-tuned vs Teacher", fl, "teacher"), (f"Fine-tuned vs Base", fl, bl)]:
        va, vb = gm(a), gm(b)
        if va is not None and vb is not None:
            print(f"  {desc}: {va:.4f} vs {vb:.4f} (D = {va-vb:+.4f})")

    # PART 2
    if not args.skip_significance:
        run_full_significance(all_generations, examples, output_dir)
    # PART 3
    if not args.skip_ldfact:
        run_ldfact_eval(all_generations, examples, output_dir)
    # PART 4
    if not args.skip_abstractiveness:
        run_abstractiveness_analysis(all_generations, examples, output_dir)

    log.info(f"\n{'='*80}\nALL RESULTS SAVED TO: {output_dir}\n{'='*80}\n")


if __name__ == "__main__":
    main()