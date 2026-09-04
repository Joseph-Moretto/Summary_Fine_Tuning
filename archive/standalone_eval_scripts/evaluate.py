#!/usr/bin/env python3
"""
Evaluation script for Scientific Paper Summarization via Knowledge Distillation.

Generates summaries from fine-tuned (student) and base models on the held-out
test set, then computes metrics against two reference sets:
  1. Paper abstracts (human baseline)
  2. Teacher summaries from Claude Opus (distillation target)

Metrics:
  - Reference-based: ROUGE-1/2/L, BERTScore, BLEU, METEOR
  - Factual consistency: SummaC (NLI-based, source vs. summary)

Reports results per-domain (cs, physics, math) and overall.

Usage:
    # Full pipeline: inference + metrics
    python evaluate.py --base_model meta-llama/Llama-3.3-70B-Instruct \
                       --adapter_path /scratch/jam5cq/Summary_Fine_Tuning/Output/adapter \
                       --output_dir /scratch/jam5cq/Summary_Fine_Tuning/Output \
                       --model_tag 70B --use_qlora

    # 8B model (no quantization)
    python evaluate.py --base_model meta-llama/Llama-3.1-8B-Instruct \
                       --adapter_path /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/adapter \
                       --output_dir /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16 \
                       --model_tag 8B

    # Metrics only (skip inference, use cached generations)
    python evaluate.py --metrics_only \
                       --output_dir /scratch/jam5cq/Summary_Fine_Tuning/Output

    # Skip SummaC if not installed
    python evaluate.py ... --skip_summac
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------
SYSTEM_MESSAGE = (
    "You are an expert scientific paper summarizer. Generate concise, accurate "
    "summaries of scientific papers. Write in third person, present tense, using "
    "flowing prose without bullet points."
)

INSTRUCTION = (
    "Summarize the following scientific paper in 150-250 words, covering the "
    "main contribution, methodology, key results, and significance."
)

DOMAIN_PREFIXES = {
    "cs": "cs",
    "physics": "physics",
    "math": "math",
    "stat": "math",
    "quant-ph": "physics",
    "astro-ph": "physics",
    "cond-mat": "physics",
    "gr-qc": "physics",
    "hep": "physics",
    "nucl": "physics",
    "nlin": "physics",
}


# ===================================================================== #
#                           DATA LOADING                                 #
# ===================================================================== #

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_teacher_prefix(summary: str) -> str:
    """Remove the '## Summary\\n\\n' prefix from teacher summaries."""
    if summary.startswith("## Summary"):
        summary = summary[len("## Summary"):].lstrip("\n").strip()
    return summary


def build_test_data(
    dataset_path: str,
    summaries_path: str,
    test_ids_path: str,
) -> list[dict]:
    """
    Join dataset, teacher summaries, and test split metadata into a single
    list of test examples with all fields needed for inference and evaluation.
    """
    test_meta = load_json(test_ids_path)
    test_id_to_domain = {item["paper_id"]: item["domain"] for item in test_meta}
    test_ids = set(test_id_to_domain.keys())
    log.info(f"Test set: {len(test_ids)} papers")

    dataset = load_jsonl(dataset_path)
    dataset_by_id = {r["paper_id"]: r for r in dataset}

    teacher_records = load_jsonl(summaries_path)
    teacher_by_id = {}
    for r in teacher_records:
        if r.get("success", False):
            teacher_by_id[r["paper_id"]] = r

    examples = []
    missing_dataset = 0
    missing_teacher = 0
    for paper_id in sorted(test_ids):
        if paper_id not in dataset_by_id:
            missing_dataset += 1
            continue
        if paper_id not in teacher_by_id:
            missing_teacher += 1
            continue

        d = dataset_by_id[paper_id]
        t = teacher_by_id[paper_id]
        examples.append({
            "paper_id": paper_id,
            "domain": test_id_to_domain[paper_id],
            "title": d["title"],
            "input_text": d["input_text"],
            "abstract": d["reference_summary"],
            "teacher_summary": strip_teacher_prefix(t["generated_summary"]),
        })

    if missing_dataset:
        log.warning(f"{missing_dataset} test papers missing from dataset.jsonl")
    if missing_teacher:
        log.warning(f"{missing_teacher} test papers missing from summaries.jsonl")
    log.info(f"Loaded {len(examples)} test examples with all fields")
    return examples


# ===================================================================== #
#                        PROMPT / MODEL LOADING                          #
# ===================================================================== #

def build_prompt(tokenizer, title: str, input_text: str) -> str:
    """Build the chat-template prompt identical to training."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n{input_text}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_finetuned_model(base_model_name: str, adapter_path: str, use_qlora: bool):
    """Load the base model + LoRA/QLoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    log.info(f"Loading base model: {base_model_name}")
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    if use_qlora:
        log.info("Using QLoRA 4-bit quantization")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
    log.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model.eval()
    return model, tokenizer


def load_base_model(base_model_name: str, use_qlora: bool):
    """Load the base (non-fine-tuned) model for zero-shot comparison."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info(f"Loading base (non-fine-tuned) model: {base_model_name}")
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    if use_qlora:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model.eval()
    return model, tokenizer


# ===================================================================== #
#                            INFERENCE                                   #
# ===================================================================== #

def _truncate_input_text(tokenizer, title: str, input_text: str, max_seq_length: int = 8192) -> str:
    """
    Truncate paper content so the full prompt fits within max_seq_length.

    We can't truncate the final token sequence because that would cut off the
    assistant generation header at the end. Instead, we measure how many tokens
    the instruction wrapper uses, then truncate the paper content to fit.
    """
    # Build prompt with a minimal placeholder to measure overhead
    overhead_messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n"},
    ]
    overhead_prompt = tokenizer.apply_chat_template(
        overhead_messages, tokenize=False, add_generation_prompt=True
    )
    overhead_tokens = len(tokenizer.encode(overhead_prompt, add_special_tokens=False))

    # Budget for paper content
    content_budget = max_seq_length - overhead_tokens - 10  # small safety margin

    if content_budget <= 0:
        return input_text[:1000]  # fallback

    # Tokenize the paper content and truncate if needed
    content_ids = tokenizer.encode(input_text, add_special_tokens=False)
    if len(content_ids) > content_budget:
        content_ids = content_ids[:content_budget]
        input_text = tokenizer.decode(content_ids, skip_special_tokens=True)
        input_text += "\n\n[Truncated]"

    return input_text


def generate_summary(model, tokenizer, title: str, input_text: str) -> str:
    """Generate a single summary using the loaded model."""
    import torch

    # Truncate paper content (not the prompt) to preserve instruction + generation header
    input_text = _truncate_input_text(tokenizer, title, input_text, max_seq_length=8192)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n{input_text}"},
    ]

    # Get prompt string - chat template already includes BOS token
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize with add_special_tokens=False to avoid duplicate BOS
    inputs = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.device)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=pad_token_id,
        )

    generated = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_inference(model, tokenizer, examples: list[dict], label: str) -> dict:
    """Run inference on all test examples. Returns {paper_id: summary}."""
    results = {}
    n = len(examples)
    log.info(f"Running inference ({label}) on {n} examples...")
    t0 = time.time()

    for i, ex in enumerate(examples):
        try:
            summary = generate_summary(
                model, tokenizer, ex["title"], ex["input_text"]
            )
            results[ex["paper_id"]] = summary
        except Exception as e:
            log.error(f"[{label}] Error on {ex['paper_id']}: {e}")
            results[ex["paper_id"]] = ""

        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            log.info(f"  [{label}] {i+1}/{n} done ({rate:.2f} examples/s)")

    elapsed = time.time() - t0
    log.info(f"Inference ({label}) complete: {elapsed:.1f}s total")
    return results


# ===================================================================== #
#                             METRICS                                    #
# ===================================================================== #

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L (F1 scores)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: float(np.mean(v)) for k, v in scores.items()}


_bertscore_patched = False

def _patch_bertscore_overflow():
    """
    Replace bert_score.utils.sent_encode to fix OverflowError.

    bert_score 0.3.13 reads model.config.max_position_embeddings (which can
    be extremely large) and passes it to tokenizer.encode(). The Rust-backed
    tokenizers library casts this to a C int and overflows.

    This replaces sent_encode entirely with a safe version that caps
    max_length at 512 (DeBERTa-xlarge's actual context window).
    """
    global _bertscore_patched
    if _bertscore_patched:
        return
    import bert_score.utils as bsu

    def _safe_sent_encode(tokenizer, text, *args, **kwargs):
        return tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
        )

    bsu.sent_encode = _safe_sent_encode
    _bertscore_patched = True


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """BERTScore (precision, recall, F1)."""
    _patch_bertscore_overflow()
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False,
        batch_size=16,
    )
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    """Corpus-level BLEU and average sentence-level BLEU."""
    from nltk.translate.bleu_score import (
        sentence_bleu,
        corpus_bleu,
        SmoothingFunction,
    )

    smooth = SmoothingFunction().method1
    ref_tokenized = [[ref.split()] for ref in references]
    pred_tokenized = [pred.split() for pred in predictions]

    c_bleu = corpus_bleu(ref_tokenized, pred_tokenized, smoothing_function=smooth)

    s_bleus = [
        sentence_bleu(ref_tok, pred_tok, smoothing_function=smooth)
        for pred_tok, ref_tok in zip(pred_tokenized, ref_tokenized)
    ]

    return {
        "bleu_corpus": float(c_bleu),
        "bleu_sentence_avg": float(np.mean(s_bleus)),
    }


def compute_meteor(predictions: list[str], references: list[str]) -> dict:
    """Average sentence-level METEOR score."""
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    from nltk.tokenize import word_tokenize

    scores = []
    for pred, ref in zip(predictions, references):
        ref_tokens = word_tokenize(ref)
        pred_tokens = word_tokenize(pred)
        scores.append(nltk_meteor([ref_tokens], pred_tokens))

    return {"meteor": float(np.mean(scores))}


_summac_model = None

def _get_summac_model():
    """Lazily load and cache the SummaC model."""
    global _summac_model
    if _summac_model is not None:
        return _summac_model
    try:
        from summac.model_summac import SummaCZS
    except ImportError:
        log.warning(
            "SummaC not installed - skipping. "
            "Install via: pip install summac"
        )
        return None
    try:
        _summac_model = SummaCZS(
            model_name="vitc",
            granularity="sentence",
            device="cuda",
        )
        return _summac_model
    except Exception as e:
        log.warning(f"SummaC model loading failed: {e}")
        return None


def compute_summac(
    predictions: list[str],
    sources: list[str],
) -> Optional[dict]:
    """
    SummaC - NLI-based factual consistency (source document vs. summary).
    Splits each summary into sentences and checks entailment against the source.
    Note: SummaC's NLI model has a 512-token window, so only the first ~512
    tokens of each source document are used for consistency checking.
    Returns None if summac is not installed or fails.
    """
    model = _get_summac_model()
    if model is None:
        return None

    try:
        result = model.score(sources, predictions)
        scores = result["scores"]
        return {"summac": float(np.mean(scores))}
    except Exception as e:
        log.warning(f"SummaC computation failed: {e}")
        return None


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    sources: list[str],
    skip_summac: bool = False,
) -> dict:
    """Compute all metrics for a set of predictions against references."""
    metrics = {}

    log.info("    ROUGE...")
    metrics.update(compute_rouge(predictions, references))

    log.info("    BERTScore...")
    metrics.update(compute_bertscore(predictions, references))

    log.info("    BLEU...")
    metrics.update(compute_bleu(predictions, references))

    log.info("    METEOR...")
    metrics.update(compute_meteor(predictions, references))

    if not skip_summac:
        log.info("    SummaC...")
        result = compute_summac(predictions, sources)
        if result:
            metrics.update(result)

    return metrics


# ===================================================================== #
#                       EVALUATION ORCHESTRATION                         #
# ===================================================================== #

def evaluate_model_summaries(
    examples: list[dict],
    summaries: dict,
    label: str,
    skip_summac: bool = False,
) -> dict:
    """
    Evaluate one set of generated summaries against both reference sets,
    overall and per-domain.

    Returns:
        {
            "vs_abstract": {"overall": {...}, "cs": {...}, ...},
            "vs_teacher":  {"overall": {...}, "cs": {...}, ...},
            "summary_stats": {...},
        }
    """
    valid = [
        ex for ex in examples
        if ex["paper_id"] in summaries and summaries[ex["paper_id"]]
    ]
    log.info(f"Evaluating [{label}]: {len(valid)} examples with summaries")

    domain_groups = defaultdict(list)
    for ex in valid:
        domain_groups[ex["domain"]].append(ex)

    word_counts = [len(summaries[ex["paper_id"]].split()) for ex in valid]
    results = {
        "summary_stats": {
            "avg_word_count": float(np.mean(word_counts)),
            "median_word_count": float(np.median(word_counts)),
            "min_word_count": int(np.min(word_counts)),
            "max_word_count": int(np.max(word_counts)),
            "total_examples": len(valid),
        }
    }

    for ref_key, ref_field, ref_display in [
        ("vs_abstract", "abstract", "abstracts"),
        ("vs_teacher", "teacher_summary", "teacher summaries"),
    ]:
        # Skip evaluating teacher summaries against themselves (perfect scores)
        if label == "teacher" and ref_key == "vs_teacher":
            log.info(f"  [{label}] vs {ref_display} - SKIPPED (self-reference)")
            continue

        log.info(f"  [{label}] vs {ref_display}")
        ref_results = {}

        preds = [summaries[ex["paper_id"]] for ex in valid]
        refs = [ex[ref_field] for ex in valid]
        sources = [ex["input_text"] for ex in valid]
        ref_results["overall"] = compute_all_metrics(
            preds, refs, sources, skip_summac
        )
        ref_results["overall"]["n"] = len(preds)

        for domain in ["cs", "physics", "math"]:
            d_exs = domain_groups.get(domain, [])
            if not d_exs:
                continue
            d_preds = [summaries[ex["paper_id"]] for ex in d_exs]
            d_refs = [ex[ref_field] for ex in d_exs]
            d_sources = [ex["input_text"] for ex in d_exs]
            ref_results[domain] = compute_all_metrics(
                d_preds, d_refs, d_sources, skip_summac
            )
            ref_results[domain]["n"] = len(d_preds)

        results[ref_key] = ref_results

    return results


# ===================================================================== #
#                            REPORTING                                   #
# ===================================================================== #

def format_metrics_table(results: dict, comparisons: list[str]) -> str:
    """Format results into a readable text table."""
    lines = []
    sep = "-" * 120

    sample = None
    for comp in comparisons:
        if comp in results and "vs_abstract" in results[comp]:
            sample = results[comp]["vs_abstract"]["overall"]
            break
    if not sample:
        return "No results to display."

    metric_keys = [k for k in sample.keys() if k != "n"]

    for ref_label in ["vs_abstract", "vs_teacher"]:
        ref_display = (
            "vs. Abstracts (human reference)"
            if ref_label == "vs_abstract"
            else "vs. Teacher Summaries (Claude Opus)"
        )
        lines.append(f"\n{'=' * 120}")
        lines.append(f"  REFERENCE: {ref_display}")
        lines.append(f"{'=' * 120}")

        for scope in ["overall", "cs", "physics", "math"]:
            scope_display = scope.upper() if scope != "overall" else "OVERALL"
            lines.append(f"\n  {scope_display}")
            lines.append(f"  {sep}")

            header = f"  {'Metric':<25}"
            for comp in comparisons:
                header += f" {comp:>20}"
            lines.append(header)
            lines.append(f"  {sep}")

            for mk in metric_keys:
                row = f"  {mk:<25}"
                for comp in comparisons:
                    val = (
                        results.get(comp, {})
                        .get(ref_label, {})
                        .get(scope, {})
                        .get(mk, None)
                    )
                    if val is not None:
                        row += f" {val:>20.4f}"
                    else:
                        row += f" {'-':>20}"
                lines.append(row)

            row_n = f"  {'n':<25}"
            for comp in comparisons:
                n = (
                    results.get(comp, {})
                    .get(ref_label, {})
                    .get(scope, {})
                    .get("n", "-")
                )
                row_n += f" {str(n):>20}"
            lines.append(row_n)

    lines.append(f"\n{'=' * 120}")
    lines.append("  SUMMARY STATISTICS (word counts)")
    lines.append(f"{'=' * 120}")
    header = f"  {'Stat':<25}"
    for comp in comparisons:
        header += f" {comp:>20}"
    lines.append(header)
    lines.append(f"  {sep}")
    for stat in [
        "avg_word_count",
        "median_word_count",
        "min_word_count",
        "max_word_count",
        "total_examples",
    ]:
        row = f"  {stat:<25}"
        for comp in comparisons:
            val = results.get(comp, {}).get("summary_stats", {}).get(stat, None)
            if val is not None:
                if isinstance(val, float):
                    row += f" {val:>20.1f}"
                else:
                    row += f" {val:>20}"
            else:
                row += f" {'-':>20}"
        lines.append(row)

    return "\n".join(lines)


# ===================================================================== #
#                               MAIN                                     #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate scientific summarization models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--dataset",
        type=str,
        default="/scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl",
    )
    parser.add_argument(
        "--summaries",
        type=str,
        default="/scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl",
        help="Teacher summaries from Claude Opus",
    )
    parser.add_argument(
        "--test_ids",
        type=str,
        default=None,
        help="Path to test_paper_ids.json (default: <output_dir>/test_paper_ids.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for results and default location for test_paper_ids.json",
    )

    # Model config
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default="8B",
                        help="Label for this model in results (e.g. '8B', '70B')")
    parser.add_argument("--use_qlora", action="store_true",
                        help="Use 4-bit QLoRA quantization when loading")

    # Inference control
    parser.add_argument("--metrics_only", action="store_true",
                        help="Skip inference, load generations from file")
    parser.add_argument("--generations_file", type=str, default=None,
                        help="Path to pre-generated summaries JSON (for --metrics_only)")
    parser.add_argument("--skip_base", action="store_true",
                        help="Skip base model inference (only evaluate fine-tuned)")

    # Metric control
    parser.add_argument("--skip_summac", action="store_true",
                        help="Skip SummaC factual consistency computation")

    args = parser.parse_args()

    # Resolve paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_ids_path = args.test_ids or str(output_dir / "test_paper_ids.json")

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    log.info("Loading test data...")
    examples = build_test_data(args.dataset, args.summaries, test_ids_path)
    if not examples:
        log.error("No test examples found. Check your data paths.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Inference (or load cached)
    # ------------------------------------------------------------------
    all_generations = {}

    if args.metrics_only:
        gen_path = args.generations_file or str(output_dir / "eval_generations.json")
        log.info(f"Loading cached generations from {gen_path}")
        all_generations = load_json(gen_path)
    else:
        import torch

        if args.adapter_path:
            model, tokenizer = load_finetuned_model(
                args.base_model, args.adapter_path, args.use_qlora
            )
            ft_label = f"finetuned_{args.model_tag}"
            all_generations[ft_label] = run_inference(
                model, tokenizer, examples, ft_label
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not args.skip_base:
            model, tokenizer = load_base_model(args.base_model, args.use_qlora)
            base_label = f"base_{args.model_tag}"
            all_generations[base_label] = run_inference(
                model, tokenizer, examples, base_label
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gen_path = str(output_dir / "eval_generations.json")
        with open(gen_path, "w") as f:
            json.dump(all_generations, f, indent=2)
        log.info(f"Saved generations to {gen_path}")

    # Include teacher summaries as a comparison "model"
    teacher_summaries = {ex["paper_id"]: ex["teacher_summary"] for ex in examples}
    all_generations["teacher"] = teacher_summaries

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    all_results = {}
    comparison_labels = []

    for label, summaries in all_generations.items():
        log.info(f"\n{'='*60}")
        log.info(f"Evaluating: {label}")
        log.info(f"{'='*60}")
        all_results[label] = evaluate_model_summaries(
            examples, summaries, label, skip_summac=args.skip_summac
        )
        comparison_labels.append(label)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = format_metrics_table(all_results, comparison_labels)
    print(report)

    results_path = str(output_dir / "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nFull results saved to {results_path}")

    report_path = str(output_dir / "eval_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    log.info(f"Text report saved to {report_path}")

    # ------------------------------------------------------------------
    # Key comparisons summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("  KEY COMPARISONS (ROUGE-L F1, overall, vs. abstracts)")
    print(f"{'=' * 80}")

    ft_label = f"finetuned_{args.model_tag}"
    base_label = f"base_{args.model_tag}"

    def get_metric(label, metric="rougeL", ref="vs_abstract"):
        return (
            all_results.get(label, {})
            .get(ref, {})
            .get("overall", {})
            .get(metric, None)
        )

    pairs = [
        (f"Fine-tuned {args.model_tag} vs Teacher", ft_label, "teacher"),
        (f"Fine-tuned {args.model_tag} vs Base {args.model_tag}", ft_label, base_label),
    ]

    for desc, a, b in pairs:
        va, vb = get_metric(a), get_metric(b)
        if va is not None and vb is not None:
            print(f"  {desc}: {va:.4f} vs {vb:.4f} (diff = {va - vb:+.4f})")
        else:
            status = f"{a}={va}" if va else f"{b}={vb}" if vb else "neither available"
            print(f"  {desc}: incomplete ({status})")

    print()


if __name__ == "__main__":
    main()