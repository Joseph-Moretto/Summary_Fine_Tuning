#!/usr/bin/env python3
"""
QLoRA Fine-Tuning for Scientific Document Summarization (v2)

Fine-tunes LLaMA 3.3 (8B or 70B) using QLoRA on teacher-generated summaries
from Claude Opus 4.6, following the research proposal methodology.

Changes from v1:
  - Smart truncation: measures exact token budget for paper content after
    accounting for system prompt, instruction, title, chat template tokens,
    and teacher summary — then truncates paper text to fit, preserving
    beginning (abstract/intro) and end (conclusion/results) of the paper.
  - Lower default learning rate for 70B (1e-4 vs 2e-4)qww2
  - Saves checkpoints per epoch with load_best_model_at_end
  - Collator validation: logs how many examples have masked-out targets
  - Higher LoRA dropout default (0.1) for small datasets
  - Configurable number of epochs with early stopping patience

Pipeline:
1. Combine dataset.jsonl + summaries.jsonl into instruction-response pairs
2. Apply chat template formatting with token-budget-aware truncation
3. Fine-tune with optional 4-bit NF4 quantization + LoRA adapters
4. Save merged or adapter-only weights

Usage:
    # 8B model with QLoRA (4-bit quantized, single GPU, ~8GB VRAM)
    python qlora_finetune.py \
        --dataset data/dataset.jsonl \
        --summaries data/summaries.jsonl \
        --model meta-llama/Llama-3.3-8B-Instruct \
        --output ./output/llama-8b-summarizer

    # 70B model (multi-GPU, ~48GB VRAM per GPU)
    python qlora_finetune.py \
        --dataset data/dataset.jsonl \
        --summaries data/summaries.jsonl \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --output ./output/llama-70b-summarizer \
        --lora_rank 16 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 16

Dependencies:
    pip install torch transformers peft bitsandbytes datasets trl accelerate
"""

import argparse
import json
import re
import sys
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Completion-only data collator (loss masking)
# =============================================================================

class DataCollatorForCompletionOnlyLM:
    """Masks labels so loss is only computed on the assistant's response.

    This replaces the TRL version which was removed in TRL >= 1.1.0.
    It finds the response_template token sequence in each example and sets
    all labels before it to ignore_index (-100), so the model only learns
    to produce the summary, not reproduce the prompt.

    Tracks how many examples had the template found vs not found for
    diagnostic purposes.
    """

    def __init__(self, response_template: str, tokenizer, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        # Diagnostic counters
        self.total_seen = 0
        self.template_found = 0
        self.template_missing = 0
        self._logged_warning = False

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        labels = batch["input_ids"].clone()

        # Mask pad tokens in labels using attention_mask — this correctly
        # distinguishes padding from legitimate EOS tokens (which share
        # the same token ID when pad_token = eos_token)
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = self.ignore_index

        for i in range(len(labels)):
            self.total_seen += 1
            # Search for response template in the original input_ids,
            # not labels (which may already have -100 for pad tokens)
            token_ids = batch["input_ids"][i].tolist()
            # Find the response template in the token sequence
            response_start = None
            template_len = len(self.response_token_ids)
            for j in range(len(token_ids) - template_len + 1):
                if token_ids[j:j + template_len] == self.response_token_ids:
                    response_start = j + template_len
                    break

            if response_start is not None:
                # Mask everything before the response
                labels[i, :response_start] = self.ignore_index
                self.template_found += 1
            else:
                # Template not found — mask entire sequence to avoid bad gradients
                labels[i, :] = self.ignore_index
                self.template_missing += 1
                if not self._logged_warning:
                    logger.warning(
                        "Response template not found in example — entire sequence masked. "
                        "This example contributes nothing to training. "
                        f"Template token IDs: {self.response_token_ids}"
                    )
                    # Log first 50 tokens for debugging
                    logger.warning(
                        f"First 50 tokens of problematic example: {token_ids[:50]}"
                    )
                    self._logged_warning = True

            # Log diagnostics periodically
            if self.total_seen % 500 == 0:
                logger.info(
                    f"[Collator] Processed {self.total_seen} examples: "
                    f"{self.template_found} OK, {self.template_missing} masked "
                    f"({100*self.template_missing/self.total_seen:.1f}% loss)"
                )

        batch["labels"] = labels
        return batch


# =============================================================================
# Constants
# =============================================================================

SYSTEM_MESSAGE = (
    "You are an expert scientific paper summarizer. Generate concise, accurate "
    "summaries of scientific papers. Write in third person, present tense, using "
    "flowing prose without bullet points."
)

INSTRUCTION = (
    "Summarize the following scientific paper in 150-250 words, covering the "
    "main contribution, methodology, key results, and significance."
)

# Default LoRA target modules for LLaMA 3.x
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Domain classification based on arXiv category prefixes
DOMAIN_PREFIXES = {
    "cs": "cs",
    "physics": "physics",
    "math": "math",
    "stat": "math",       # group stat with math
    "quant-ph": "physics", # quantum physics
    "astro-ph": "physics",
    "cond-mat": "physics",
    "gr-qc": "physics",
    "hep": "physics",
    "nucl": "physics",
    "nlin": "physics",
}

# Truncation marker — use the same marker in evaluate.py for consistency
TRUNCATION_MARKER = "\n\n[...TRUNCATED...]\n\n"

# Section header patterns for smart truncation
SECTION_PATTERNS = [
    # LaTeX-style section headers
    r'^\s*\\(?:section|subsection|subsubsection)\{',
    # Numbered sections: "1. Introduction", "2 Methods", "III. Results"
    r'^\s*(?:\d+\.?\s+|[IVX]+\.?\s+)[A-Z]',
    # Named sections (common in extracted text)
    r'^\s*(?:Abstract|Introduction|Background|Related Work|Methodology|Methods|'
    r'Approach|Model|Framework|Experiments?|Results?|Discussion|Conclusion|'
    r'Summary|Acknowledgments?|References|Appendix|Supplementary)',
    # Markdown-style headers
    r'^\s*#{1,4}\s+',
]
SECTION_RE = re.compile('|'.join(SECTION_PATTERNS), re.MULTILINE | re.IGNORECASE)


# =============================================================================
# Domain helpers
# =============================================================================

def classify_domain(categories: list[str]) -> str:
    """Classify a paper into a broad domain (cs, physics, math) from its arXiv categories."""
    for cat in categories:
        cat_lower = cat.lower()
        for prefix, domain in DOMAIN_PREFIXES.items():
            if cat_lower.startswith(prefix):
                return domain
    if categories:
        top = categories[0].split(".")[0].split("-")[0].lower()
        return DOMAIN_PREFIXES.get(top, "other")
    return "other"


# =============================================================================
# Smart Truncation
# =============================================================================

def compute_paper_token_budget(
    tokenizer,
    title: str,
    summary: str,
    max_seq_length: int,
    safety_margin: int = 50,
) -> int:
    """Compute exactly how many tokens are available for paper content.

    Tokenizes the full chat template with a placeholder for paper content,
    measures all non-paper tokens, and returns the remaining budget.

    Args:
        tokenizer: The tokenizer to use for exact measurement
        title: Paper title (included in the prompt)
        summary: Teacher summary (the target output)
        max_seq_length: Maximum sequence length
        safety_margin: Extra tokens to reserve (for BPE edge cases)

    Returns:
        Number of tokens available for paper content
    """
    # Build the full template with an empty paper content placeholder
    user_content = f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n"
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": summary},
    ]

    # Tokenize the template (everything except paper content)
    template_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    template_tokens = len(tokenizer.encode(template_text, add_special_tokens=False))

    budget = max_seq_length - template_tokens - safety_margin
    return max(budget, 256)  # Ensure at least 256 tokens for paper content


def find_section_boundaries(text: str) -> list[int]:
    """Find character positions of section headers in the text."""
    boundaries = []
    for match in SECTION_RE.finditer(text):
        boundaries.append(match.start())
    return sorted(boundaries)


def smart_truncate_paper(
    paper_text: str,
    tokenizer,
    token_budget: int,
    head_ratio: float = 0.7,
) -> str:
    """Truncate paper text to fit within token budget, preserving key sections.

    IMPORTANT: If you use this during training, your evaluation/inference
    pipeline must use the same truncation strategy. Import and call this
    function from evaluate.py instead of using a separate truncation method.

    Strategy:
    1. If the paper fits within budget, return as-is.
    2. Otherwise, keep the first ~70% of budget from the beginning (abstract,
       intro, methodology) and the last ~30% from the end (results, conclusion).
    3. Try to cut at section boundaries to avoid mid-sentence breaks.
    4. Insert a [TRUNCATED] marker at the cut point.

    Args:
        paper_text: Full paper text
        tokenizer: Tokenizer for exact token counting
        token_budget: Max tokens allowed for paper content
        head_ratio: Fraction of budget for the beginning (rest goes to end)

    Returns:
        Truncated paper text that fits within token_budget
    """
    # Quick check: does it already fit?
    paper_tokens = tokenizer.encode(paper_text, add_special_tokens=False)
    if len(paper_tokens) <= token_budget:
        return paper_text

    # Allocate tokens for head and tail
    head_tokens = int(token_budget * head_ratio)
    tail_tokens = token_budget - head_tokens - 5  # Reserve ~5 tokens for marker

    # Decode head and tail portions from token IDs
    head_text = tokenizer.decode(paper_tokens[:head_tokens], skip_special_tokens=True)
    tail_text = tokenizer.decode(paper_tokens[-tail_tokens:], skip_special_tokens=True)

    # Try to snap head_text to a section boundary or paragraph break
    # Look backwards from the end of head_text for a good break point
    # (within the last 20% of head_text)
    search_start = int(len(head_text) * 0.8)
    best_break = len(head_text)

    # Prefer section boundaries
    section_bounds = find_section_boundaries(head_text[search_start:])
    if section_bounds:
        best_break = search_start + section_bounds[-1]
    else:
        # Fall back to paragraph breaks
        last_para = head_text.rfind('\n\n', search_start)
        if last_para > search_start:
            best_break = last_para
        else:
            # Fall back to sentence breaks
            last_sentence = head_text.rfind('. ', search_start)
            if last_sentence > search_start:
                best_break = last_sentence + 1

    head_text = head_text[:best_break].rstrip()

    # Try to snap tail_text to start at a section or paragraph boundary
    # Look forward from the beginning of tail_text (within first 20%)
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

    # Final safety check: verify we're actually within budget
    final_tokens = tokenizer.encode(truncated, add_special_tokens=False)
    if len(final_tokens) > token_budget:
        # Hard cutoff as last resort (shouldn't happen often)
        truncated = tokenizer.decode(
            tokenizer.encode(truncated, add_special_tokens=False)[:token_budget],
            skip_special_tokens=True
        )

    return truncated


# =============================================================================
# Data Preparation
# =============================================================================

def load_and_merge_data(
    dataset_path: str,
    summaries_path: str,
    tokenizer,
    max_seq_length: int,
) -> list[dict]:
    """
    Merge dataset.jsonl and summaries.jsonl into training examples.

    Uses smart truncation to ensure every example fits within max_seq_length
    without losing the teacher summary or any prompt tokens.

    Returns list of dicts with keys:
        paper_id, title, input_text, output, reference_summary, categories, domain
    """
    # Load successful teacher summaries
    summaries = {}
    with open(summaries_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("success") and rec.get("generated_summary"):
                    summaries[rec["paper_id"]] = rec["generated_summary"]
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(summaries)} successful teacher summaries")

    # Merge with paper inputs
    examples = []
    skipped_no_summary = 0
    skipped_too_short = 0
    truncated_count = 0
    token_budgets = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            pid = paper.get("paper_id", "")
            if pid not in summaries:
                skipped_no_summary += 1
                continue

            input_text = paper.get("input_text", "")
            if not input_text or len(input_text) < 200:
                skipped_too_short += 1
                continue

            title = paper.get("title", "Untitled")
            summary = summaries[pid].lstrip("## Summary\n").strip()
            categories = paper.get("categories", [])
            domain = classify_domain(categories)

            # Compute exact token budget for this example's paper content
            budget = compute_paper_token_budget(
                tokenizer, title, summary, max_seq_length
            )
            token_budgets.append(budget)

            # Smart truncation
            original_len = len(tokenizer.encode(input_text, add_special_tokens=False))
            if original_len > budget:
                input_text = smart_truncate_paper(input_text, tokenizer, budget)
                truncated_count += 1

            examples.append({
                "paper_id": pid,
                "title": title,
                "input_text": input_text,
                "output": summary,
                "reference_summary": paper.get("reference_summary", ""),
                "categories": categories,
                "domain": domain,
            })

    logger.info(
        f"Created {len(examples)} training examples "
        f"({skipped_no_summary} no summary, {skipped_too_short} too short)"
    )
    if examples:
        logger.info(
            f"Truncated {truncated_count}/{len(examples)} examples "
            f"({100*truncated_count/len(examples):.1f}%) to fit within {max_seq_length} tokens"
        )
    if token_budgets:
        avg_budget = sum(token_budgets) / len(token_budgets)
        min_budget = min(token_budgets)
        max_budget = max(token_budgets)
        logger.info(
            f"Paper token budgets — avg: {avg_budget:.0f}, "
            f"min: {min_budget}, max: {max_budget}"
        )

    # Log domain distribution
    domain_counts = Counter(ex["domain"] for ex in examples)
    logger.info(f"Domain distribution: {dict(domain_counts)}")

    # Debug: print first merged example
    if examples:
        ex = examples[0]
        logger.info("=== First merged example ===")
        logger.info(f"  paper_id: {ex['paper_id']}")
        logger.info(f"  title: {ex['title']}")
        logger.info(f"  domain: {ex['domain']}")
        logger.info(f"  categories: {ex['categories']}")
        logger.info(f"  input_text length: {len(ex['input_text'])} chars")
        logger.info(f"  input_text: {ex['input_text'][:300]}...")
        logger.info(f"  output: {ex['output'][:300]}...")

    return examples


def format_chat_messages(example: dict) -> list[dict]:
    """Format a single example as chat messages for the tokenizer's chat template."""
    user_content = f"{INSTRUCTION}\n\nTitle: {example['title']}\n\nPaper Content:\n{example['input_text']}"
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]


def stratified_split(
    examples: list[dict],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split examples into train/val/test with proportional domain representation.

    Each domain (cs, physics, math) contributes proportionally to each split,
    ensuring balanced evaluation across domains.
    """
    import random
    rng = random.Random(seed)

    # Group by domain
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_domain[ex["domain"]].append(ex)

    train_all, val_all, test_all = [], [], []

    for domain, domain_examples in sorted(by_domain.items()):
        rng.shuffle(domain_examples)
        n = len(domain_examples)

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))

        if n_test + n_val >= n:
            n_test = max(1, n // 3)
            n_val = max(1, n // 3)
            if n_test + n_val >= n:
                logger.warning(
                    f"Domain '{domain}' has only {n} examples — "
                    f"assigning all to train (too few to split)"
                )
                train_all.extend(domain_examples)
                continue

        test_all.extend(domain_examples[:n_test])
        val_all.extend(domain_examples[n_test:n_test + n_val])
        train_all.extend(domain_examples[n_test + n_val:])

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    for split_name, split_data in [("train", train_all), ("val", val_all), ("test", test_all)]:
        counts = Counter(ex["domain"] for ex in split_data)
        logger.info(f"  {split_name}: {len(split_data)} total — {dict(counts)}")

    return train_all, val_all, test_all


def build_datasets(
    examples: list[dict],
    tokenizer,
    max_seq_length: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Build HuggingFace DatasetDict with stratified train/val/test splits.

    Since smart truncation already guarantees each example fits within
    max_seq_length, we only do a verification check here (no dropping).
    """
    formatted_map = {}
    too_long = 0
    token_lengths = []

    for ex in examples:
        messages = format_chat_messages(ex)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Exact token length check (should pass since we pre-truncated)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_lengths.append(len(token_ids))

        if len(token_ids) > max_seq_length:
            too_long += 1
            logger.warning(
                f"Example {ex['paper_id']} still exceeds max_seq_length "
                f"({len(token_ids)} > {max_seq_length}) after smart truncation — skipping"
            )
            continue

        formatted_map[ex["paper_id"]] = {
            "text": text,
            "paper_id": ex["paper_id"],
            "domain": ex["domain"],
        }

    if too_long > 0:
        logger.warning(
            f"Dropped {too_long} examples still exceeding max_seq_length={max_seq_length}"
        )

    # Log token length statistics
    if token_lengths:
        avg_len = sum(token_lengths) / len(token_lengths)
        logger.info(
            f"Token lengths — avg: {avg_len:.0f}, "
            f"min: {min(token_lengths)}, max: {max(token_lengths)}, "
            f"median: {sorted(token_lengths)[len(token_lengths)//2]}"
        )

    # Filter examples to only those that passed
    valid_examples = [ex for ex in examples if ex["paper_id"] in formatted_map]

    # Debug: print first formatted example
    if formatted_map:
        first_key = next(iter(formatted_map))
        first = formatted_map[first_key]
        logger.info("=== First formatted example (chat template applied) ===")
        logger.info(f"  paper_id: {first['paper_id']}")
        logger.info(f"  domain: {first['domain']}")
        logger.info(f"  text (first 800 chars):\n{first['text'][:800]}")
        logger.info(f"  text (last 400 chars):\n...{first['text'][-400:]}")

    # Stratified split
    logger.info("Performing stratified split by domain...")
    train_ex, val_ex, test_ex = stratified_split(
        valid_examples, val_ratio, test_ratio, seed
    )

    def to_hf_records(split_examples):
        records = []
        for ex in split_examples:
            if ex["paper_id"] in formatted_map:
                records.append(formatted_map[ex["paper_id"]])
        return records

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(to_hf_records(train_ex)),
        "validation": Dataset.from_list(to_hf_records(val_ex)),
        "test": Dataset.from_list(to_hf_records(test_ex)),
    })

    logger.info(
        f"Splits — train: {len(dataset_dict['train'])}, "
        f"val: {len(dataset_dict['validation'])}, "
        f"test: {len(dataset_dict['test'])}"
    )
    return dataset_dict


# =============================================================================
# Model Setup
# =============================================================================

def load_model(
    model_name: str,
    bnb_config: Optional[BitsAndBytesConfig] = None,
    attn_implementation: str = "flash_attention_2",
    trust_remote_code: bool = False,
    max_memory: Optional[dict] = None,
):
    """Load a model, optionally with 4-bit quantization."""
    if bnb_config:
        logger.info(f"Loading model: {model_name} (4-bit quantized)")
    else:
        logger.info(f"Loading model: {model_name} (bf16, no quantization)")

    # Check flash attention availability
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.warning(
            "flash-attn not installed; falling back to sdpa attention. "
            "Install with: pip install flash-attn --no-build-isolation"
        )
        attn_implementation = "sdpa"

    kwargs = dict(
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    if max_memory:
        kwargs["max_memory"] = max_memory
        logger.info(f"Using max_memory: {max_memory}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False  # Incompatible with gradient checkpointing

    return model


def load_tokenizer(model_name: str, max_seq_length: int):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_length

    return tokenizer


# =============================================================================
# Training
# =============================================================================

def train(
    model_name: str,
    dataset_path: str,
    summaries_path: str,
    output_dir: str,
    # QLoRA params
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list[str]] = None,
    # Training params
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 8192,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    max_grad_norm: float = 0.3,
    lr_scheduler_type: str = "cosine",
    # Data params
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    # Misc
    seed: int = 42,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    eval_strategy: str = "epoch",
    resume_from_checkpoint: bool = False,
    merge_and_save: bool = False,
    bf16: bool = True,
    report_to: str = "none",
    quantize: bool = True,
    early_stopping_patience: int = 0,
    max_memory_per_gpu: Optional[str] = None,
):
    """Full QLoRA/LoRA fine-tuning pipeline."""
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    # -------------------------------------------------------------------------
    # 1. Quantization config (optional)
    # -------------------------------------------------------------------------
    bnb_config = None
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # -------------------------------------------------------------------------
    # 2. Load tokenizer first (needed for smart truncation)
    # -------------------------------------------------------------------------
    tokenizer = load_tokenizer(model_name, max_seq_length)

    # -------------------------------------------------------------------------
    # 3. Prepare data (with smart truncation using the tokenizer)
    # -------------------------------------------------------------------------
    examples = load_and_merge_data(
        dataset_path, summaries_path, tokenizer, max_seq_length
    )
    if not examples:
        logger.error("No training examples found. Check your data files.")
        sys.exit(1)

    datasets = build_datasets(
        examples, tokenizer, max_seq_length, val_ratio, test_ratio, seed
    )

    # Save test set metadata for evaluation (paper IDs + domains)
    test_metadata = [
        {"paper_id": pid, "domain": dom}
        for pid, dom in zip(datasets["test"]["paper_id"],
                            datasets["test"]["domain"])
    ]
    test_meta_path = output_path / "test_paper_ids.json"
    with open(test_meta_path, "w") as f:
        json.dump(test_metadata, f, indent=2)
    logger.info(f"Test set metadata saved to {test_meta_path}")

    # Save domain distribution summary
    test_domains = Counter(m["domain"] for m in test_metadata)
    val_domains = Counter(datasets["validation"]["domain"])
    train_domains = Counter(datasets["train"]["domain"])
    split_summary = {
        "train": {"total": len(datasets["train"]), "by_domain": dict(train_domains)},
        "validation": {"total": len(datasets["validation"]), "by_domain": dict(val_domains)},
        "test": {"total": len(datasets["test"]), "by_domain": dict(test_domains)},
    }
    split_path = output_path / "split_summary.json"
    with open(split_path, "w") as f:
        json.dump(split_summary, f, indent=2)
    logger.info(f"Split summary saved to {split_path}")

    # -------------------------------------------------------------------------
    # 4. Load model
    # -------------------------------------------------------------------------
    max_memory = None
    if max_memory_per_gpu:
        num_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory_per_gpu for i in range(num_gpus)}
        logger.info(f"Setting max_memory for {num_gpus} GPUs: {max_memory_per_gpu} each")

    model = load_model(model_name, bnb_config, max_memory=max_memory)

    # -------------------------------------------------------------------------
    # 5. LoRA config
    # -------------------------------------------------------------------------
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -------------------------------------------------------------------------
    # 6. Training arguments
    # -------------------------------------------------------------------------
    use_early_stopping = early_stopping_patience > 0

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to=report_to,
        seed=seed,
        max_length=max_seq_length,
        packing=False,
    )

    # -------------------------------------------------------------------------
    # 7. Train with completion-only loss masking
    # -------------------------------------------------------------------------
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Build callbacks
    callbacks = []
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
        logger.info(f"Early stopping enabled with patience={early_stopping_patience}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
    )

    # Log trainable parameters
    if hasattr(trainer.model, "get_nb_trainable_parameters"):
        trainable, total = trainer.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    # Validate checkpoint exists before attempting resume
    actual_checkpoint = resume_from_checkpoint
    if resume_from_checkpoint is True:
        checkpoints = sorted(output_path.glob("checkpoint-*"))
        if not checkpoints:
            logger.warning(
                "No checkpoints found in %s — starting from scratch", output_path
            )
            actual_checkpoint = False
        else:
            logger.info(f"Resuming from latest checkpoint: {checkpoints[-1]}")

    logger.info("Starting training...")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Batch size: {per_device_batch_size} x {gradient_accumulation_steps} accumulation")
    logger.info(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    logger.info(f"  Max sequence length: {max_seq_length}")

    train_result = trainer.train(
        resume_from_checkpoint=actual_checkpoint
    )

    # Log collator diagnostics
    logger.info(
        f"[Collator Final] Total: {collator.total_seen}, "
        f"Template found: {collator.template_found}, "
        f"Template missing: {collator.template_missing} "
        f"({100*collator.template_missing/max(collator.total_seen,1):.1f}% loss)"
    )

    # -------------------------------------------------------------------------
    # 8. Save
    # -------------------------------------------------------------------------
    adapter_path = output_path / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"Adapter saved to {adapter_path}")

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate on validation set
    logger.info("Running final validation evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Save training config for reproducibility
    config_to_save = {
        "model_name": model_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "quantize": quantize,
        "seed": seed,
        "train_examples": len(datasets["train"]),
        "val_examples": len(datasets["validation"]),
        "test_examples": len(datasets["test"]),
    }
    config_path = output_path / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)
    logger.info(f"Training config saved to {config_path}")

    # Optionally merge adapter into base model
    if merge_and_save:
        logger.info("Merging adapter into base model...")
        merged_path = output_path / "merged"
        merge_adapter(model_name, str(adapter_path), str(merged_path))

    logger.info("Training complete!")
    return trainer


def merge_adapter(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
):
    """Merge LoRA adapter into the base model and save full weights."""
    logger.warning(
        "Merging requires loading the full base model in bf16. "
        "For 70B models this needs ~140GB RAM. "
        "If you OOM, skip --merge_and_save and use the adapter directly."
    )
    logger.info(f"Loading base model for merging: {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except OSError:
        logger.warning(
            f"No tokenizer found in {adapter_path}, loading from {base_model_name}"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    logger.info("Merge complete!")


# =============================================================================
# Inference Helper
# =============================================================================

def generate_summary(
    model_path: str,
    title: str,
    paper_text: str,
    max_new_tokens: int = 512,
    use_adapter: bool = True,
    base_model_name: Optional[str] = None,
    quantize: bool = True,
) -> str:
    """Generate a summary using a fine-tuned model."""
    kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    if use_adapter:
        if not base_model_name:
            raise ValueError("base_model_name required when use_adapter=True")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        if base_model_name:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        else:
            raise
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                f"{INSTRUCTION}\n\nTitle: {title}\n\n"
                f"Paper Content:\n{paper_text}"
            ),
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # apply_chat_template already includes BOS — don't add it again
    inputs = tokenizer(
        input_text, return_tensors="pt", add_special_tokens=False
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning for scientific summarization (v2)"
    )

    # Data
    p.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    p.add_argument("--summaries", required=True, help="Path to summaries.jsonl")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--model", default="meta-llama/Llama-3.3-8B-Instruct",
                    help="HuggingFace model name/path")

    # QLoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", nargs="+", default=None,
                    help="LoRA target modules (default: all projection layers)")

    # Training
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--early_stopping_patience", type=int, default=0,
                    help="Stop if eval_loss doesn't improve for N evals (0=disabled)")

    # Data processing
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--resume_from_checkpoint", action="store_true")
    p.add_argument("--merge_and_save", action="store_true",
                    help="Merge adapter into base model after training")
    p.add_argument("--no_bf16", action="store_true",
                    help="Disable bf16 (use fp16 instead)")
    p.add_argument("--no_quantize", action="store_true",
                    help="Skip 4-bit quantization (use bf16 LoRA instead of QLoRA)")
    p.add_argument("--report_to", default="none",
                    choices=["none", "wandb", "tensorboard"])
    p.add_argument("--max_memory_per_gpu", type=str, default=None,
                    help="Max memory per GPU (e.g., '45GiB') for balanced device_map")

    # Inference mode
    p.add_argument("--inference", action="store_true",
                    help="Run inference instead of training")
    p.add_argument("--adapter_path", type=str,
                    help="Path to adapter for inference")
    p.add_argument("--paper_file", type=str,
                    help="JSON file with title and input_text for inference")

    return p.parse_args()


def main():
    args = parse_args()

    if args.inference:
        if not args.adapter_path or not args.paper_file:
            print("Inference requires --adapter_path and --paper_file")
            sys.exit(1)

        with open(args.paper_file, "r") as f:
            paper = json.load(f)

        summary = generate_summary(
            model_path=args.adapter_path,
            title=paper["title"],
            paper_text=paper["input_text"],
            use_adapter=True,
            base_model_name=args.model,
            quantize=not args.no_quantize,
        )
        print("\n=== Generated Summary ===\n")
        print(summary)
        print(f"\nWord count: {len(summary.split())}")
        return

    train(
        model_name=args.model,
        dataset_path=args.dataset,
        summaries_path=args.summaries,
        output_dir=args.output,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        logging_steps=args.logging_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        merge_and_save=args.merge_and_save,
        bf16=not args.no_bf16,
        report_to=args.report_to,
        quantize=not args.no_quantize,
        early_stopping_patience=args.early_stopping_patience,
        max_memory_per_gpu=args.max_memory_per_gpu,
    )


if __name__ == "__main__":
    main()
