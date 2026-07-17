#!/usr/bin/env python3
"""
QLoRA Fine-Tuning for Scientific Document Summarization

Fine-tunes LLaMA 3.3 (8B or 70B) using QLoRA on teacher-generated summaries
from Claude Opus 4.6, following the research proposal methodology.

Pipeline:
1. Combine dataset.jsonl + summaries.jsonl into instruction-response pairs
2. Apply chat template formatting
3. Fine-tune with optional 4-bit NF4 quantization + LoRA adapters
4. Save merged or adapter-only weights

Usage:
    # 8B model with QLoRA (4-bit quantized, single GPU, ~8GB VRAM)
    python qlora_finetune.py \
        --dataset data/dataset.jsonl \
        --summaries data/summaries.jsonl \
        --model meta-llama/Llama-3.3-8B-Instruct \
        --output ./output/llama-8b-summarizer

    # 8B model with LoRA only (bf16, no quantization, ~25GB VRAM)
    python qlora_finetune.py \
        --dataset data/dataset.jsonl \
        --summaries data/summaries.jsonl \
        --model meta-llama/Llama-3.3-8B-Instruct \
        --output ./output/llama-8b-summarizer-bf16 \
        --no_quantize

    # 70B model (multi-GPU or single A100 80GB, ~48GB VRAM)
    python qlora_finetune.py \
        --dataset data/dataset.jsonl \
        --summaries data/summaries.jsonl \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --output ./output/llama-70b-summarizer \
        --lora_rank 32 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 16

    # Resume from checkpoint
    python qlora_finetune.py \
        --dataset data/dataset.jsonl \
        --summaries data/summaries.jsonl \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --output ./output/llama-70b-summarizer \
        --resume_from_checkpoint

Dependencies:
    pip install torch transformers peft bitsandbytes datasets trl accelerate
"""

import argparse
import json
import os
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
    """

    def __init__(self, response_template: str, tokenizer, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        labels = batch["input_ids"].clone()

        for i in range(len(labels)):
            token_ids = labels[i].tolist()
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
            else:
                # Template not found — mask entire sequence to avoid bad gradients
                labels[i, :] = self.ignore_index

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


# =============================================================================
# Domain helpers
# =============================================================================

def classify_domain(categories: list[str]) -> str:
    """Classify a paper into a broad domain (cs, physics, math) from its arXiv categories."""
    for cat in categories:
        cat_lower = cat.lower()
        # Check exact prefix matches first (e.g., "quant-ph", "cond-mat")
        for prefix, domain in DOMAIN_PREFIXES.items():
            if cat_lower.startswith(prefix):
                return domain
    # Fallback: use the first category's top-level prefix
    if categories:
        top = categories[0].split(".")[0].split("-")[0].lower()
        return DOMAIN_PREFIXES.get(top, "other")
    return "other"


# =============================================================================
# Data Preparation
# =============================================================================

def load_and_merge_data(
    dataset_path: str,
    summaries_path: str,
    max_input_chars: int = 12000,
) -> list[dict]:
    """
    Merge dataset.jsonl and summaries.jsonl into training examples.

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
    skipped = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            pid = paper.get("paper_id", "")
            if pid not in summaries:
                skipped += 1
                continue

            input_text = paper.get("input_text", "")
            if not input_text or len(input_text) < 200:
                skipped += 1
                continue

            # Truncate very long inputs to stay within context window
            if len(input_text) > max_input_chars:
                input_text = input_text[:max_input_chars] + "\n\n[Truncated]"

            categories = paper.get("categories", [])
            domain = classify_domain(categories)

            examples.append({
                "paper_id": pid,
                "title": paper.get("title", "Untitled"),
                "input_text": input_text,
                "output": summaries[pid].lstrip("## Summary\n").strip(),
                "reference_summary": paper.get("reference_summary", ""),
                "categories": categories,
                "domain": domain,
            })

    logger.info(
        f"Created {len(examples)} training examples ({skipped} skipped)"
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

        # Ensure at least 1 example remains for training
        if n_test + n_val >= n:
            n_test = max(1, n // 3)
            n_val = max(1, n // 3)
            if n_test + n_val >= n:
                # Fewer than 3 examples: put all in train, skip test/val
                logger.warning(
                    f"Domain '{domain}' has only {n} examples — "
                    f"assigning all to train (too few to split)"
                )
                train_all.extend(domain_examples)
                continue

        test_all.extend(domain_examples[:n_test])
        val_all.extend(domain_examples[n_test:n_test + n_val])
        train_all.extend(domain_examples[n_test + n_val:])

    # Shuffle each split (so domains aren't grouped together during training)
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    # Log per-domain split sizes
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

    Each example is stored with a 'text' field containing the fully formatted
    chat-template string (including special tokens). SFTTrainer will tokenize this.
    """
    # First, format all examples and filter by length
    formatted_map = {}  # paper_id -> formatted dict
    too_long = 0

    for ex in examples:
        messages = format_chat_messages(ex)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Quick token length estimate (chars / 3.0 is a rough proxy for LLaMA)
        # We do an exact check for borderline cases
        if len(text) / 3.0 > max_seq_length:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > max_seq_length:
                too_long += 1
                continue

        formatted_map[ex["paper_id"]] = {
            "text": text,
            "paper_id": ex["paper_id"],
            "domain": ex["domain"],
        }

    if too_long > 0:
        logger.warning(
            f"Dropped {too_long} examples exceeding max_seq_length={max_seq_length}"
        )

    # Filter examples to only those that passed length check
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

    # Convert to formatted dicts for HuggingFace Dataset
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
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    # Training params
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 4096,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    max_grad_norm: float = 0.3,
    lr_scheduler_type: str = "cosine",
    # Data params
    max_input_chars: int = 12000,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    # Misc
    seed: int = 42,
    logging_steps: int = 10,
    save_strategy: str = "no",
    eval_strategy: str = "epoch",
    resume_from_checkpoint: bool = False,
    merge_and_save: bool = False,
    bf16: bool = True,
    report_to: str = "none",
    quantize: bool = True,
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
    # 2. Load tokenizer and model
    # -------------------------------------------------------------------------
    tokenizer = load_tokenizer(model_name, max_seq_length)
    model = load_model(model_name, bnb_config)

    # -------------------------------------------------------------------------
    # 3. LoRA config (passed to SFTTrainer, not applied manually)
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
    # 4. Prepare data
    # -------------------------------------------------------------------------
    examples = load_and_merge_data(dataset_path, summaries_path, max_input_chars)
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

    # Also save domain distribution summary
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
    # 5. Training arguments (TRL >= 1.0 API)
    # -------------------------------------------------------------------------
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
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to=report_to,
        seed=seed,
        # TRL >= 1.0: use max_length instead of max_seq_length
        max_length=max_seq_length,
        # TRL >= 1.0: dataset_text_field removed; 'text' column auto-detected
        packing=False,  # Don't pack; variable-length summaries
    )

    # -------------------------------------------------------------------------
    # 6. Train
    #    Use DataCollatorForCompletionOnlyLM so the loss is computed only on
    #    the assistant's response (the summary), not on the system prompt,
    #    instruction, or paper content.
    # -------------------------------------------------------------------------
    # The response template is the token sequence that marks the start of the
    # assistant turn in LLaMA 3's chat format.
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
        data_collator=collator,
    )

    # Log trainable parameters after SFTTrainer wraps the model with PEFT
    if hasattr(trainer.model, "get_nb_trainable_parameters"):
        trainable, total = trainer.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    # Validate checkpoint exists before attempting resume (#4)
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
    train_result = trainer.train(
        resume_from_checkpoint=actual_checkpoint
    )

    # -------------------------------------------------------------------------
    # 7. Save
    # -------------------------------------------------------------------------
    # Save adapter weights
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
    """Merge LoRA adapter into the base model and save full weights.

    WARNING: This loads the full model in bf16, which requires ~2x the model's
    parameter count in bytes (e.g., ~140GB for 70B). Only use if you have
    sufficient CPU/GPU memory.
    """
    # Estimate memory needed (2 bytes per param for bf16)
    logger.warning(
        "Merging requires loading the full base model in bf16. "
        "For 70B models this needs ~140GB RAM. "
        "If you OOM, skip --merge_and_save and use the adapter directly."
    )
    logger.info(f"Loading base model for merging: {base_model_name}")

    # Load in 16-bit for merging (no quantization)
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

    # Load tokenizer from adapter path, fall back to base model
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
    """
    Generate a summary using a fine-tuned model.

    Args:
        model_path: Path to adapter or merged model
        title: Paper title
        paper_text: Paper content
        max_new_tokens: Max tokens to generate
        use_adapter: If True, load as adapter on top of base_model_name
        base_model_name: Required if use_adapter=True
        quantize: If True, load with 4-bit quantization; otherwise bf16
    """
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
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # Decode only the generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return summary


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning for scientific summarization"
    )

    # Data
    p.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    p.add_argument("--summaries", required=True, help="Path to summaries.jsonl")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--model", default="meta-llama/Llama-3.3-8B-Instruct",
                    help="HuggingFace model name/path")

    # QLoRA
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="+", default=None,
                    help="LoRA target modules (default: all projection layers)")

    # Training
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=0.01)

    # Data processing
    p.add_argument("--max_input_chars", type=int, default=12000)
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
        max_input_chars=args.max_input_chars,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        logging_steps=args.logging_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        merge_and_save=args.merge_and_save,
        bf16=not args.no_bf16,
        report_to=args.report_to,
        quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()