#!/usr/bin/env python3
"""
Interactive inference script for debugging and testing summarization models.

Load a base model with or without a LoRA adapter, inspect prompts, generate
summaries, compare base vs fine-tuned outputs side by side, and toggle the
adapter on/off — all from a single interactive session.

Usage:
    # 8B with LoRA adapter
    python interactive_inference.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/adapter

    # 70B with QLoRA
    python interactive_inference.py \
        --base_model meta-llama/Llama-3.3-70B-Instruct \
        --adapter_path /scratch/jam5cq/Summary_Fine_Tuning/Output/adapter \
        --use_qlora

    # Base model only (no adapter)
    python interactive_inference.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct

    # Load test data for quick paper selection
    python interactive_inference.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/adapter \
        --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
        --summaries /scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl \
        --test_ids /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/test_paper_ids.json
"""

import argparse
import json
import os
import textwrap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

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


# ---------------------------------------------------------------------------
# Data loading helpers
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


def load_test_data(dataset_path, summaries_path, test_ids_path):
    """Load test examples if paths are provided."""
    test_meta = load_json(test_ids_path)
    test_id_to_domain = {item["paper_id"]: item["domain"] for item in test_meta}
    test_ids = set(test_id_to_domain.keys())

    dataset = load_jsonl(dataset_path)
    dataset_by_id = {r["paper_id"]: r for r in dataset}

    teacher_records = load_jsonl(summaries_path)
    teacher_by_id = {}
    for r in teacher_records:
        if r.get("success", False):
            teacher_by_id[r["paper_id"]] = r

    examples = {}
    for paper_id in sorted(test_ids):
        if paper_id not in dataset_by_id:
            continue
        d = dataset_by_id[paper_id]
        t = teacher_by_id.get(paper_id)
        teacher_summary = ""
        if t:
            teacher_summary = t["generated_summary"]
            if teacher_summary.startswith("## Summary"):
                teacher_summary = teacher_summary[len("## Summary"):].lstrip("\n").strip()

        examples[paper_id] = {
            "paper_id": paper_id,
            "domain": test_id_to_domain[paper_id],
            "title": d["title"],
            "input_text": d["input_text"],
            "abstract": d.get("reference_summary", ""),
            "teacher_summary": teacher_summary,
        }
    return examples


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_messages(title, input_text):
    """Build the chat messages list."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n{input_text}"},
    ]


def build_prompt(tokenizer, title, input_text):
    """Build the full prompt string using the tokenizer's chat template."""
    messages = build_messages(title, input_text)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _truncate_input_text(tokenizer, title, input_text, max_seq_length=8192):
    """
    Truncate paper content so the full prompt fits within max_seq_length.
    Truncates the paper body, not the final token sequence, so the instruction
    header and assistant generation header are always preserved.
    """
    overhead_messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {title}\n\nPaper Content:\n"},
    ]
    overhead_prompt = tokenizer.apply_chat_template(
        overhead_messages, tokenize=False, add_generation_prompt=True
    )
    overhead_tokens = len(tokenizer.encode(overhead_prompt, add_special_tokens=False))

    content_budget = max_seq_length - overhead_tokens - 10
    if content_budget <= 0:
        return input_text[:1000]

    content_ids = tokenizer.encode(input_text, add_special_tokens=False)
    if len(content_ids) > content_budget:
        content_ids = content_ids[:content_budget]
        input_text = tokenizer.decode(content_ids, skip_special_tokens=True)
        input_text += "\n\n[Truncated]"

    return input_text


def generate(model, tokenizer, title, input_text, max_new_tokens=512,
             temperature=0.7, top_p=0.9, repetition_penalty=1.1, do_sample=True):
    """Generate a summary and return (summary_text, prompt_text, n_prompt_tokens)."""
    # Truncate paper content (not the prompt) to preserve instruction + generation header
    input_text = _truncate_input_text(tokenizer, title, input_text, max_seq_length=8192)

    messages = build_messages(title, input_text)

    # Get the prompt string — chat template already includes BOS token
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize with add_special_tokens=False to avoid duplicate BOS
    inputs = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.device)
    n_prompt_tokens = input_ids.shape[1]

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
        )

    generated_ids = output_ids[0][n_prompt_tokens:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return summary, prompt, n_prompt_tokens


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def wrap_print(text, width=80, prefix="  "):
    """Print wrapped text with prefix."""
    for line in textwrap.wrap(text, width=width):
        print(f"{prefix}{line}")


def print_divider(char="─", width=80):
    print(char * width)


def print_header(text, width=80):
    print()
    print("═" * width)
    print(f"  {text}")
    print("═" * width)


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------
class InteractiveSession:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.base_tokenizer = None  # tokenizer from base model (for comparison)
        self.adapter_loaded = False
        self.adapter_path = args.adapter_path
        self.examples = {}
        self.selected = None
        self.gen_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "repetition_penalty": 1.1,
        }

        # Load model
        self._load_model()

        # Load test data if available
        if args.dataset and args.test_ids:
            print("Loading test data...")
            self.examples = load_test_data(
                args.dataset,
                args.summaries or "",
                args.test_ids,
            )
            print(f"  Loaded {len(self.examples)} test examples")

    def _load_model(self):
        """Load the base model and optionally the adapter."""
        args = self.args
        print(f"\nLoading base model: {args.base_model}")
        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

        if args.use_qlora:
            print("  Using QLoRA 4-bit quantization")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
        self.base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        if args.adapter_path:
            print(f"  Loading adapter from: {args.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, args.adapter_path)
            self.tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
            self.adapter_loaded = True
            print("  Adapter loaded and ACTIVE")
        else:
            self.tokenizer = self.base_tokenizer
            print("  No adapter (base model only)")

        self.model.eval()
        print("  Model ready.\n")

    def cmd_help(self):
        """Print available commands."""
        print_header("COMMANDS")
        cmds = [
            ("help", "Show this help"),
            ("list", "List available test papers"),
            ("paper <id>", "Select a test paper by ID (partial match OK)"),
            ("generate", "Generate summary for the selected paper"),
            ("gen <id>", "Shortcut: select paper + generate in one step"),
            ("compare", "Generate with adapter ON and OFF, show side by side"),
            ("prompt", "Show the full prompt text for the selected paper"),
            ("tokens", "Show prompt tokens with IDs (first/last 30)"),
            ("template", "Show the raw chat_template string from tokenizer"),
            ("template_diff", "Compare adapter vs base tokenizer chat templates"),
            ("adapter_on", "Enable the LoRA adapter"),
            ("adapter_off", "Disable the LoRA adapter (use base model)"),
            ("adapter_status", "Show whether adapter is active"),
            ("set <param> <value>", "Set generation param (temperature, top_p, max_tokens, repetition_penalty)"),
            ("params", "Show current generation parameters"),
            ("info", "Show selected paper metadata and references"),
            ("quit / exit", "Exit the session"),
        ]
        for cmd, desc in cmds:
            print(f"  {cmd:<25} {desc}")
        print()

    def cmd_list(self):
        """List available test papers."""
        if not self.examples:
            print("  No test data loaded. Use --dataset and --test_ids flags.")
            return
        print_header(f"TEST PAPERS ({len(self.examples)} total)")
        for i, (pid, ex) in enumerate(self.examples.items()):
            title_short = ex["title"][:60] + "..." if len(ex["title"]) > 60 else ex["title"]
            print(f"  [{ex['domain']:<8}] {pid:<18} {title_short}")
            if i >= 29:
                remaining = len(self.examples) - 30
                if remaining > 0:
                    print(f"  ... and {remaining} more. Use 'paper <id>' with partial ID.")
                break
        print()

    def cmd_select_paper(self, paper_id):
        """Select a paper by ID (supports partial match)."""
        if not self.examples:
            print("  No test data loaded.")
            return None

        # Exact match first
        if paper_id in self.examples:
            self.selected = self.examples[paper_id]
            print(f"  Selected: {paper_id} — {self.selected['title'][:70]}")
            return self.selected

        # Partial match
        matches = [pid for pid in self.examples if paper_id in pid]
        if len(matches) == 1:
            self.selected = self.examples[matches[0]]
            print(f"  Selected: {matches[0]} — {self.selected['title'][:70]}")
            return self.selected
        elif len(matches) > 1:
            print(f"  Ambiguous ID '{paper_id}'. Matches:")
            for m in matches[:10]:
                print(f"    {m}")
            return None
        else:
            print(f"  No paper found matching '{paper_id}'")
            return None

    def cmd_generate(self, paper=None):
        """Generate a summary for the selected paper."""
        paper = paper or getattr(self, "selected", None)
        if not paper:
            print("  No paper selected. Use 'paper <id>' first.")
            return None

        adapter_status = "ADAPTER ON" if self.adapter_loaded else "BASE MODEL"
        print_header(f"GENERATING ({adapter_status})")
        print(f"  Paper: {paper['paper_id']}")
        print(f"  Title: {paper['title'][:80]}")
        print(f"  Input length: {len(paper['input_text'].split())} words")
        print()

        summary, prompt, n_tokens = generate(
            self.model, self.tokenizer,
            paper["title"], paper["input_text"],
            max_new_tokens=self.gen_params.get("max_tokens", 512),
            temperature=self.gen_params.get("temperature", 0.7),
            top_p=self.gen_params.get("top_p", 0.9),
            repetition_penalty=self.gen_params.get("repetition_penalty", 1.1),
        )

        print(f"  Prompt tokens: {n_tokens}")
        print(f"  Summary words: {len(summary.split())}")
        print()
        print_divider()
        print("  GENERATED SUMMARY:")
        print_divider()
        wrap_print(summary)
        print_divider()
        print()
        return summary

    def cmd_compare(self):
        """Generate with adapter ON and OFF, show side by side."""
        paper = getattr(self, "selected", None)
        if not paper:
            print("  No paper selected. Use 'paper <id>' first.")
            return
        if not self.adapter_path:
            print("  No adapter loaded — cannot compare.")
            return

        gen_kwargs = dict(
            max_new_tokens=self.gen_params.get("max_tokens", 512),
            temperature=self.gen_params.get("temperature", 0.7),
            top_p=self.gen_params.get("top_p", 0.9),
            repetition_penalty=self.gen_params.get("repetition_penalty", 1.1),
        )

        # Generate with adapter ON
        if not self.adapter_loaded:
            self.model.enable_adapter_layers()
            self.adapter_loaded = True
        print("\n  Generating with ADAPTER ON...")
        summary_ft, _, n_tok_ft = generate(
            self.model, self.tokenizer,
            paper["title"], paper["input_text"],
            **gen_kwargs,
        )

        # Generate with adapter OFF
        self.model.disable_adapter_layers()
        self.adapter_loaded = False
        print("  Generating with BASE MODEL...")
        summary_base, _, n_tok_base = generate(
            self.model, self.base_tokenizer,
            paper["title"], paper["input_text"],
            **gen_kwargs,
        )

        # Re-enable adapter
        self.model.enable_adapter_layers()
        self.adapter_loaded = True

        # Display
        print_header(f"COMPARISON: {paper['paper_id']}")

        print(f"\n  FINE-TUNED (adapter ON) — {len(summary_ft.split())} words, {n_tok_ft} prompt tokens:")
        print_divider("─")
        wrap_print(summary_ft)
        print_divider("─")

        print(f"\n  BASE MODEL (adapter OFF) — {len(summary_base.split())} words, {n_tok_base} prompt tokens:")
        print_divider("─")
        wrap_print(summary_base)
        print_divider("─")

        if paper.get("abstract"):
            print(f"\n  REFERENCE ABSTRACT — {len(paper['abstract'].split())} words:")
            print_divider("─")
            wrap_print(paper["abstract"])
            print_divider("─")

        if paper.get("teacher_summary"):
            print(f"\n  TEACHER (Claude Opus) — {len(paper['teacher_summary'].split())} words:")
            print_divider("─")
            wrap_print(paper["teacher_summary"])
            print_divider("─")
        print()

    def cmd_prompt(self):
        """Show the full prompt for the selected paper."""
        paper = getattr(self, "selected", None)
        if not paper:
            print("  No paper selected.")
            return

        prompt = build_prompt(self.tokenizer, paper["title"], paper["input_text"])
        print_header("FULL PROMPT (adapter tokenizer)")
        print(prompt[:2000])
        if len(prompt) > 2000:
            print(f"\n  ... [{len(prompt) - 2000} chars truncated] ...")
            print(prompt[-500:])
        print()

        # Also show with base tokenizer if different
        if self.base_tokenizer and self.adapter_path:
            base_prompt = build_prompt(self.base_tokenizer, paper["title"], paper["input_text"])
            if base_prompt != prompt:
                print_header("FULL PROMPT (base tokenizer) — DIFFERENT!")
                print(base_prompt[:2000])
                if len(base_prompt) > 2000:
                    print(f"\n  ... [{len(base_prompt) - 2000} chars truncated] ...")
                    print(base_prompt[-500:])
                print()
            else:
                print("  (Base tokenizer produces identical prompt.)\n")

    def cmd_tokens(self):
        """Show prompt token IDs and decoded tokens."""
        paper = getattr(self, "selected", None)
        if not paper:
            print("  No paper selected.")
            return

        messages = build_messages(paper["title"], paper["input_text"])

        # Get the prompt string (includes BOS from chat template)
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # CORRECT: encode without adding another BOS
        token_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        # BUGGY (old method): encode with default add_special_tokens=True → double BOS
        old_token_ids = self.tokenizer.encode(prompt_str)

        if len(old_token_ids) != len(token_ids):
            print(f"\n  ⚠ BOS BUG DETECTED:")
            print(f"    encode(add_special_tokens=False): {len(token_ids)} tokens  ← CORRECT")
            print(f"    encode(add_special_tokens=True):  {len(old_token_ids)} tokens  ← OLD/BUGGY")
            print(f"    Difference: {len(old_token_ids) - len(token_ids)} extra token(s)")
            print(f"    Old first 3: {old_token_ids[:3]}")
            print(f"    New first 3: {token_ids[:3]}")
            print()

        print_header(f"PROMPT TOKENS — CORRECT ({len(token_ids)} total)")
        show_n = 30
        for i, tid in enumerate(token_ids[:show_n]):
            decoded = self.tokenizer.decode([tid])
            print(f"  [{i:>5}] {tid:>8}  →  {repr(decoded)}")
        if len(token_ids) > show_n * 2:
            print(f"\n  ... ({len(token_ids) - show_n * 2} tokens omitted) ...\n")
        for i, tid in enumerate(token_ids[-show_n:]):
            idx = len(token_ids) - show_n + i
            decoded = self.tokenizer.decode([tid])
            print(f"  [{idx:>5}] {tid:>8}  →  {repr(decoded)}")

        print(f"\n  Total prompt tokens: {len(token_ids)}")
        print(f"  Last 5 tokens (generation starts after these):")
        for tid in token_ids[-5:]:
            decoded = self.tokenizer.decode([tid])
            print(f"    {tid:>8}  →  {repr(decoded)}")
        print()

    def cmd_template(self):
        """Show the raw chat template."""
        print_header("CHAT TEMPLATE (adapter tokenizer)")
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            print(template)
        else:
            print("  No chat_template attribute found.")
        print()

    def cmd_template_diff(self):
        """Compare adapter vs base tokenizer templates."""
        if not self.adapter_path:
            print("  No adapter loaded — nothing to compare.")
            return

        adapter_template = getattr(self.tokenizer, "chat_template", "")
        base_template = getattr(self.base_tokenizer, "chat_template", "")

        if adapter_template == base_template:
            print("\n  ✓ Chat templates are IDENTICAL between adapter and base tokenizer.\n")
        else:
            print_header("TEMPLATE DIFFERENCE DETECTED")
            print("\n  ADAPTER TOKENIZER TEMPLATE:")
            print_divider("─")
            print(adapter_template or "  (none)")
            print_divider("─")
            print("\n  BASE TOKENIZER TEMPLATE:")
            print_divider("─")
            print(base_template or "  (none)")
            print_divider("─")
            print()

        # Also compare special tokens
        print("  Special tokens comparison:")
        for attr in ["bos_token", "eos_token", "pad_token", "unk_token"]:
            a_val = getattr(self.tokenizer, attr, None)
            b_val = getattr(self.base_tokenizer, attr, None)
            match = "✓" if a_val == b_val else "✗ MISMATCH"
            print(f"    {attr:<15} adapter={repr(a_val):<20} base={repr(b_val):<20} {match}")
        print()

    def cmd_adapter_on(self):
        if not self.adapter_path:
            print("  No adapter was loaded at startup.")
            return
        self.model.enable_adapter_layers()
        self.adapter_loaded = True
        print("  Adapter ENABLED.\n")

    def cmd_adapter_off(self):
        if not self.adapter_path:
            print("  No adapter was loaded at startup.")
            return
        self.model.disable_adapter_layers()
        self.adapter_loaded = False
        print("  Adapter DISABLED (using base model).\n")

    def cmd_info(self):
        """Show selected paper metadata."""
        paper = getattr(self, "selected", None)
        if not paper:
            print("  No paper selected.")
            return
        print_header(f"PAPER INFO: {paper['paper_id']}")
        print(f"  Domain:       {paper['domain']}")
        print(f"  Title:        {paper['title']}")
        print(f"  Input length: {len(paper['input_text'].split())} words / {len(paper['input_text'])} chars")
        if paper.get("abstract"):
            print(f"  Abstract:     {len(paper['abstract'].split())} words")
        if paper.get("teacher_summary"):
            print(f"  Teacher summ: {len(paper['teacher_summary'].split())} words")
        print()

    def run(self):
        """Main REPL loop."""
        print_header("INTERACTIVE INFERENCE SESSION")
        adapter_status = "ADAPTER ON" if self.adapter_loaded else "BASE MODEL ONLY"
        print(f"  Model:   {self.args.base_model}")
        print(f"  Adapter: {self.args.adapter_path or 'none'}")
        print(f"  Status:  {adapter_status}")
        print(f"  Papers:  {len(self.examples)} loaded")
        print(f"\n  Type 'help' for commands.\n")

        while True:
            try:
                raw = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not raw:
                continue

            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                print("Exiting.")
                break
            elif cmd == "help":
                self.cmd_help()
            elif cmd == "list":
                self.cmd_list()
            elif cmd == "paper":
                if arg:
                    self.cmd_select_paper(arg)
                else:
                    print("  Usage: paper <id>")
            elif cmd == "generate" or cmd == "g":
                self.cmd_generate()
            elif cmd == "gen":
                if arg:
                    paper = self.cmd_select_paper(arg)
                    if paper:
                        self.cmd_generate(paper)
                else:
                    print("  Usage: gen <paper_id>")
            elif cmd == "compare":
                self.cmd_compare()
            elif cmd == "prompt":
                self.cmd_prompt()
            elif cmd == "tokens":
                self.cmd_tokens()
            elif cmd == "template":
                self.cmd_template()
            elif cmd == "template_diff":
                self.cmd_template_diff()
            elif cmd == "adapter_on":
                self.cmd_adapter_on()
            elif cmd == "adapter_off":
                self.cmd_adapter_off()
            elif cmd == "adapter_status":
                status = "ACTIVE (adapter ON)" if self.adapter_loaded else "INACTIVE (base model)"
                print(f"  Adapter: {status}\n")
            elif cmd == "set":
                self._cmd_set(arg)
            elif cmd == "params":
                print(f"  Current generation parameters:")
                for k, v in self.gen_params.items():
                    print(f"    {k}: {v}")
                print()
            elif cmd == "info":
                self.cmd_info()
            else:
                print(f"  Unknown command: '{cmd}'. Type 'help' for options.")

    def _cmd_set(self, arg):
        """Set a generation parameter."""
        parts = arg.split()
        if len(parts) != 2:
            print("  Usage: set <param> <value>")
            print(f"  Available: {', '.join(self.gen_params.keys())}")
            return
        param, value = parts
        if param not in self.gen_params:
            print(f"  Unknown param '{param}'. Available: {', '.join(self.gen_params.keys())}")
            return
        try:
            if param == "max_tokens":
                self.gen_params[param] = int(value)
            else:
                self.gen_params[param] = float(value)
            print(f"  Set {param} = {self.gen_params[param]}\n")
        except ValueError:
            print(f"  Invalid value: {value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive inference for summarization models")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset.jsonl for loading test papers")
    parser.add_argument("--summaries", type=str, default=None,
                        help="Path to summaries.jsonl (teacher summaries)")
    parser.add_argument("--test_ids", type=str, default=None,
                        help="Path to test_paper_ids.json")
    args = parser.parse_args()

    session = InteractiveSession(args)
    session.run()


if __name__ == "__main__":
    main()