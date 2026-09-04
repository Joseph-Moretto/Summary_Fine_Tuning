# training/

Fine-tuning of Llama 3 on the teacher summaries. Two versions of the script
are kept because the two reported runs used different ones; the job scripts
reproduce those runs.

| File | Purpose |
|---|---|
| `finetune_v1.py` | first version; used for the Llama 3.1 8B run (bf16 LoRA) |
| `finetune_v2.py` | second version with token-budgeted truncation, per-epoch checkpoints and 70B-oriented defaults; used for the Llama 3.3 70B run (4-bit QLoRA) |
| `run_finetune_8b.slurm` | SLURM job for the 8B run: `finetune_v1.py`, one A6000, 100 GB RAM |
| `run_finetune_70b.slurm` | SLURM job for the 70B run: `finetune_v2.py`, two A6000s, 160 GB RAM |

Submit from the repository root (`sbatch training/run_finetune_70b.slurm`).
The public `data/dataset.jsonl` carries paper text only for papers whose
arXiv license permits redistribution; the scripts skip papers without text,
so a run on the public file trains on that subset. `data/README.md` explains
how to rebuild the full file used for the reported runs.
Outputs go to `runs/<name>/`, which is ignored by git; the metadata of the
reported runs is preserved under `results/<run>/train/`.

## What a run does

1. Load and join the data. `data/summaries.jsonl` is read first and only
   records with `success: true` and a non-empty summary are kept. Each paper in
   `data/dataset.jsonl` is joined on `paper_id`; papers without a summary or
   with fewer than 200 characters of text are skipped (none were, in either
   run). The `## Summary` header some teacher summaries start with is stripped
   (see the caveat below). Each paper is assigned a domain from its arXiv
   categories: `cs`; `math` or `stat`; or `physics`, `quant-ph`, `astro-ph`,
   `cond-mat`, `gr-qc`, `hep`, `nucl`, `nlin`. The first category with a known
   prefix wins.

2. Build the prompt. Every example becomes a three-turn chat rendered with
   the model's own chat template:

   ```
   system:    You are an expert scientific paper summarizer. Generate concise, accurate
              summaries of scientific papers. Write in third person, present tense, using
              flowing prose without bullet points.
   user:      Summarize the following scientific paper in 150-250 words, covering the
              main contribution, methodology, key results, and significance.

              Title: <title>

              Paper Content:
              <input_text, possibly truncated>
   assistant: <teacher summary>
   ```

3. Fit the paper into the context window. This is where the two versions
   differ; see the next section.

4. Split. Papers are shuffled per domain with seed 42 and each domain
   contributes 10% to test and 10% to validation (rounded, at least one each),
   the rest to train. The three splits are then shuffled again. With the
   released data this gives 1,200 / 150 / 150 and an identical test set in
   both runs. The test papers and their domains are written to
   `test_paper_ids.json`, which the evaluation scripts consume.

5. Train with TRL's `SFTTrainer` and a LoRA config passed as
   `peft_config`. A custom collator masks every label up to and including the
   assistant header (`<|start_header_id|>assistant<|end_header_id|>\n\n`), so
   the loss covers only the summary tokens. If the header is not found in a
   sequence the whole sequence is masked; `finetune_v2.py` counts these and
   the 70B run reported zero. Gradient checkpointing (non-reentrant) is on,
   `use_cache` is off, packing is off, the schedule is cosine with linear
   warmup, and the validation loss is computed once per epoch.

6. Save. The adapter goes to `<output>/adapter/` together with the
   tokenizer, and trainer metrics to `train_results.json`,
   `eval_results.json`, `all_results.json` and `trainer_state.json`.
   `--merge_and_save` additionally merges the adapter into the base model in
   bf16 (about 140 GB of memory for 70B); it was not used.

## Handling papers longer than the context window

Both runs used a maximum sequence length of 8,192 tokens.

`finetune_v1.py` cuts `input_text` at `--max_input_chars` characters
(default 12,000, about 3,000 tokens) and appends `[Truncated]`. After the chat
template is applied, examples whose token count still exceeds
`--max_seq_length` are dropped. With the 12,000-character cut no example was
dropped in the 8B run.

`finetune_v2.py` keeps much more of each paper. For every example it
renders the full chat with an empty paper body, counts those tokens, and
gives the paper `max_seq_length - overhead - 50` tokens (never fewer than
256). Papers that fit are left alone. Papers that do not are cut to
roughly the first 70% of the budget and the last 30% (minus 5 tokens for
the marker), so the introduction and the results/conclusion both survive.
The head is trimmed back to the last section heading, paragraph break or
sentence end in its final 20%, the tail is advanced to the first such
boundary in its initial 20%, and the two are joined with
`\n\n[...TRUNCATED...]\n\n`. A final check hard-cuts anything still over
budget. In the 70B run this affected 163 of 1,500 papers; per-example
budgets ranged from 7,616 to 7,770 tokens and the finished sequences
averaged 4,474 tokens (median 4,139, longest 8,137).

The same function is copied into `evaluation/evaluate_train_aligned.py` so
inference can truncate identically. `evaluation/evaluate.py`, which produced
the reported results, uses a simpler head-only cut instead.

## Configuration

Defaults of the two scripts and the values used for the reported runs.

| Option | v1 default | v2 default | 8B run (v1) | 70B run (v2) |
|---|---|---|---|---|
| `--lora_rank` / `--lora_alpha` | 32 / 64 | 16 / 32 | 16 / 32 | 16 / 32 |
| `--lora_dropout` | 0.05 | 0.1 | 0.05 | 0.1 |
| `--target_modules` | q, k, v, o, gate, up, down proj | same | same | same |
| quantization | 4-bit NF4 unless `--no_quantize` | same | none (`--no_quantize`) | 4-bit NF4, double quant, bf16 compute |
| `--learning_rate` | 2e-4 | 1e-4 | 2e-4 | 1e-4 |
| `--lr_scheduler_type` / `--warmup_ratio` | cosine / 0.03 | cosine / 0.05 | cosine / 0.03 | cosine / 0.05 |
| `--weight_decay` / `--max_grad_norm` | 0.01 / 0.3 | 0.01 / 0.3 | same | same |
| `--per_device_batch_size` x `--gradient_accumulation_steps` | 4 x 4 | 4 x 4 | 4 x 4 | 1 x 16 |
| `--num_epochs` | 3 | 3 | 3 | 3 |
| `--max_seq_length` | 4096 | 8192 | 8192 | 8192 |
| `--max_input_chars` | 12,000 | (token budget instead) | 12,000 | |
| checkpoints | none | every epoch, best eval loss restored | none | every epoch |
| `--early_stopping_patience` | | 0 (off) | | 0 |
| `--max_memory_per_gpu` | | none | | 45GiB |
| `--seed` | 42 | 42 | 42 | 42 |

Other options shared by both scripts: `--val_ratio` and `--test_ratio`
(0.1), `--logging_steps` (10), `--resume_from_checkpoint`, `--merge_and_save`,
`--no_bf16` (fp16 instead), `--report_to` (`none`, `wandb`, `tensorboard`),
and a minimal `--inference` mode for a single paper given as JSON with
`title` and `input_text`.

Both runs had an effective batch size of 16 and 75 optimizer steps per
epoch (225 in total).

## The two reported runs

| | Llama 3.1 8B Instruct | Llama 3.3 70B Instruct |
|---|---|---|
| SLURM job, date | 12228610, 2026-04-26 | 12501039, 2026-05-04 to 05-05 |
| Hardware | 1x RTX A6000 48 GB, 16 CPUs, 100 GB RAM | 2x RTX A6000 48 GB, 16 CPUs, 160 GB RAM |
| Wall time | 3 h 45 min (3 h 41 min in `trainer.train`) | 24 h 08 min (23 h 47 min in `trainer.train`) |
| Trainable parameters | 41.9 M of 8.07 B (0.52%) | 207.1 M of 70.8 B (0.29%) |
| Train loss (mean over run) | 0.369 | 0.570 |
| Validation loss per epoch | 0.491, 0.494, 0.529 | 0.652, 0.634, 0.667 |
| Saved adapter | final weights (epoch 3) | epoch-2 checkpoint (lowest validation loss) |
| Papers truncated | 0 dropped after the 12,000-char cut | 163 of 1,500 |

The 8B script has no checkpointing, so the saved adapter is the end of
training even though validation loss was lowest after epoch 1. The 70B
script restores the best checkpoint, so its `eval_results.json` reports the
epoch-2 loss. `trainer_state.json` in `results/<run>/train/` holds the full
loss, gradient-norm, learning-rate and token-accuracy trace every 10 steps.

## Multi-GPU and memory

Models are loaded with `device_map="auto"`, which splits the 70B model across
the two GPUs. `--max_memory_per_gpu 45GiB` caps what `accelerate` may place on
each 48 GB card so that activations and optimizer state fit. The 8B model in
bf16 with LoRA, batch size 4 and 8,192-token sequences fits on a single
48 GB card with gradient checkpointing. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
in the job scripts reduces fragmentation for the long, variable-length
sequences.

## Caveats

- Teacher header stripping. Both scripts remove the `## Summary` header
  with `str.lstrip("## Summary\n")`, which strips a set of characters rather
  than a prefix. For 8 of the 1,500 summaries that begin with an S, u, m, a,
  r or y (for example `SPAC ...`, `StructMem ...`) the first letters of the
  training target were clipped. The line is left as it was, with a comment,
  so that the released adapters can be reproduced; the evaluation scripts use
  a proper prefix check.
- Attention implementation. The scripts request `flash_attention_2` and
  fall back to SDPA when `flash-attn` is not importable, which was the case
  for both runs.
- TRL version. The code targets TRL 1.1 (`SFTConfig(max_length=...)`,
  `processing_class=`); the completion-only collator replaces
  `DataCollatorForCompletionOnlyLM`, which TRL removed.
- Checkpoint layout. `--resume_from_checkpoint` looks for
  `checkpoint-*` directories in `--output` and starts fresh if none exist.
