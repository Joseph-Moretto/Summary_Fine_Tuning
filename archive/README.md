# archive/

Superseded runs, pilot data and scripts that were replaced during the
project. Kept for completeness; nothing here is needed to reproduce the
reported results, and nothing in `training/` or `evaluation/` reads from it.
The adapter weight files that belong here exceed GitHub's 100 MB limit and are
not in the repository.

## adapters/

Complete PEFT adapter directories for the three trained adapters, minus the
weight file itself (`adapter_model.safetensors`, excluded by `.gitignore`):
`adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `training_args.bin` (pickled `TrainingArguments`) and
the auto-generated model card.

| Directory | Adapter | Weights |
|---|---|---|
| `llama3.1-8b-lora/` | the reported 8B adapter (`finetune_v1.py`, job 12228610) | fp32, 41.9 M parameters, 168 MB |
| `llama3.3-70b-qlora-v1/` | first 70B attempt with `finetune_v1.py` (2026-04-26), superseded | bf16, 207.1 M parameters, 414 MB |
| `llama3.3-70b-qlora-v2/` | the reported 70B adapter (`finetune_v2.py`, job 12501039) | fp32, 207.1 M parameters, 829 MB |

With the weight file in place, each directory loads with
`PeftModel.from_pretrained(base_model, "<dir>")`.

## llama3.3-70b-qlora-v1/

The first 70B run, trained with `finetune_v1.py` and the v1 defaults for
truncation (12,000-character cut) on 2026-04-26: train loss 0.631,
validation loss 0.949.

- `train/`: trainer metrics (`all_results.json`, `eval_results.json`,
  `train_results.json`, `trainer_state.json`).
- `eval_2026-04-28/`: generations from the first evaluation.
- `eval_2026-05-02/`: generations, `eval_results.json` and `eval_report.txt`
  from a second evaluation (job 12405784), in which the base model returned an
  empty summary for 56 of the 150 papers, so its metrics rest on 94 papers.

The run was replaced by the v2 run in `results/llama3.3-70b-qlora/`.

## llama3.1-8b-lora/eval_2026-04-30/

The first evaluation of the reported 8B adapter (job 12394065, log included),
made with the earlier `evaluate.py` in `standalone_eval_scripts/`. It holds
that run's generations, `eval_results.json` and `eval_report.txt`, a
`ldfact_results.json` computed separately on 2026-05-02 (job 12418200), and a
`summac_results.json` from the SummaC attempt. The generations were
regenerated on 2026-05-03 with the merged pipeline; those are the ones in
`results/`.

## standalone_eval_scripts/

The evaluation code before the analyses were merged into
`evaluation/evaluate.py`:

| File | Purpose |
|---|---|
| `evaluate.py` | inference and reference metrics, with a SummaC hook; used the training prompt |
| `significance_test.py` | paired bootstrap (resampling variant) and Wilcoxon tests on per-paper scores |
| `ldfact_eval.py`, `run_ldfact.sh` | standalone LongDocFACTScore scoring and its SLURM job (log in `logs/`) |
| `abstractiveness_analysis.py` | novel n-grams, coverage, density, compression ratio, type-token ratio, sentence statistics |

The merged pipeline reimplements these; the permutation test in
`evaluate.py` replaced the resampling bootstrap here.

## summac/

SummaC factual-consistency scoring (`summac_eval.py`, `run_summac.sh`) and
the separate conda environment it needed because it pins transformers 4.35
(`setup_summac_env.sh`, with the environment build log and two job logs in
`logs/`). SummaC failed at runtime with the current tokenizers
(`AlbertTokenizer has no attribute batch_encode_plus`) and was dropped in
favour of LongDocFACTScore.

## pilot_300_papers/

A 300-paper pilot dataset in the same format as `data/` (`dataset.jsonl`
without the `extraction_quality` and `sections_extracted` fields), generated
in March 2026. No overlap with the final 1,500 papers.

## logs/

The first run of the per-summary extraction job (13496021), superseded by
job 13496819 whose log is in `results/per_summary/logs/`.
