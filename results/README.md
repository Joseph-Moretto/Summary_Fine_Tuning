# results/

Everything produced by the scripts in `training/` and `evaluation/` for the
two runs reported in the paper, plus the per-summary CSVs. All jobs ran on
the UVA cluster on RTX A6000 48 GB GPUs. Each run directory has the same
shape:

```
<run>/
  train/      output of the training script: adapter configuration, split
              metadata, trainer metrics (weights excluded, see below)
  eval/       output of evaluation/evaluate.py for that adapter
  logs/       SLURM job output of the training and evaluation jobs
```

## Run provenance

### llama3.1-8b-lora

Llama 3.1 8B Instruct with a bf16 LoRA adapter, trained by
`training/finetune_v1.py` (job 12228610, 2026-04-26, one GPU, 3 h 45 min).
Train loss 0.369; validation loss 0.491 / 0.494 / 0.529 after epochs 1-3.
The script keeps the final weights only, so the evaluated adapter is the
epoch-3 state.

Test-set generations were produced by `evaluation/evaluate.py` on 2026-05-03
(job 12458254, about 1.5 h on one GPU; its log was not kept). LongDocFACTScore
was computed in that job. The reference metrics, significance tests and
abstractiveness analysis in `eval/` come from a re-scoring of the same
generations on 2026-05-05 with `--metrics_only --skip_ldfact` (job 12553996,
7 min, log in `logs/eval_metrics_12553996.out`). An earlier evaluation of this
adapter from 2026-04-30, with different generations and a SummaC attempt, is
in `archive/llama3.1-8b-lora/`.

### llama3.3-70b-qlora

Llama 3.3 70B Instruct with a 4-bit NF4 QLoRA adapter, trained by
`training/finetune_v2.py` (job 12501039, 2026-05-04 to 05-05, two GPUs,
24 h 08 min). Train loss 0.570; validation loss 0.652 / 0.634 / 0.667 after
epochs 1-3. Checkpoints were saved per epoch and the best one by validation
loss (epoch 2) was restored and saved, so `train/eval_results.json` reports
0.634. `train/training_config.json` records the full configuration. An
earlier 70B run with `finetune_v1.py` is in `archive/llama3.3-70b-qlora-v1/`.

Evaluation was one full run of `evaluation/evaluate.py --use_qlora` on
2026-05-06 (job 12570403, 5 h 05 min on two GPUs, log in
`logs/eval_12570403.out`): 3 h 15 min of inference for the fine-tuned model,
1 h 40 min for the base model, then all four analyses.

Both runs used the same data files and seed and produced byte-identical
`test_paper_ids.json`: 150 papers, 55 cs / 48 physics / 47 math.

## Files in `train/`

| File | Written by | Contents |
|---|---|---|
| `adapter/adapter_config.json` | PEFT | LoRA rank, alpha, dropout, target modules, base model id, PEFT version |
| `adapter/tokenizer_config.json`, `adapter/chat_template.jinja` | tokenizer save | tokenizer settings and the Llama 3 chat template as saved with the adapter; `tokenizer.json` itself is identical to the base model's and is only kept in `archive/adapters/` |
| `adapter/README.md` | PEFT | auto-generated model card stub |
| `test_paper_ids.json` | training script | `[{"paper_id", "domain"}, ...]` for the 150 test papers |
| `split_summary.json` | training script | per-domain counts of train / validation / test |
| `train_results.json` | Trainer | mean train loss, runtime, throughput, FLOPs |
| `eval_results.json` | Trainer | validation loss and runtime of the final evaluation |
| `all_results.json` | Trainer | the two files above merged |
| `trainer_state.json` | Trainer | `log_history` with loss, gradient norm, learning rate, entropy, mean token accuracy and token count every 10 steps, plus validation loss per epoch; `best_metric`; 225 steps in total |
| `training_config.json` (70B only) | `finetune_v2.py` | every hyperparameter of the run and the split sizes |
| `trl_model_card.md` (70B only) | TRL | auto-generated card; records TRL 1.1.0, Transformers 5.6.2, PyTorch 2.11.0+cu130, Datasets 4.8.4, Tokenizers 0.22.2 |

## Files in `eval/`

| File | Contents |
|---|---|
| `eval_generations.json` | `{system: {paper_id: summary}}` for `finetuned_*` and `base_*`; the teacher summaries are read from `data/summaries.jsonl` at scoring time |
| `eval_results.json` | ROUGE-1/2/L, BERTScore P/R/F1, corpus and sentence BLEU, METEOR for every system, against abstracts (`vs_abstract`) and teacher summaries (`vs_teacher`), overall and per domain, plus word-count statistics |
| `eval_report.txt` | the same numbers as a table |
| `significance_results.txt` | fine-tuned vs. base per metric: means, difference, paired permutation p-value, Wilcoxon p-value, Cohen's d, overall and per domain; plus a summary-length comparison |
| `ldfact_results.json` | LongDocFACTScore mean / std / n per system, overall and per domain |
| `abstractiveness_results.json` | novel 1-4-gram rates, extractive coverage and density, type-token ratio and length per system |

`evaluation/README.md` describes how each number is computed.

## Results: Llama 3.1 8B Instruct, bf16 LoRA

**Against the abstracts, all 150 test papers**

| Metric | base_8B | finetuned_8B | teacher |
|---|---|---|---|
| ROUGE-1 | 0.4280 | 0.4663 | 0.4887 |
| ROUGE-2 | 0.1608 | 0.1640 | 0.1861 |
| ROUGE-L | 0.2216 | 0.2374 | 0.2582 |
| BERTScore P | 0.5790 | 0.6277 | 0.6358 |
| BERTScore R | 0.6354 | 0.6619 | 0.6744 |
| BERTScore F1 | 0.6050 | 0.6435 | 0.6536 |
| BLEU (corpus) | 0.0632 | 0.0662 | 0.0794 |
| BLEU (sentence avg.) | 0.0592 | 0.0584 | 0.0725 |
| METEOR | 0.3342 | 0.3359 | 0.3572 |

**Against the teacher summaries, all 150 test papers**

| Metric | base_8B | finetuned_8B |
|---|---|---|
| ROUGE-1 | 0.5409 | 0.6357 |
| ROUGE-2 | 0.2288 | 0.3106 |
| ROUGE-L | 0.3001 | 0.3902 |
| BERTScore P | 0.6608 | 0.7453 |
| BERTScore R | 0.6764 | 0.7392 |
| BERTScore F1 | 0.6684 | 0.7422 |
| BLEU (corpus) | 0.1079 | 0.1820 |
| BLEU (sentence avg.) | 0.1076 | 0.1750 |
| METEOR | 0.3947 | 0.4645 |

**Per domain, ROUGE-L / BERTScore F1**

| Reference | Domain | n | base_8B | finetuned_8B | teacher |
|---|---|---|---|---|---|
| abstract | cs | 55 | 0.2310 / 0.6175 | 0.2475 / 0.6555 | 0.2685 / 0.6646 |
| abstract | physics | 48 | 0.2185 / 0.6119 | 0.2254 / 0.6438 | 0.2449 / 0.6559 |
| abstract | math | 47 | 0.2137 / 0.5834 | 0.2377 / 0.6290 | 0.2598 / 0.6383 |
| teacher | cs | 55 | 0.3100 / 0.6775 | 0.4157 / 0.7519 |  |
| teacher | physics | 48 | 0.2933 / 0.6732 | 0.3697 / 0.7412 |  |
| teacher | math | 47 | 0.2956 / 0.6527 | 0.3813 / 0.7318 |  |

**Summary length in words**

| | base_8B | finetuned_8B | teacher |
|---|---|---|---|
| mean | 288.9 | 212.6 | 221.9 |
| median | 284.0 | 210.5 | 221.0 |
| min | 198 | 167 | 190 |
| max | 414 | 255 | 254 |

**LongDocFACTScore, mean (std)**

| Scope | base_8B | finetuned_8B | teacher |
|---|---|---|---|
| overall | -4.072 (0.705) | -4.327 (0.449) | -4.184 (0.480) |
| cs | -4.069 (0.741) | -4.344 (0.475) | -4.205 (0.533) |
| physics | -3.972 (0.725) | -4.289 (0.388) | -4.128 (0.442) |
| math | -4.176 (0.622) | -4.347 (0.472) | -4.215 (0.445) |

**Abstractiveness, means over the test set**

| Measure | base_8B | finetuned_8B | teacher |
|---|---|---|---|
| Novel unigrams | 15.2% | 19.0% | 12.8% |
| Novel bigrams | 45.1% | 56.8% | 53.0% |
| Novel trigrams | 62.8% | 76.5% | 75.6% |
| Novel 4-grams | 72.9% | 86.4% | 86.1% |
| Extractive coverage | 84.8% | 81.0% | 87.2% |
| Extractive density | 5.16 | 2.62 | 2.74 |
| Type-token ratio | 0.52 | 0.68 | 0.66 |
| Length in tokens | 300.89 | 227.49 | 235.90 |

**Fine-tuned vs. base, all 150 papers** (`significance_results.txt` has the per-domain tables)

*against the abstracts*

| Metric | fine-tuned | base | diff | p (permutation) | p (Wilcoxon) | Cohen's d | |
|---|---|---|---|---|---|---|---|
| rouge1 | 0.4663 | 0.4280 | +0.0383 | 0.0001 | 0.0000 | 0.685 | *** |
| rouge2 | 0.1640 | 0.1608 | +0.0032 | 0.3625 | 0.4672 | 0.074 |  |
| rougeL | 0.2374 | 0.2216 | +0.0158 | 0.0001 | 0.0002 | 0.353 | *** |
| bertscore_f1 | 0.6435 | 0.6050 | +0.0384 | 0.0001 | 0.0000 | 1.398 | *** |
| bleu_sentence | 0.0584 | 0.0592 | -0.0008 | 0.7608 | 0.7533 | -0.025 |  |
| meteor | 0.3359 | 0.3342 | +0.0017 | 0.6501 | 0.8022 | 0.038 |  |

*against the teacher summaries*

| Metric | fine-tuned | base | diff | p (permutation) | p (Wilcoxon) | Cohen's d | |
|---|---|---|---|---|---|---|---|
| rouge1 | 0.6357 | 0.5409 | +0.0948 | 0.0001 | 0.0000 | 1.688 | *** |
| rouge2 | 0.3106 | 0.2288 | +0.0818 | 0.0001 | 0.0000 | 1.397 | *** |
| rougeL | 0.3902 | 0.3001 | +0.0901 | 0.0001 | 0.0000 | 1.452 | *** |
| bertscore_f1 | 0.7422 | 0.6684 | +0.0738 | 0.0001 | 0.0000 | 2.371 | *** |
| bleu_sentence | 0.1750 | 0.1076 | +0.0674 | 0.0001 | 0.0000 | 1.201 | *** |
| meteor | 0.4645 | 0.3947 | +0.0699 | 0.0001 | 0.0000 | 1.200 | *** |

## Results: Llama 3.3 70B Instruct, 4-bit QLoRA

**Against the abstracts, all 150 test papers**

| Metric | base_70B | finetuned_70B | teacher |
|---|---|---|---|
| ROUGE-1 | 0.4594 | 0.4692 | 0.4887 |
| ROUGE-2 | 0.1712 | 0.1671 | 0.1861 |
| ROUGE-L | 0.2356 | 0.2400 | 0.2582 |
| BERTScore P | 0.6102 | 0.6291 | 0.6358 |
| BERTScore R | 0.6378 | 0.6649 | 0.6744 |
| BERTScore F1 | 0.6228 | 0.6455 | 0.6536 |
| BLEU (corpus) | 0.0805 | 0.0687 | 0.0794 |
| BLEU (sentence avg.) | 0.0699 | 0.0612 | 0.0725 |
| METEOR | 0.3195 | 0.3417 | 0.3572 |

**Against the teacher summaries, all 150 test papers**

| Metric | base_70B | finetuned_70B |
|---|---|---|
| ROUGE-1 | 0.5587 | 0.6630 |
| ROUGE-2 | 0.2410 | 0.3465 |
| ROUGE-L | 0.3106 | 0.4315 |
| BERTScore P | 0.6934 | 0.7611 |
| BERTScore R | 0.6723 | 0.7588 |
| BERTScore F1 | 0.6826 | 0.7599 |
| BLEU (corpus) | 0.1235 | 0.2138 |
| BLEU (sentence avg.) | 0.1168 | 0.2067 |
| METEOR | 0.3637 | 0.5003 |

**Per domain, ROUGE-L / BERTScore F1**

| Reference | Domain | n | base_70B | finetuned_70B | teacher |
|---|---|---|---|---|---|
| abstract | cs | 55 | 0.2414 / 0.6311 | 0.2540 / 0.6593 | 0.2685 / 0.6646 |
| abstract | physics | 48 | 0.2367 / 0.6295 | 0.2314 / 0.6482 | 0.2449 / 0.6559 |
| abstract | math | 47 | 0.2278 / 0.6063 | 0.2323 / 0.6268 | 0.2598 / 0.6383 |
| teacher | cs | 55 | 0.3207 / 0.6869 | 0.4593 / 0.7708 |  |
| teacher | physics | 48 | 0.3057 / 0.6883 | 0.4124 / 0.7590 |  |
| teacher | math | 47 | 0.3037 / 0.6717 | 0.4185 / 0.7480 |  |

**Summary length in words**

| | base_70B | finetuned_70B | teacher |
|---|---|---|---|
| mean | 198.4 | 217.4 | 221.9 |
| median | 195.5 | 217.0 | 221.0 |
| min | 151 | 179 | 190 |
| max | 353 | 266 | 254 |

**LongDocFACTScore, mean (std)**

| Scope | base_70B | finetuned_70B | teacher |
|---|---|---|---|
| overall | -3.761 (0.449) | -4.391 (0.432) | -4.184 (0.480) |
| cs | -3.760 (0.451) | -4.414 (0.504) | -4.205 (0.533) |
| physics | -3.664 (0.384) | -4.303 (0.331) | -4.128 (0.442) |
| math | -3.863 (0.486) | -4.455 (0.419) | -4.215 (0.445) |

**Abstractiveness, means over the test set**

| Measure | base_70B | finetuned_70B | teacher |
|---|---|---|---|
| Novel unigrams | 15.3% | 19.6% | 12.8% |
| Novel bigrams | 47.4% | 57.8% | 53.0% |
| Novel trigrams | 66.3% | 78.2% | 75.6% |
| Novel 4-grams | 76.6% | 88.0% | 86.1% |
| Extractive coverage | 84.7% | 80.4% | 87.2% |
| Extractive density | 4.06 | 2.46 | 2.74 |
| Type-token ratio | 0.60 | 0.68 | 0.66 |
| Length in tokens | 208.21 | 232.35 | 235.90 |

**Fine-tuned vs. base, all 150 papers** (`significance_results.txt` has the per-domain tables)

*against the abstracts*

| Metric | fine-tuned | base | diff | p (permutation) | p (Wilcoxon) | Cohen's d | |
|---|---|---|---|---|---|---|---|
| rouge1 | 0.4692 | 0.4594 | +0.0098 | 0.0297 | 0.0325 | 0.183 | * |
| rouge2 | 0.1671 | 0.1712 | -0.0041 | 0.3012 | 0.2639 | -0.085 |  |
| rougeL | 0.2400 | 0.2356 | +0.0044 | 0.2494 | 0.4501 | 0.094 |  |
| bertscore_f1 | 0.6455 | 0.6228 | +0.0228 | 0.0001 | 0.0000 | 0.846 | *** |
| bleu_sentence | 0.0612 | 0.0699 | -0.0087 | 0.0025 | 0.0013 | -0.255 | ** |
| meteor | 0.3417 | 0.3195 | +0.0222 | 0.0001 | 0.0000 | 0.399 | *** |

*against the teacher summaries*

| Metric | fine-tuned | base | diff | p (permutation) | p (Wilcoxon) | Cohen's d | |
|---|---|---|---|---|---|---|---|
| rouge1 | 0.6630 | 0.5587 | +0.1043 | 0.0001 | 0.0000 | 1.953 | *** |
| rouge2 | 0.3465 | 0.2410 | +0.1055 | 0.0001 | 0.0000 | 1.685 | *** |
| rougeL | 0.4315 | 0.3106 | +0.1209 | 0.0001 | 0.0000 | 1.847 | *** |
| bertscore_f1 | 0.7599 | 0.6826 | +0.0773 | 0.0001 | 0.0000 | 2.550 | *** |
| bleu_sentence | 0.2067 | 0.1168 | +0.0899 | 0.0001 | 0.0000 | 1.451 | *** |
| meteor | 0.5003 | 0.3637 | +0.1366 | 0.0001 | 0.0000 | 2.021 | *** |

## per_summary/

Per-summary values behind the density-vs-factuality analysis, from
`evaluation/extract_per_summary.py` (job 13496819, 2026-05-25, 8 min on one
GPU; log in `per_summary/logs/`). One row per (system, paper):

| Column | Meaning |
|---|---|
| `system` | `finetuned_8B`, `base_8B`, `finetuned_70B`, `base_70B` or `teacher` |
| `scale` | `8B` or `70B`, the run the row was computed in |
| `paper_id`, `domain` | test paper and its domain |
| `extractive_density` | Grusky et al. (2018) density of the summary against the full paper text |
| `summary_length_words` | tokens in the summary under the same tokenization |
| `longdocfactscore` | LongDocFACTScore of the summary against the full paper text |

`per_summary_8B.csv` and `per_summary_70B.csv` each hold 450 rows (three
systems); `per_summary_combined.csv` stacks them with the teacher rows taken
from the 70B file only, 750 rows. The system means match `ldfact_results.json`
and `abstractiveness_results.json` in the run directories.

## Adapter weights

`train/adapter/` holds the configuration but not the weights.
`adapter_model.safetensors` is 168 MB for the 8B adapter and 829 MB for the
70B adapter (fp32), both over GitHub's 100 MB file limit, so they are not in
the repository. `archive/adapters/` has complete adapter directories minus
that one file. The adapters can be retrained with the job scripts in
`training/`.
