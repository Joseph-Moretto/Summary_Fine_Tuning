# evaluation/

Scoring of the fine-tuned models against their base models and the teacher,
plus the per-summary analysis and an interactive inspection tool.

| File | Purpose |
|---|---|
| `evaluate.py` | the full pipeline: inference on the test papers, reference metrics, significance tests, LongDocFACTScore, abstractiveness. Produced every number in `results/*/eval/`. |
| `evaluate_train_aligned.py` | same pipeline with the training prompt and the training-time truncation; written after the reported runs, not used for them |
| `extract_per_summary.py` | per-summary extractive density and LongDocFACTScore as CSV, for the density-vs-factuality analysis |
| `interactive_inference.py` | REPL for generating with an adapter on or off and inspecting prompts and tokens |
| `run_eval_8b.slurm`, `run_eval_70b.slurm` | SLURM jobs for `evaluate.py` on the two reported adapters |
| `run_extract_per_summary.slurm` | SLURM job that runs `extract_per_summary.py` for both scales and combines the CSVs |

All scripts read the data files from `data/` by default and resolve paths
relative to the repository root, so they run from any working directory.

## evaluate.py

### Systems and labels

Every run scores up to three systems, keyed in the outputs as
`finetuned_<tag>`, `base_<tag>` (with `--model_tag`, e.g. `8B`) and
`teacher`. The teacher summaries are read from `data/summaries.jsonl` and
scored like any generated output, except that they are not compared against
themselves. Downstream analyses find the fine-tuned and base systems by
looking for `finetuned` and `base` in the keys.

### Test set

`build_test_data` takes the paper ids and domains from `--test_ids` (the
`test_paper_ids.json` written by training; identical for both runs), joins
them with `data/dataset.jsonl` for the title, text and abstract and with
`data/summaries.jsonl` for the teacher summary, and drops any paper lacking
a successful teacher summary (none in the released data). Papers are
processed in sorted `paper_id` order.

### Inference

The base model is loaded in bf16, or in 4-bit NF4 with bf16 compute when
`--use_qlora` is given (used for 70B). For the fine-tuned system the adapter
is applied with `PeftModel.from_pretrained` and the tokenizer is taken from
the adapter directory; the base system uses the base model's tokenizer.

Both systems see the same prompt:

```
system: You are an expert scientific summarizer. You produce concise, accurate summaries
        of scientific papers that capture the key contributions, methodology, results,
        and significance.
user:   Summarize the following scientific paper in approximately 150-250 words. Focus on:
        1. The main contribution or finding
        2. The methodology or approach used
        3. Key results
        4. Significance or implications

        Write in clear, technical language appropriate for a scientific audience.

        Title: <title>

        Paper Content:
        <input_text, possibly truncated>
```

The paper text is cut so the whole prompt fits in 8,192 tokens: the prompt
is rendered once with an empty paper body to measure the overhead, the paper
keeps `8192 - overhead - 10` tokens from the front, and `[Truncated]` is
appended when a cut was made. Generation uses `max_new_tokens=512`,
`do_sample=True`, `temperature=0.7`, `top_p=0.9`, `repetition_penalty=1.1`;
the seed is reset to 42 before each system's loop. A generation that raises
is stored as an empty string and excluded from all scores.

The prompt wording differs from the one used at training time (see
`training/README.md`), and the truncation keeps only the head of the paper
rather than head and tail. `evaluate_train_aligned.py` removes both
differences; it was written after the reported runs and is not what the
numbers in the paper come from.

### Part 1: reference metrics

Computed for each system against two references, the abstract (`vs_abstract`)
and the teacher summary (`vs_teacher`), for all test papers (`overall`) and
per domain (`cs`, `physics`, `math`).

| Metric | Implementation |
|---|---|
| `rouge1`, `rouge2`, `rougeL` | `rouge_score.RougeScorer` with Porter stemming; F-measure averaged over papers |
| `bertscore_precision`, `_recall`, `_f1` | `bert_score.score` with `microsoft/deberta-xlarge-mnli`, `lang="en"`, batch size 16, no baseline rescaling |
| `bleu_corpus` | NLTK `corpus_bleu` on whitespace tokens, smoothing method 1 |
| `bleu_sentence_avg` | NLTK `sentence_bleu` per paper, same tokenization and smoothing, averaged |
| `meteor` | NLTK `meteor_score` on `word_tokenize` tokens, averaged |

`bert_score`'s `sent_encode` is patched at import to truncate inputs to 512
tokens (`_patch_bertscore_overflow`); without it an over-length abstract
raises an indexing error inside DeBERTa.

Word-count statistics (mean, median, min, max) of each system's summaries are
recorded under `summary_stats`.

### Part 2: significance tests

For the fine-tuned and base system, per-paper scores on the papers both
produced a non-empty summary for (150 in both runs) are compared as paired
samples, for ROUGE-1/2/L, BERTScore F1, sentence BLEU and METEOR, against
both references, overall and per domain. Each row reports:

- the two means and their difference;
- `p(boot)`: a two-sided paired sign-flip permutation test. The sign of each
  paired difference is flipped with probability 1/2 in each of 10,000
  iterations (seed 42) and the p-value is the fraction of iterations whose
  absolute mean difference reaches the observed one, with a +1 correction;
- `p(wilcox)`: two-sided Wilcoxon signed-rank test from SciPy;
- Cohen's d for paired samples (mean difference over the standard deviation
  of the differences);
- significance stars from the permutation p-value: `*` < 0.05, `**` < 0.01,
  `***` < 0.001.

A final block compares summary lengths in words (means, standard deviations,
how many fall in the requested 150-250 range, Wilcoxon p).

### Part 3: LongDocFACTScore

`LongDocFACTScore.score_src_hyp_long(sources, summaries)` from the
`longdocfactscore` package scores each summary against the full paper text
(not the truncated prompt). The metric splits the summary into sentences,
retrieves the most similar source passages with sentence embeddings, and
applies BARTScore to each pair, so values are negative log-likelihoods and
higher is better. Scores are averaged overall and per domain; if batch
scoring fails, papers are scored one at a time and failures dropped as NaN.
The scorer is instantiated once per system. `--skip_ldfact` disables this
part, and it is skipped with a warning if the package is not installed.

### Part 4: abstractiveness

Summary and source are lower-cased and tokenized on `\b\w+\b`. For each
summary:

- `novel_{1,2,3,4}gram`: fraction of the summary's n-grams absent from the
  source;
- extractive fragments (Grusky et al., 2018): greedy longest matches between
  summary and source; `coverage` is the fraction of summary tokens inside a
  fragment and `density` is the sum of squared fragment lengths over the
  summary length, so longer copied spans weigh more;
- `type_token_ratio`: distinct tokens over tokens;
- `summary_length` in tokens.

Means (and standard deviations overall) are reported per system, and a
Wilcoxon signed-rank test compares the fine-tuned and base system on every
measure.

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--output_dir` | required | where all outputs are written |
| `--dataset`, `--summaries` | `data/dataset.jsonl`, `data/summaries.jsonl` | data files |
| `--test_ids` | `<output_dir>/test_paper_ids.json` | test papers and domains from training |
| `--base_model` | `meta-llama/Llama-3.1-8B-Instruct` | Hub id or path of the base model |
| `--adapter_path` | none | LoRA adapter to evaluate; without it only the base model runs |
| `--model_tag` | `8B` | suffix for the system labels |
| `--use_qlora` | off | load the base model in 4-bit NF4 |
| `--skip_base` | off | do not run the base model |
| `--metrics_only` | off | skip inference and score cached generations |
| `--generations_file` | `<output_dir>/eval_generations.json` | generations to score with `--metrics_only` |
| `--skip_significance`, `--skip_ldfact`, `--skip_abstractiveness` | off | disable parts 2-4 |

### Outputs

| File | Contents |
|---|---|
| `eval_generations.json` | `{system: {paper_id: summary}}` for the generated systems (not the teacher) |
| `eval_results.json` | `{system: {"summary_stats": {...}, "vs_abstract": {scope: {metric: value, "n": count}}, "vs_teacher": {...}}}` |
| `eval_report.txt` | the same as a table, plus the word-count block |
| `significance_results.txt` | part 2 tables |
| `ldfact_results.json` | `{system: {scope: {"mean", "std", "n"}}}` |
| `abstractiveness_results.json` | `{system: {"overall": {metric: {"mean", "std"}}, domain: {metric: {"mean", "n"}}}}` |

### Runtime

On one A6000, 8B inference took 50 minutes for the fine-tuned model and 31
for the base model. On two A6000s with the 70B model in 4-bit, 3 h 15 min and
1 h 40 min. Scoring is fast by comparison: re-scoring cached 8B generations
with everything except LongDocFACTScore took 7 minutes (BERTScore dominates),
and LongDocFACTScore took about a minute per system for 150 papers.

## evaluate_train_aligned.py

Identical to `evaluate.py` except that `SYSTEM_MESSAGE` and `INSTRUCTION`
are the training prompt and `_truncate_input_text` calls
`smart_truncate_paper`, copied from `training/finetune_v2.py` (head 70%,
tail 30%, section-aware cut points, `[...TRUNCATED...]` marker). Options and
outputs are the same. Keep the copied function in sync with the training
script if either changes.

## extract_per_summary.py

Writes one CSV row per (system, paper) with `system, scale, paper_id, domain,
extractive_density, summary_length_words, longdocfactscore`. Density is
computed the same way as in `evaluate.py`; LongDocFACTScore is scored per summary
with the same batch-then-fallback strategy. Run once per scale with
`--generations` pointing at that run's `eval_generations.json` and `--scale
8B` or `70B`; the teacher is included in every run's output. `--rename
OLD=NEW` relabels a system key before processing, for generation files that
were written with a different `--model_tag`. `run_extract_per_summary.slurm`
runs both scales and stacks the CSVs into `per_summary_combined.csv` with
the teacher rows taken from the 70B file only.

## interactive_inference.py

Loads a base model (`--use_qlora` for 4-bit) and optionally an adapter, and
with `--dataset`, `--summaries` and `--test_ids` also the test papers. The
prompt is the training prompt, and generation parameters match the training
script's inference helper (adjustable with `set`). Commands:

| Command | Effect |
|---|---|
| `list` | list test papers (first 30) |
| `paper <id>` | select a paper; partial ids are matched |
| `generate` / `gen <id>` | summarize the selected paper (or select and summarize) |
| `compare` | generate with the adapter on and off and show both with the abstract and teacher summary |
| `prompt` | print the rendered prompt, with the base tokenizer's rendering if it differs |
| `tokens` | print the first and last 30 prompt tokens with ids, and flag a duplicated BOS |
| `template`, `template_diff` | show the chat template; compare adapter and base tokenizers and their special tokens |
| `adapter_on`, `adapter_off`, `adapter_status` | toggle the LoRA layers |
| `set <param> <value>`, `params` | temperature, top_p, max_tokens, repetition_penalty |
| `info` | metadata of the selected paper |

The adapter directories under `results/` hold configuration only; point
`--adapter_path` at a directory that contains `adapter_model.safetensors`.
