# data/

1,500 arXiv papers (identifiers 2604.12008 to 2604.21931, April 2026
submissions) from computer science, physics and mathematics, each with its
extracted body text, its abstract, and a summary written by the teacher
model. Both files are JSON Lines with one record per paper and the same set
of `paper_id` values.

| File | Records | Size | Role |
|---|---|---|---|
| `dataset.jsonl` | 1,500 | 34 MB | model input (`input_text`) and human reference (`reference_summary`) |
| `summaries.jsonl` | 1,500 | 5 MB | training target and second reference (`generated_summary`) |

## dataset.jsonl

| Field | Description |
|---|---|
| `paper_id` | arXiv identifier with version, e.g. `2604.21931v1` |
| `title` | paper title |
| `input_text` | body text assembled from the extracted sections, each introduced by a heading line (`Introduction:`, `Methods:`, `Results:`, `Conclusion:`), separated by blank lines. This is what the models see. |
| `reference_summary` | the authors' abstract; the human reference at evaluation time |
| `categories` | arXiv categories, primary first |
| `extraction_quality` | `full` if all four sections were found, otherwise `partial` |
| `sections_extracted` | the sections present, in order |

Example (text fields shortened):

```json
{"paper_id": "2604.21931v1",
 "title": "Seeing Fast and Slow: Learning the Flow of Time in Videos",
 "input_text": "Introduction:\nHumans exhibit a strong intuition for the flow of time. ...",
 "reference_summary": "How can we tell whether a video has been sped up or slowed down? ...",
 "categories": ["cs.CV", "cs.AI", "cs.GR"],
 "extraction_quality": "full",
 "sections_extracted": ["introduction", "methods", "results", "conclusion"]}
```

## summaries.jsonl

| Field | Description |
|---|---|
| `paper_id`, `title` | as above |
| `generated_summary` | teacher summary from Claude Opus 4.6. Nine records begin with a `## Summary` header line, which the scripts strip. |
| `original_abstract` | identical to `reference_summary` in `dataset.jsonl` |
| `word_count` | word count of `generated_summary` |
| `success`, `error_message` | generation status; all 1,500 succeeded and every `error_message` is empty |
| `model` | `claude-opus-4-6` for every record |
| `timestamp` | generation time, all on 2026-04-24 |

Example:

```json
{"paper_id": "2604.21931v1",
 "title": "Seeing Fast and Slow: Learning the Flow of Time in Videos",
 "generated_summary": "This paper addresses the fundamental limitation of current video models in reasoning about temporal speed by ...",
 "original_abstract": "How can we tell whether a video has been sped up or slowed down? ...",
 "word_count": 212, "success": true, "error_message": "",
 "model": "claude-opus-4-6", "timestamp": "2026-04-24T23:36:43.327694"}
```

## Statistics

| | mean | median | min | max |
|---|---|---|---|---|
| `input_text`, words | 3,095 | 2,629 | 236 | 14,682 |
| abstract, words | 168 | 172 | 17 | 292 |
| teacher summary, words | 222 | 222 | 190 | 258 |

Section coverage: 491 papers have all four sections (`extraction_quality:
full`); the most common partial combinations are introduction + conclusion
(315), introduction + results + conclusion (242), introduction + methods +
conclusion (136), introduction + methods (129) and introduction + results
(92). 1,463 papers start with the introduction.

Papers carry one (683), two (502), three (229), four (62) or five (23)
categories. Primary categories: `cs.*` 535, `math.*` 439, `physics.*` 323,
`cond-mat.*` 52, `quant-ph` 51, `stat.*` 24, `eess.*` 24, `astro-ph.*` 22,
and a few `q-bio`/`q-fin`, `hep-*`, `nlin.*`, `nucl-*`.

At 8,192 tokens of context, the median paper fits without truncation and
roughly one in nine does not (163 of 1,500 in the 70B training run).

## Domains and split

The training scripts assign each paper to one of three domains by scanning
its categories in order and taking the first with a known prefix: `cs` gives
cs; `math` and `stat` give math; `physics`, `quant-ph`, `astro-ph`,
`cond-mat`, `gr-qc`, `hep`, `nucl` and `nlin` give physics. Papers whose
primary category is `eess` or `q-bio` fall into a domain through a secondary
category. This yields 550 cs, 476 physics and 474 math papers.

The 80/10/10 split is stratified by domain with seed 42:

| Split | Total | cs | physics | math |
|---|---|---|---|---|
| train | 1,200 | 440 | 380 | 380 |
| validation | 150 | 55 | 48 | 47 |
| test | 150 | 55 | 48 | 47 |

The test papers are listed with their domains in
`results/*/train/test_paper_ids.json` (identical for both reported runs), and
that file is what the evaluation scripts read; they never re-split.

## How the scripts use these files

- `training/finetune_*.py` joins the two files on `paper_id`, keeps records
  with `success: true`, uses `title` and `input_text` for the prompt and
  `generated_summary` as the target, and derives the domain from
  `categories`. `reference_summary` is carried along but not trained on.
- `evaluation/evaluate.py` uses `input_text` as the model input and as the
  source for LongDocFACTScore and the abstractiveness measures,
  `reference_summary` as the abstract reference and `generated_summary` as the
  teacher reference.

## Provenance

The scripts that fetched the papers, extracted the sections and generated the
teacher summaries are not part of this repository. Paper text and abstracts
originate from arXiv; every record's `paper_id` is its arXiv identifier.
