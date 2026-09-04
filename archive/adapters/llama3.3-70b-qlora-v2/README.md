---
base_model: meta-llama/Llama-3.3-70B-Instruct
library_name: peft
license: llama3.3
language:
- en
pipeline_tag: summarization
tags:
- lora
- qlora
- summarization
- scientific-papers
- distillation
---

# Llama 3.3 70B Instruct, QLoRA adapter for scientific paper summarization

Built with Llama. This is a LoRA adapter for `meta-llama/Llama-3.3-70B-Instruct`
that summarizes a scientific paper in 150-250 words of prose. It was trained
by distillation: the targets are summaries written by Claude Opus 4.6 for
1,200 arXiv papers in computer science, physics and mathematics.

## Training

- Base model loaded in 4-bit NF4 with double quantization and bf16 compute (QLoRA)
- LoRA rank 16, alpha 32, dropout 0.1 on q, k, v, o, gate, up and down projections (207.1 M trainable parameters, 0.29%)
- 3 epochs over 1,200 papers, effective batch size 16, learning rate 1e-4 with cosine decay and 5% warmup, weight decay 0.01, gradient clipping 0.3
- Context 8,192 tokens; papers that do not fit are cut to the first 70% and last 30% of the token budget
- Loss on the summary tokens only
- The epoch-2 checkpoint (validation loss 0.634, the lowest of the three epochs) is the released adapter
- Two RTX A6000 48 GB GPUs, 24 h 08 min

Training script, data and job script: `training/finetune_v2.py`, `data/`,
`training/run_finetune_70b.slurm` in this repository.

## Prompt format

Chat format with the training-time system message and instruction from
`training/finetune_v2.py`; the user turn carries the paper title and text
after `Paper Content:`. Generation was run with temperature 0.7, top-p 0.9,
repetition penalty 1.1 and up to 512 new tokens.

## Evaluation

On 150 held-out papers, against the authors' abstracts: ROUGE-L 0.240,
BERTScore F1 0.646 (base model: 0.236, 0.623). Against the teacher summaries:
ROUGE-L 0.432, BERTScore F1 0.760 (base model: 0.311, 0.683). Full tables in
`results/README.md`.

## Intended use and limitations

Research on summarization and distillation. English scientific papers from
cs, physics and math only; the adapter reproduces the teacher's style and can
reproduce the teacher's mistakes. Summaries of long papers are produced from
a truncated view of the paper. Not evaluated for factual reliability beyond
the automatic metrics reported.

## License

Derivative of Llama 3.3 and subject to the Llama 3.3 Community License
Agreement. The weight file is not stored in this repository (829 MB, over
GitHub's file limit); this directory holds the adapter configuration and
tokenizer files.
