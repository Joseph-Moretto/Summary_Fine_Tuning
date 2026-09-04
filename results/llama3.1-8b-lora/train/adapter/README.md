---
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: peft
license: llama3.1
language:
- en
pipeline_tag: summarization
tags:
- lora
- summarization
- scientific-papers
- distillation
---

# Llama 3.1 8B Instruct, LoRA adapter for scientific paper summarization

Built with Llama. This is a LoRA adapter for `meta-llama/Llama-3.1-8B-Instruct`
that summarizes a scientific paper in 150-250 words of prose. It was trained
by distillation: the targets are summaries written by Claude Opus 4.6 for
1,200 arXiv papers in computer science, physics and mathematics.

## Training

- Base model in bf16, no quantization
- LoRA rank 16, alpha 32, dropout 0.05 on q, k, v, o, gate, up and down projections (41.9 M trainable parameters, 0.52%)
- 3 epochs over 1,200 papers, effective batch size 16, learning rate 2e-4 with cosine decay and 3% warmup, weight decay 0.01, gradient clipping 0.3
- Context 8,192 tokens; paper text cut at 12,000 characters
- Loss on the summary tokens only
- Final weights after epoch 3 (validation loss 0.529; 0.491 after epoch 1)
- One RTX A6000 48 GB GPU, 3 h 45 min

Training script, data and job script: `training/finetune_v1.py`, `data/`,
`training/run_finetune_8b.slurm` in this repository.

## Prompt format

Chat format with the training-time system message and instruction from
`training/finetune_v1.py`; the user turn carries the paper title and text
after `Paper Content:`. Generation was run with temperature 0.7, top-p 0.9,
repetition penalty 1.1 and up to 512 new tokens.

## Evaluation

On 150 held-out papers, against the authors' abstracts: ROUGE-L 0.237,
BERTScore F1 0.644 (base model: 0.222, 0.605). Against the teacher summaries:
ROUGE-L 0.390, BERTScore F1 0.742 (base model: 0.300, 0.668). Full tables in
`results/README.md`.

## Intended use and limitations

Research on summarization and distillation. English scientific papers from
cs, physics and math only; the adapter reproduces the teacher's style and can
reproduce the teacher's mistakes. Only the first 12,000 characters of a paper
are seen. Not evaluated for factual reliability beyond the automatic metrics
reported.

## License

Derivative of Llama 3.1 and subject to the Llama 3.1 Community License
Agreement. The weight file is not stored in this repository (168 MB, over
GitHub's file limit); this directory holds the adapter configuration and
tokenizer files.
