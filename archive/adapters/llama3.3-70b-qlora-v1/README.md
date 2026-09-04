---
base_model: meta-llama/Llama-3.3-70B-Instruct
library_name: peft
license: llama3.3
language:
- en
tags:
- lora
- qlora
- summarization
---

# Llama 3.3 70B Instruct, first QLoRA attempt (superseded)

Built with Llama. First 70B adapter, trained on 2026-04-26 with
`training/finetune_v1.py`: 4-bit NF4 base, LoRA rank 16, alpha 32, dropout
0.05, learning rate 2e-4, 3 epochs, paper text cut at 12,000 characters.
Validation loss 0.949 against 0.634 for the released adapter in
`results/llama3.3-70b-qlora/`, which replaced it. Kept for the record; not
used for any reported number. Subject to the Llama 3.3 Community License
Agreement. The weight file (414 MB, bf16) is not stored in the repository.
