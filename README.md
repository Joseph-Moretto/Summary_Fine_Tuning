# Summary_Fine_Tuning

Step 1 — Extraction You get 1500 papers. Each one has input_text (the cleaned body) and reference_summary (the abstract). No summaries are generated yet.
Step 2 — Teacher summary generation. You feed all 1500 input_text fields to GPT-4 or Claude and get back 1500 teacher-generated summaries. Now each paper has three things: the body text, the abstract, and a teacher summary.
Step 3 — Split. You split into 80/10/10 (train/val/test). So roughly 1200 training pairs, 150 validation,1500 test. The training pairs are (input_text → teacher summary) formatted as instruction-response examples.
Step 4 — QLoRA training. You fine-tune LLaMA 70B and 8B on the 1200 training pairs, where the model learns to produce summaries that look like the teacher's.
Step 5 — Inference. Your fine-tuned models generate summaries for the 150 test papers.
-ensure test set it kept track of and has 
Step 6 — Evaluation. You compare those student summaries against two references: the abstracts (human baseline) and the held-out teacher summaries (distillation target). ROUGE, BERTScore, etc. give you separate scores against each reference, which answer different questions — comparison against teacher summaries tells you how well distillation worked, comparison against abstracts tells you how the summaries relate to what humans actually wrote.
Also Run the base (non-fine-tuned) LLaMA models on the same test set so you can show the improvement from training.