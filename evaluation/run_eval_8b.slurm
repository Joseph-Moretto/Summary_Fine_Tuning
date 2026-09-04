#!/bin/bash
#SBATCH --job-name=eval_8b_summ
#SBATCH --account=nsdpi_ext_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval_8b_%j.out
#SBATCH --error=logs/eval_8b_%j.out
#
# Evaluate the Llama 3.1 8B adapter: inference for the fine-tuned and base
# model on the held-out test papers, then reference metrics, significance
# tests, LongDocFACTScore and the abstractiveness analysis. Inference took
# about 80 minutes for the two models on one A6000.
#
# Submit from the repository root:
#     sbatch evaluation/run_eval_8b.slurm

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-/scratch/$USER/HF_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep ~/.local user-site packages out of the environment.
export PYTHONNOUSERSITE=1

module purge
module load miniforge
conda activate summ_eval

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Python:  $(which python)"
echo "Project: $PROJECT_DIR"
nvidia-smi
echo ""

echo "========================================"
echo "Package Versions"
echo "========================================"
python -c "
import torch; print(f'torch          {torch.__version__}  CUDA={torch.cuda.is_available()}')
import transformers; print(f'transformers   {transformers.__version__}')
import peft; print(f'peft           {peft.__version__}')
import bitsandbytes; print(f'bitsandbytes   {bitsandbytes.__version__}')
"
python -c "from rouge_score import rouge_scorer; print('rouge-score    OK')"
python -c "from bert_score import score; print('bert-score     OK')"
python -c "from nltk.translate.bleu_score import sentence_bleu; print('BLEU           OK')"
python -c "from nltk.translate.meteor_score import meteor_score; print('METEOR         OK')"
python -c "from longdocfactscore.ldfacts import LongDocFACTScore; print('LongDocFACTScore OK')" 2>/dev/null \
    || echo "LongDocFACTScore NOT AVAILABLE (will be skipped)"
echo "========================================"
echo ""

RUN_DIR="$PROJECT_DIR/results/llama3.1-8b-lora"
OUTPUT_DIR="$PROJECT_DIR/runs/eval-llama3.1-8b-lora"

echo "Evaluation started at $(date)"

python "$PROJECT_DIR/evaluation/evaluate.py" \
    --dataset "$PROJECT_DIR/data/dataset.jsonl" \
    --summaries "$PROJECT_DIR/data/summaries.jsonl" \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path "$RUN_DIR/train/adapter" \
    --test_ids "$RUN_DIR/train/test_paper_ids.json" \
    --output_dir "$OUTPUT_DIR" \
    --model_tag 8B
EXIT_CODE=$?

# To re-score existing generations without running inference (this is how the
# reference metrics in results/llama3.1-8b-lora/eval were last produced), use:
#
# python "$PROJECT_DIR/evaluation/evaluate.py" \
#     --metrics_only \
#     --output_dir "$OUTPUT_DIR" \
#     --test_ids "$RUN_DIR/train/test_paper_ids.json" \
#     --model_tag 8B \
#     --skip_ldfact

echo ""
echo "========================================"
echo "Evaluation completed at $(date), exit code $EXIT_CODE"
echo "========================================"
nvidia-smi

exit $EXIT_CODE
