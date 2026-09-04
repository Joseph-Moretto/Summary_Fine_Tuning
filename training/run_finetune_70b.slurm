#!/bin/bash
#SBATCH --job-name=ft_70b_summ
#SBATCH --account=nsdpi_ext_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=30:00:00
#SBATCH --output=logs/finetune_70b_%j.out
#SBATCH --error=logs/finetune_70b_%j.out
#
# QLoRA fine-tuning of Llama 3.3 70B Instruct. This is the configuration of
# the reported 70B run (job 12501039: two RTX A6000 48 GB, 24 h 08 min).
#
# Submit from the repository root so that logs/ and the relative paths resolve:
#     sbatch training/run_finetune_70b.slurm
# Account, partition and GPU type are specific to the UVA cluster.

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-/scratch/$USER/HF_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module purge
module load miniforge
conda activate qlora_summ_ft3

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPUs:    $(nvidia-smi -L | wc -l)"
echo "Project: $PROJECT_DIR"
echo "HF_HOME: $HF_HOME"
echo ""
pip show trl
nvidia-smi

echo "========================================"
echo "Environment Configuration"
echo "========================================"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU memory: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")' 2>/dev/null)"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "PEFT version: $(python -c 'import peft; print(peft.__version__)')"
echo "bitsandbytes version: $(python -c 'import bitsandbytes; print(bitsandbytes.__version__)')"
echo "========================================"
echo ""

echo "Training started at $(date)"

python "$PROJECT_DIR/training/finetune_v2.py" \
    --dataset "$PROJECT_DIR/data/dataset.jsonl" \
    --summaries "$PROJECT_DIR/data/summaries.jsonl" \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --output "$PROJECT_DIR/runs/llama3.3-70b-qlora" \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --max_seq_length 8192 \
    --max_memory_per_gpu 45GiB \
    --warmup_ratio 0.05
EXIT_CODE=$?

echo "Training completed at $(date)"
echo "Exit code: $EXIT_CODE"
nvidia-smi

exit $EXIT_CODE
