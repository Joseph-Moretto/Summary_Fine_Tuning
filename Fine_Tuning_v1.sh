#!/bin/bash
#SBATCH -A nsdpi_ext_paid
#SBATCH --job-name="qlora_70b_summ"
#SBATCH --error="/scratch/jam5cq/Summary_Fine_Tuning/Logs/qlora_summ_ft_%j.out"
#SBATCH --output="/scratch/jam5cq/Summary_Fine_Tuning/Logs/qlora_summ_ft_%j.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=160G
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=16

# Load environment
module purge
module load miniforge
conda activate qlora_summ_ft3

# Display GPU information
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"
echo ""
pip show trl
nvidia-smi

# Set cache directory
export HF_HOME=/scratch/jam5cq/HF_cache
echo "HF_HOME: $HF_HOME"

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Print environment info
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

# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Llama-3.3-70B-Instruct

python /scratch/jam5cq/Summary_Fine_Tuning/Fine_Tuning_v2.py \
    --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
    --summaries /scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --output /scratch/jam5cq/Summary_Fine_Tuning/Output_v2 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --max_seq_length 8192 \
    --max_memory_per_gpu 45GiB \
    --warmup_ratio 0.05

# python /scratch/jam5cq/Summary_Fine_Tuning/Fine_Tuning_v1.py \
#     --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
#     --summaries /scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl \
#     --model  meta-llama/Llama-3.3-70B-Instruct \
#     --output /scratch/jam5cq/Summary_Fine_Tuning/Output \
#     --per_device_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --lora_rank 16 \
#     --lora_alpha 32 \
#     --max_seq_length 8196 \

# python /scratch/jam5cq/Summary_Fine_Tuning/Fine_Tuning_v1.py \
#     --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
#     --summaries /scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --output /scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16 \
#     --no_quantize \
#     --per_device_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lora_rank 16 \
#     --lora_alpha 32 \
#     --max_seq_length 8192 \


EXIT_CODE=$?

echo "Training completed at $(date)"
echo "Exit code: $EXIT_CODE"

# Print final GPU memory stats
nvidia-smi

exit $EXIT_CODE