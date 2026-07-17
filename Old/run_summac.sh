#!/bin/bash
#SBATCH -A nsdpi_ext_paid
#SBATCH --job-name="summac_eval"
#SBATCH --error="/scratch/jam5cq/Summary_Fine_Tuning/Logs/summac_eval_%j.out"
#SBATCH --output="/scratch/jam5cq/Summary_Fine_Tuning/Logs/summac_eval_%j.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4

module purge
module load miniforge
conda activate summac_eval

export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/jam5cq/HF_cache

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Python: $(which python)"
nvidia-smi
echo ""

python -c "
import torch; print(f'torch          {torch.__version__}  CUDA={torch.cuda.is_available()}')
import transformers; print(f'transformers   {transformers.__version__}')
from summac.model_summac import SummaCZS; print('SummaC         OK')
"
echo "========================================"
echo ""

BASE_DIR="/scratch/jam5cq/Summary_Fine_Tuning"

echo "SummaC evaluation started at $(date)"

# =====================================================================
# 8B model SummaC scores
# =====================================================================
python "/scratch/jam5cq/Summary_Fine_Tuning/Evaluation/Eval_Output/summac_eval.py" \
    --generations "${BASE_DIR}/Output_8b_bf16/eval_generations.json" \
    --dataset "${BASE_DIR}/dataset.jsonl" \
    --test_ids "${BASE_DIR}/Output/Llama3.1_8b_lora/test_paper_ids.json" \
    --output "${BASE_DIR}/Output_8b_bf16/summac_results.json"

#/scratch/jam5cq/Summary_Fine_Tuning/Output_8b_bf16/eval_generations.json

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"

# =====================================================================
# 70B model SummaC scores (uncomment when ready)
# =====================================================================
# python "${BASE_DIR}/summac_eval.py" \
#     --generations "${BASE_DIR}/Output/eval_generations.json" \
#     --dataset "${BASE_DIR}/dataset.jsonl" \
#     --test_ids "${BASE_DIR}/Output/test_paper_ids.json" \
#     --output "${BASE_DIR}/Output/summac_results.json"

echo ""
echo "SummaC evaluation completed at $(date)"
nvidia-smi

exit ${EXIT_CODE:-0}
