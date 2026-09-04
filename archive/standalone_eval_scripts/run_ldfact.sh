#!/bin/bash
#SBATCH -A nsdpi_ext_paid
#SBATCH --job-name="ldfact_eval"
#SBATCH --error="/scratch/jam5cq/Summary_Fine_Tuning/Logs/ldfact_eval_%j.out"
#SBATCH --output="/scratch/jam5cq/Summary_Fine_Tuning/Logs/ldfact_eval_%j.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=50G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4

module purge
module load miniforge
conda activate summ_eval

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
from longdocfactscore.ldfacts import LongDocFACTScore; print('LongDocFACTScore OK')
"
echo "========================================"
echo ""

BASE_DIR="/scratch/jam5cq/Summary_Fine_Tuning"
EVAL_SCRIPT="${BASE_DIR}/ldfact_eval.py"
DATASET="${BASE_DIR}/dataset.jsonl"
SUMMARIES="${BASE_DIR}/summaries.jsonl"
TEST_IDS="${BASE_DIR}/Output/Llama3.1_8b_lora/test_paper_ids.json"

echo "LongDocFACTScore evaluation started at $(date)"

# =====================================================================
# 8B model
# =====================================================================
echo ""
echo "========================================"
echo "LongDocFACTScore: 8B model"
echo "========================================"

python "/scratch/jam5cq/Summary_Fine_Tuning/Evaluation/Eval_Output/ldfact_eval.py" \
    --generations "${BASE_DIR}/Output_8b_bf16/eval_generations.json" \
    --dataset "${DATASET}" \
    --summaries "${SUMMARIES}" \
    --test_ids "${TEST_IDS}" \
    --output "${BASE_DIR}/Output_8b_bf16/ldfact_results.json"

EXIT_8B=$?
echo "8B LongDocFACTScore exit code: $EXIT_8B"

# =====================================================================
# 70B model (uncomment when generations are ready)
# =====================================================================
# echo ""
# echo "========================================"
# echo "LongDocFACTScore: 70B model"
# echo "========================================"
#
# python "${EVAL_SCRIPT}" \
#     --generations "${BASE_DIR}/Output/eval_generations.json" \
#     --dataset "${DATASET}" \
#     --summaries "${SUMMARIES}" \
#     --test_ids "${TEST_IDS}" \
#     --output "${BASE_DIR}/Output/ldfact_results.json"
#
# EXIT_70B=$?
# echo "70B LongDocFACTScore exit code: $EXIT_70B"

echo ""
echo "========================================"
echo "LongDocFACTScore evaluation completed at $(date)"
echo "========================================"
nvidia-smi

exit ${EXIT_8B:-0}
