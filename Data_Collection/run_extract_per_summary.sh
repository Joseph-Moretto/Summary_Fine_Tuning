#!/bin/bash
#SBATCH -A nsdpi_ext_paid
#SBATCH --job-name="extract_persumm"
#SBATCH --error="/scratch/jam5cq/Summary_Fine_Tuning/Logs/extract_persumm_%j.out"
#SBATCH --output="/scratch/jam5cq/Summary_Fine_Tuning/Logs/extract_persumm_%j.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1

module purge
module load miniforge
conda activate summ_eval

# Isolate from ~/.local user-site packages
export PYTHONNOUSERSITE=1

export HF_HOME=/scratch/jam5cq/HF_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Job info ----
echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Python: $(which python)"
nvidia-smi
echo ""

echo "========================================"
echo "Package Versions"
echo "========================================"
python -c "
import torch; print(f'torch          {torch.__version__}  CUDA={torch.cuda.is_available()}')
import numpy; print(f'numpy          {numpy.__version__}')
"
python -c "from longdocfactscore.ldfacts import LongDocFACTScore; print('longdocfactscore OK')" \
    || { echo 'longdocfactscore NOT AVAILABLE — aborting.'; exit 1; }
echo "========================================"
echo ""

SCRIPT=/scratch/jam5cq/Summary_Fine_Tuning/Data_Collection/extract_per_summary.py
OUTPUT_DIR=/scratch/jam5cq/Summary_Fine_Tuning/Data_Collection
TEST_IDS=/scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/test_paper_ids.json
DATASET=/scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl
SUMMARIES=/scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl

GEN_70B=/scratch/jam5cq/Summary_Fine_Tuning/Results/Output_70b_qlora/eval_generations.json
GEN_8B=/scratch/jam5cq/Summary_Fine_Tuning/Results/Output_8b_bf16/eval_generations.json

mkdir -p "$OUTPUT_DIR"

echo "Extraction started at $(date)"

# =====================================================================
# 70B (keys in the generations JSON are already correctly labeled)
# =====================================================================
echo ""
echo "========================================"
echo "Per-summary extraction: 70B"
echo "========================================"

python "$SCRIPT" \
    --generations "$GEN_70B" \
    --scale 70B \
    --dataset "$DATASET" \
    --summaries "$SUMMARIES" \
    --test_ids "$TEST_IDS" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

EXIT_70B=$?
echo "70B extraction exit code: $EXIT_70B"

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

# =====================================================================
# 8B
# Note: the 8B eval_generations.json was written with --model_tag 70B,
# so its keys are 'finetuned_70B' and 'base_70B'. Rename them on the fly.
# =====================================================================
echo ""
echo "========================================"
echo "Per-summary extraction: 8B (with key renames)"
echo "========================================"

python "$SCRIPT" \
    --generations "$GEN_8B" \
    --scale 8B \
    --dataset "$DATASET" \
    --summaries "$SUMMARIES" \
    --test_ids "$TEST_IDS" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda \
    --rename finetuned_70B=finetuned_8B \
    --rename base_70B=base_8B

EXIT_8B=$?
echo "8B extraction exit code: $EXIT_8B"

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

# =====================================================================
# Combine into single CSV. Include teacher rows once (from the 70B file).
# =====================================================================
echo ""
echo "========================================"
echo "Combining per-scale CSVs"
echo "========================================"

CSV_70B="$OUTPUT_DIR/per_summary_70B.csv"
CSV_8B="$OUTPUT_DIR/per_summary_8B.csv"
CSV_OUT="$OUTPUT_DIR/per_summary_combined.csv"

if [[ -f "$CSV_70B" && -f "$CSV_8B" ]]; then
    head -n 1 "$CSV_70B" > "$CSV_OUT"
    tail -n +2 "$CSV_70B" >> "$CSV_OUT"
    tail -n +2 "$CSV_8B" | awk -F',' '$1 != "teacher"' >> "$CSV_OUT"

    N_ROWS=$(($(wc -l < "$CSV_OUT") - 1))
    echo "Combined CSV: $CSV_OUT ($N_ROWS data rows)"

    echo ""
    echo "Per-system row counts in combined CSV:"
    tail -n +2 "$CSV_OUT" | awk -F',' '{print $1}' | sort | uniq -c
elif [[ -f "$CSV_70B" ]]; then
    echo "Only 70B CSV present. Skipping combine."
elif [[ -f "$CSV_8B" ]]; then
    echo "Only 8B CSV present. Skipping combine."
else
    echo "No CSVs produced. Combine skipped."
fi

echo ""
echo "========================================"
echo "Extraction completed at $(date)"
echo "========================================"
nvidia-smi

if [[ $EXIT_70B -ne 0 ]]; then exit $EXIT_70B; fi
if [[ $EXIT_8B -ne 0 ]]; then exit $EXIT_8B; fi
exit 0