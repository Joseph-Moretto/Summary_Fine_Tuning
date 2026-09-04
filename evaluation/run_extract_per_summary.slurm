#!/bin/bash
#SBATCH --job-name=extract_persumm
#SBATCH --account=nsdpi_ext_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract_persumm_%j.out
#SBATCH --error=logs/extract_persumm_%j.out
#
# Per-summary extractive density and LongDocFACTScore for every system, one
# CSV per model scale, then a combined CSV. Reads the generations from
# results/ and writes to results/per_summary/. About 8 minutes on one A6000.
#
# Submit from the repository root:
#     sbatch evaluation/run_extract_per_summary.slurm

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-/scratch/$USER/HF_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
import numpy; print(f'numpy          {numpy.__version__}')
"
python -c "from longdocfactscore.ldfacts import LongDocFACTScore; print('longdocfactscore OK')" \
    || { echo 'longdocfactscore NOT AVAILABLE, aborting.'; exit 1; }
echo "========================================"
echo ""

SCRIPT="$PROJECT_DIR/evaluation/extract_per_summary.py"
DATASET="$PROJECT_DIR/data/dataset.jsonl"
SUMMARIES="$PROJECT_DIR/data/summaries.jsonl"
TEST_IDS="$PROJECT_DIR/results/llama3.1-8b-lora/train/test_paper_ids.json"
OUTPUT_DIR="$PROJECT_DIR/results/per_summary"

GEN_70B="$PROJECT_DIR/results/llama3.3-70b-qlora/eval/eval_generations.json"
GEN_8B="$PROJECT_DIR/results/llama3.1-8b-lora/eval/eval_generations.json"

mkdir -p "$OUTPUT_DIR"

echo "Extraction started at $(date)"

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

echo ""
echo "========================================"
echo "Per-summary extraction: 8B"
echo "========================================"
# If a generations file was written with a different --model_tag than its
# scale, the keys can be relabelled on the fly, e.g.
#     --rename finetuned_70B=finetuned_8B --rename base_70B=base_8B
python "$SCRIPT" \
    --generations "$GEN_8B" \
    --scale 8B \
    --dataset "$DATASET" \
    --summaries "$SUMMARIES" \
    --test_ids "$TEST_IDS" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda
EXIT_8B=$?
echo "8B extraction exit code: $EXIT_8B"

# Combine into one CSV. The teacher rows are identical in both files and are
# taken from the 70B file only.
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
