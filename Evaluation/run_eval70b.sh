#!/bin/bash
#SBATCH -A nsdpi_ext_paid
#SBATCH --job-name="eval_summ"
#SBATCH --error="/scratch/jam5cq/Summary_Fine_Tuning/Logs/eval_summ_%j.out"
#SBATCH --output="/scratch/jam5cq/Summary_Fine_Tuning/Logs/eval_summ_%j.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:2
#SBATCH --mem=200G
#SBATCH --time=24:00:00
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
import transformers; print(f'transformers   {transformers.__version__}')
import peft; print(f'peft           {peft.__version__}')
import bitsandbytes; print(f'bitsandbytes   {bitsandbytes.__version__}')
"
python -c "from rouge_score import rouge_scorer; print('rouge-score    OK')"
python -c "from bert_score import score; print('bert-score     OK')"
python -c "from nltk.translate.bleu_score import sentence_bleu; print('BLEU           OK')"
python -c "from nltk.translate.meteor_score import meteor_score; print('METEOR         OK')"
python -c "from summac.model_summac import SummaCZS; print('SummaC         OK')" 2>/dev/null \
    || echo "SummaC         NOT AVAILABLE (will be skipped)"
echo "========================================"
echo ""


echo "Evaluation started at $(date)"

# =====================================================================
# 70B QLoRA model
# =====================================================================
echo ""
echo "========================================"
echo "Evaluating 70B model (fine-tuned + base)"
echo "========================================"

python /scratch/jam5cq/Summary_Fine_Tuning/Evaluation/evaluate.py \
    --dataset /scratch/jam5cq/Summary_Fine_Tuning/dataset.jsonl \
    --summaries /scratch/jam5cq/Summary_Fine_Tuning/summaries.jsonl \
    --base_model meta-llama/Llama-3.3-70B-Instruct \
    --adapter_path /scratch/jam5cq/Summary_Fine_Tuning/Output_v2/adapter \
    --output_dir /scratch/jam5cq/Summary_Fine_Tuning/Output_70b_qlora \
    --test_ids /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/test_paper_ids.json \
    --model_tag 70B \
    --use_qlora


# python "/scratch/jam5cq/Summary_Fine_Tuning/Evaluation/evaluate.py" \
#     --metrics_only \
#     --generations_file /scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.3_70b_qlora_4bit/eval_generations.json \
#     --output_dir /scratch/jam5cq/Summary_Fine_Tuning/Output \
#     --test_ids "/scratch/jam5cq/Summary_Fine_Tuning/Output/Llama3.1_8b_lora/test_paper_ids.json"
    

EXIT_70B=$?
echo "70B evaluation exit code: $EXIT_70B"

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null


echo ""
echo "========================================"
echo "Evaluation completed at $(date)"
echo "========================================"
nvidia-smi

exit ${EXIT_70B:-0}
