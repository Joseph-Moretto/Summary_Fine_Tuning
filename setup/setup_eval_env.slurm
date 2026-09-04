#!/bin/bash
#SBATCH --job-name=setup_eval_env
#SBATCH --account=nsdpi_ext_paid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --output=logs/setup_eval_env_%j.out
#SBATCH --error=logs/setup_eval_env_%j.out
#
# Builds the "summ_eval" conda environment used by everything in evaluation/.
# Runs as a GPU job so the CUDA check at the end is meaningful, but the same
# commands work in an interactive shell. Package versions observed in the
# reported runs are listed in requirements-eval.txt.

set -e

module purge
module load miniforge

# Keep ~/.local user-site packages out of the environment.
export PYTHONNOUSERSITE=1

conda deactivate 2>/dev/null || true
conda env remove -n summ_eval -y 2>/dev/null || true
conda create -n summ_eval python=3.11 -y
conda activate summ_eval

echo "========================================"
echo "Python: $(which python)"
echo "pip:    $(which pip)"
echo "========================================"

# 1. PyTorch from the CUDA 13.0 index (matches the cluster driver)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# 2. Model loading. Installed before the metrics so that their dependency
#    resolution sets the transformers / huggingface-hub baseline.
pip install "transformers>=5.5.4" "peft>=0.19.0" "bitsandbytes>=0.49.2" accelerate

# 3. Reference-based metrics and statistics
pip install rouge-score nltk scipy pandas

# bert-score without its dependencies: the spaCy requirement fails to build
# here and is not needed for scoring. Its real runtime dependencies follow.
pip install bert-score --no-deps
pip install packaging filelock requests tqdm

# 4. LongDocFACTScore (retrieval-based factual consistency,
#    https://github.com/jbshp/LongDocFACTScore) and the sentence-embedding
#    model it retrieves with
pip install longdocfactscore sentence-transformers

# 5. NLTK data for BLEU and METEOR
python -c "
import nltk
for resource in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4'):
    nltk.download(resource)
"

echo ""
echo "========================================"
echo "Verification"
echo "========================================"
python -c "
import torch; print(f'torch          {torch.__version__}  CUDA={torch.cuda.is_available()}')
import transformers; print(f'transformers   {transformers.__version__}')
import peft; print(f'peft           {peft.__version__}')
import bitsandbytes; print(f'bitsandbytes   {bitsandbytes.__version__}')
import huggingface_hub; print(f'huggingface_hub {huggingface_hub.__version__}')
from rouge_score import rouge_scorer; print('rouge-score .... OK')
from bert_score import score; print('bert-score ..... OK')
from nltk.translate.bleu_score import sentence_bleu; print('BLEU ........... OK')
from nltk.translate.meteor_score import meteor_score; print('METEOR ......... OK')
from longdocfactscore.ldfacts import LongDocFACTScore; print('LongDocFACTScore OK')
"
echo "========================================"
echo "Environment setup complete."
echo ""
echo "To use this environment:"
echo "  export PYTHONNOUSERSITE=1"
echo "  conda activate summ_eval"
