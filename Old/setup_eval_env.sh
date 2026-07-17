#!/bin/bash
#SBATCH -A nsdpi_ext_paid
#SBATCH --job-name="setup_eval_env"
#SBATCH --error="/scratch/jam5cq/Summary_Fine_Tuning/Logs/setup_eval_%j.out"
#SBATCH --output="/scratch/jam5cq/Summary_Fine_Tuning/Logs/setup_eval_%j.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=16

set -e

module purge
module load miniforge

# Isolate from ~/.local user-site packages
export PYTHONNOUSERSITE=1

# Create fresh env
conda deactivate 2>/dev/null || true
conda env remove -n summ_eval -y 2>/dev/null || true
conda create -n summ_eval python=3.11 -y
conda activate summ_eval

echo "========================================"
echo "Python: $(which python)"
echo "pip:    $(which pip)"
echo "========================================"

# ---- 1. PyTorch (cu130 to match cluster CUDA) ----
pip install torch --index-url https://download.pytorch.org/whl/cu130

# ---- 2. Model loading (install FIRST so their deps set the baseline) ----
pip install "transformers>=5.5.4" "peft>=0.19.0" "bitsandbytes>=0.49.2" accelerate

# ---- 3. Reference-based metrics ----
pip install rouge-score nltk matplotlib

# bert-score without spacy (avoids Cython build failure)
pip install bert-score --no-deps
pip install pandas packaging filelock requests tqdm

# ---- 4. SummaC — factual consistency ----
# summac 0.0.4 pins transformers==4.35.2 in its requirements, which would
# downgrade transformers and huggingface-hub, breaking peft/accelerate.
# Install it --no-deps and manually add its real runtime dependency
# (sentence-transformers). summac only imports:
#   - transformers (already installed above, modern version works fine)
#   - sentence_transformers
#   - nltk (already installed)
#   - numpy (already installed)
pip install summac --no-deps
pip install sentence-transformers

# ---- 5. NLTK data for BLEU and METEOR ----
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
"

# ---- Verify all imports ----
echo ""
echo "========================================"
echo "Verification"
echo "========================================"
python -c "
import torch
print(f'torch          {torch.__version__}  CUDA={torch.cuda.is_available()}')
import transformers; print(f'transformers   {transformers.__version__}')
import peft; print(f'peft           {peft.__version__}')
import huggingface_hub; print(f'huggingface_hub {huggingface_hub.__version__}')
from rouge_score import rouge_scorer; print('rouge-score .... OK')
from bert_score import score; print('bert-score ..... OK')
from nltk.translate.bleu_score import sentence_bleu; print('BLEU ........... OK')
from nltk.translate.meteor_score import meteor_score; print('METEOR ......... OK')
from summac.model_summac import SummaCZS; print('SummaC ......... OK')
"
echo "========================================"
echo "Environment setup complete."
echo ""
echo "To use this env, always set:"
echo "  export PYTHONNOUSERSITE=1"
echo "  conda activate summ_eval"