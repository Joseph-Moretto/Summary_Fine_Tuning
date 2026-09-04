#!/bin/bash
#SBATCH -A nsdpi_ext_paid
#SBATCH --job-name="setup_summac"
#SBATCH --error="/scratch/jam5cq/Summary_Fine_Tuning/Logs/setup_summac_%j.out"
#SBATCH --output="/scratch/jam5cq/Summary_Fine_Tuning/Logs/setup_summac_%j.out"
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1

set -e

module purge
module load miniforge

export PYTHONNOUSERSITE=1

# Create isolated env for SummaC
conda deactivate 2>/dev/null || true
conda env remove -n summac_eval -y 2>/dev/null || true
conda create -n summac_eval python=3.11 -y
conda activate summac_eval

echo "========================================"
echo "Python: $(which python)"
echo "pip:    $(which pip)"
echo "========================================"

# PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu130

# SummaC with its pinned dependencies (this is why we need a separate env)
pip install summac

# This will install transformers==4.35.2, huggingface-hub==0.17.0, etc.
# which is exactly what SummaC needs

# Additional deps
pip install scipy numpy

# Verify
echo ""
echo "========================================"
echo "Verification"
echo "========================================"
python -c "
import torch; print(f'torch          {torch.__version__}  CUDA={torch.cuda.is_available()}')
import transformers; print(f'transformers   {transformers.__version__}')
from summac.model_summac import SummaCZS; print('SummaC ......... OK')
"
echo "========================================"
echo "SummaC environment ready."
echo ""
echo "To use:"
echo "  export PYTHONNOUSERSITE=1"
echo "  conda activate summac_eval"
