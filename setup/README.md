# setup/

Environment definitions for the two conda environments the project ran in,
and the package versions that produced the reported results.

| File | Purpose |
|---|---|
| `requirements-train.txt` | pinned packages for the training environment (`qlora_summ_ft3`) |
| `requirements-eval.txt` | packages for the evaluation environment (`summ_eval`) |
| `setup_eval_env.slurm` | builds `summ_eval` from scratch as a SLURM job, in the order that avoids the dependency conflicts described below |

## Why two environments

Training needs `trl`, `peft`, `bitsandbytes` and a recent `transformers`.
Evaluation additionally needs `bert-score`, `sentence-transformers` and
`longdocfactscore`, whose dependency chains are easier to manage separately.
An earlier attempt to include SummaC in the evaluation stack forced a
downgrade to transformers 4.35 and was abandoned (see `archive/summac/`).

Both environments use Python 3.11 and PyTorch 2.11.0 built for CUDA 13.0,
matching the driver on the cluster's RTX A6000 nodes.

## Versions used for the reported runs

Taken from the headers of the job logs in `results/*/logs/`.

| Package | 8B training (2026-04-26) | 70B training (2026-05-04) | Evaluation (2026-05-05/06) |
|---|---|---|---|
| torch | 2.11.0+cu130 | 2.11.0+cu130 | 2.11.0+cu130 |
| transformers | 5.5.4 | 5.6.2 | 5.6.2 |
| peft | 0.19.0 | 0.19.1 | 0.19.1 |
| bitsandbytes | 0.49.2 | 0.49.2 | 0.49.2 |
| trl | 1.1.0 | 1.1.0 | |
| datasets | 4.8.4 | 4.8.4 | |
| tokenizers | 0.22.2 | 0.22.2 | |

`rouge-score`, `bert-score`, `nltk`, `scipy`, `sentence-transformers` and
`longdocfactscore` were installed unpinned; the logs only record that they
imported successfully. `flash-attn` was not installed, so both training runs
used PyTorch SDPA attention (the training script logs the fallback).

## Training environment

```bash
conda create -n qlora_summ_ft3 python=3.11 -y
conda activate qlora_summ_ft3
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r setup/requirements-train.txt
```

## Evaluation environment

Either submit `setup_eval_env.slurm` from the repository root, or run the
same steps interactively:

```bash
conda create -n summ_eval python=3.11 -y
conda activate summ_eval
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install "transformers>=5.5.4" "peft>=0.19.0" "bitsandbytes>=0.49.2" accelerate
pip install rouge-score nltk scipy pandas
pip install bert-score --no-deps
pip install packaging filelock requests tqdm
pip install longdocfactscore sentence-transformers
python -c "import nltk; [nltk.download(r) for r in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4')]"
```

`bert-score` is installed with `--no-deps` because its spaCy dependency
failed to build on the cluster and is not needed for scoring; its actual
runtime dependencies (`packaging`, `filelock`, `requests`, `tqdm`, plus
`transformers` and `torch`) are installed separately. `longdocfactscore`
(https://github.com/jbshp/LongDocFACTScore) downloads its scoring and
retrieval models from the Hugging Face Hub on first use.

The NLTK downloads provide the tokenizer for METEOR/BLEU and the WordNet data
METEOR uses for synonym matching. `evaluate.py` fetches them automatically if
they are missing, which requires network access on the compute node.

## Model access and caches

The Llama 3 checkpoints are gated. Accept the license on the Hub and run
`huggingface-cli login` once in each environment. The job scripts set
`HF_HOME` to `/scratch/$USER/HF_cache` unless it is already defined; the
70B checkpoint alone is about 140 GB, so keep the cache on fast scratch
storage rather than in a home directory.

BERTScore uses `microsoft/deberta-xlarge-mnli`, also fetched into `HF_HOME`
on first use.

## Cluster specifics

The job scripts assume the UVA cluster: `module load miniforge` to get conda,
`--account`, `--partition gpu` and `--gres=gpu:a6000:N` in the `#SBATCH`
headers, and `PYTHONNOUSERSITE=1` in the evaluation jobs to keep packages
installed under `~/.local` from shadowing the environment. Edit the header
lines for another scheduler; nothing in the Python code depends on SLURM.
