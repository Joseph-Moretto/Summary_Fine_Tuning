# logs/

SLURM job output lands here when a job script is submitted from the
repository root, for example:

```bash
sbatch training/run_finetune_70b.slurm      # -> logs/finetune_70b_<jobid>.out
sbatch evaluation/run_eval_8b.slurm         # -> logs/eval_8b_<jobid>.out
```

Each script writes stdout and stderr to the same file. The directory must
exist before submission (SLURM does not create it), which is why this README
is tracked while the log files themselves are ignored by `.gitignore`.

The logs of the reported runs were moved to `results/<run>/logs/` and
`results/per_summary/logs/` so that they sit next to the outputs they produced.
