#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/clas_project"
mkdir -p logs/slurm

sbatch --job-name="sweep-test-Qwen2.5-7B" slurm/sweep_select_test.sbatch \
  Qwen/Qwen2.5-7B-Instruct "10 14 18 21 24" "0.5 1 2" de t1
sbatch --job-name="base-Qwen2.5-7B-t1" slurm/baseline.sbatch \
  Qwen/Qwen2.5-7B-Instruct test english t1 "" de

sbatch --job-name="sweep-test-Qwen2.5-14B" slurm/sweep_select_test.sbatch \
  Qwen/Qwen2.5-14B-Instruct "18 24 30 36 42" "0.5 1 2" de t1
sbatch --job-name="base-Qwen2.5-14B-t1" slurm/baseline.sbatch \
  Qwen/Qwen2.5-14B-Instruct test english t1 "" de

squeue -u "$USER"
