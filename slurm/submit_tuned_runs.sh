#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/clas_project"
mkdir -p logs/slurm

LLAMA=meta-llama/Llama-3.1-8B-Instruct

SWEEP=$(sbatch --parsable --job-name="tuning-Llama" slurm/tuning_sweep.sbatch \
  "$LLAMA" "16 20 24" "0.75 1.5" "12 16 20 24" "0.5 0.75 1 1.5")
echo "tuning sweep job $SWEEP"

for TRACK in t2 t1 t3; do
  sbatch --dependency=afterok:"$SWEEP" --job-name="tuned-Llama-$TRACK" \
    slurm/tuned_test.sbatch "$LLAMA" "$TRACK" de
done

squeue -u "$USER"
