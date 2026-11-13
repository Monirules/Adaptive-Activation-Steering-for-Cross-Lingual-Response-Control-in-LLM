#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/clas_project"
mkdir -p logs/slurm

LLAMA=meta-llama/Llama-3.1-8B-Instruct

SWEEP=$(sbatch --parsable --job-name="adaptive-sweep-Llama" slurm/adaptive_sweep.sbatch "$LLAMA")
echo "adaptive sweep job $SWEEP"

for TRACK in t2 t1 t3; do
  sbatch --dependency=afterok:"$SWEEP" --job-name="adaptive-Llama-$TRACK" \
    slurm/adaptive_test.sbatch "$LLAMA" "$TRACK" de adaptive
done

TUNE=$(squeue -h -u "$USER" -n tuning-Llama -o %i | head -n 1)
DEP="afterok:$SWEEP"
if [ -n "$TUNE" ]; then
  DEP="$DEP:$TUNE"
fi
sbatch --dependency="$DEP" --job-name="tuned-adaptive-Llama-t2" \
  slurm/adaptive_test.sbatch "$LLAMA" t2 de tuned-adaptive

squeue -u "$USER"
