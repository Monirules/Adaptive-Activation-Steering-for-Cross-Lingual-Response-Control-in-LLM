#!/usr/bin/env bash

set -euo pipefail
cd "$HOME/clas_project"
mkdir -p logs/slurm

MODELS=(meta-llama/Llama-3.1-8B-Instruct CohereLabs/tiny-aya-global)
for MODEL in "${MODELS[@]}"; do
  for SPLIT in test dev; do
    for INSTR in english none; do
      sbatch --job-name="b-${MODEL##*/}-$SPLIT-$INSTR" \
        slurm/baseline.sbatch "$MODEL" "$SPLIT" "$INSTR" t1
    done
  done
done
squeue -u "$USER"
