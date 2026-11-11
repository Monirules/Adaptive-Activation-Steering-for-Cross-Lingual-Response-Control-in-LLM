#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/clas_project"
mkdir -p logs/slurm

LLAMA=meta-llama/Llama-3.1-8B-Instruct
AYA=CohereLabs/tiny-aya-global

for TRACK in t2 t3; do
  sbatch --job-name="test-Llama-$TRACK" slurm/track_test.sbatch "$LLAMA" "$TRACK" de
  for MODEL in "$LLAMA" "$AYA"; do
    sbatch --job-name="base-${MODEL##*/}-$TRACK" \
      slurm/baseline.sbatch "$MODEL" test english "$TRACK" "" de
  done
done

sbatch --job-name="sweep-test-tiny-aya" slurm/sweep_select_test.sbatch "$AYA" "6 8 10 12" "0.5 1" de

for MODEL in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct; do
  sbatch --job-name="extract-${MODEL##*/}" slurm/extract.sbatch "$MODEL"
done

squeue -u "$USER"
