#!/usr/bin/env bash
#SBATCH --job-name=clas-base
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

MODEL="${1:?model}"; SPLIT="${2:?split}"; INSTR="${3:?instruction}"
TRACK="${4:-t1}"; K="${5:-}"; T3_TARGET="${6:-de}"

source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

case "$SPLIT" in
  dev|test) QDIR="$CLAS_DATA/validation_$SPLIT"; OUTROOT="$CLAS_GEN/$SPLIT" ;;
  smoke)    QDIR="$CLAS_DATA/validation_dev";    OUTROOT="$CLAS_GEN/smoke"; K="${K:-2}" ;;
  *) echo "SPLIT must be dev, test or smoke"; exit 1 ;;
esac
KARG=()
if [ -n "$K" ]; then KARG=(-k "$K"); fi

TRACK_DIR="$TRACK"
if [ "$TRACK" = "t3" ]; then TRACK_DIR="t3/$T3_TARGET"; fi
GEN="$OUTROOT/${MODEL##*/}/$TRACK_DIR/$INSTR"
echo "=== $(date)  host=$(hostname)  job=${SLURM_JOB_ID:-local}"
echo "model=$MODEL split=$SPLIT instruction=$INSTR track=$TRACK k=${K:-all}"
echo "questions=$QDIR"
echo "output=$GEN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$CLAS_OFFICIAL"
echo "=== generate"
"$CLAS_PY" run_baseline.py --track "$TRACK" --model "$MODEL" --instruction "$INSTR" --t3-target "$T3_TARGET" \
    --questions-dir "$QDIR" --out "$OUTROOT" ${KARG[@]+"${KARG[@]}"}

echo "=== evaluate"
"$CLAS_PY" evaluate.py --gen-dir "$GEN"

echo "=== done $(date)"
