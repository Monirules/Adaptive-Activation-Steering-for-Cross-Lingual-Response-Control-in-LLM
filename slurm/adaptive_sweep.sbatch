#!/usr/bin/env bash
#SBATCH --job-name=clas-adaptive-sweep
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
MODEL="${1:?model}"
SHORT="${MODEL##*/}"

source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CLAS_ROOT"

read -r VEC LAYER C < <("$CLAS_PY" -c "import json,sys; b=json.load(open('results/best_config.json'))[sys.argv[1]]; print(b['vector'], b['layer'], float(b['strength']))" "$SHORT")
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "model=$MODEL vector=$VEC layer=$LAYER strength=$C"

echo "=== smoke test"
"$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split dev --track t1 --pairs sweep --k 2 \
    --vectors "$VEC" --layers "$LAYER" --strengths "$C" --adaptive gate --warmup 4 \
    --max-new-tokens 48 --out-root "$CLAS_GEN/smoke_adaptive"

SETTINGS=("gate 8 0 0" "gate 8 0.1 0" "gate 32 0 0" "gate 8 0 0.5" "decay 16 0 0.5" "decay 32 0 0.25")
for S in "${SETTINGS[@]}"; do
  read -r MODE W TAU LATE <<< "$S"
  for TRACK in t1 t2; do
    "$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split dev --track "$TRACK" --pairs sweep \
        --vectors "$VEC" --layers "$LAYER" --strengths "$C" --adaptive "$MODE" --warmup "$W" \
        --tau "$TAU" --late "$LATE" --out-root "$CLAS_GEN/sweep"
  done
done

"$CLAS_PY" scripts/score_many.py --root "$CLAS_GEN/sweep/$SHORT"
"$CLAS_PY" scripts/pick_adaptive.py --model "$SHORT"

echo "=== done $(date)"
