#!/usr/bin/env bash
#SBATCH --job-name=clas-final
#SBATCH --partition=gpu-h200
#SBATCH --account=proj-606
#SBATCH --qos=proj-606
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x_%j.out
set -euo pipefail
MODEL="${1:?model}"
SHORT="${MODEL##*/}"
source "$HOME/.clas_env"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CLAS_ROOT"
read -r VEC LAYER C < <("$CLAS_PY" -c "
import json; b=json.load(open('results/best_config.json'))['$SHORT']
print(b['vector'], b['layer'], b['strength'])")
echo "=== $(date) host=$(hostname) model=$MODEL best: vector=$VEC layer=$LAYER strength=$C"
"$CLAS_PY" scripts/run_steering.py --model "$MODEL" --split test --pairs all \
    --vectors "$VEC" --layers "$LAYER" --strengths "$C" --max-new-tokens 1024
"$CLAS_PY" scripts/score_many.py --root "$CLAS_GEN/test/$SHORT"
echo "=== done $(date)"
