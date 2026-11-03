#!/usr/bin/env bash
# Downloads data and models on the LOGIN node (it has internet; GPU jobs may not).
# Run inside tmux, from ~/clas_project:
#   bash scripts/download_all.sh 2>&1 | tee logs/download.log
set -euo pipefail
source "$HOME/.clas_env"

LANGS="ar de en es fr ja zh ru hi ko cs el fa vi uk mt no sw th tr"

echo "== 1/3 FLORES+ (train) and CLaS-Bench (validation) for the 20 task languages"
cd "$CLAS_OFFICIAL"
"$CLAS_PY" download_data.py --out "$CLAS_DATA" --langs $LANGS

echo "== 2/3 models into $HF_HOME"
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*"
hf download CohereLabs/tiny-aya-global
hf download Qwen/Qwen3-8B

echo "== 3/3 fastText language ID model (used by evaluate.py)"
hf download facebook/fasttext-language-identification model.bin

echo
echo "line counts (train should be 2009, validation should be 70):"
for c in $LANGS; do
  printf "  %-3s train %5s   validation %3s\n" "$c" \
    "$(wc -l < "$CLAS_DATA/train/$c.txt")" "$(wc -l < "$CLAS_DATA/validation/$c.txt")"
done
echo "ALL DOWNLOADS DONE"
