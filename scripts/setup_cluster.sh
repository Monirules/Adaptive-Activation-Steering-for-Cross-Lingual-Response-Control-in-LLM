#!/usr/bin/env bash
# One-time setup on the ARCC2 login node.
# Usage (from ~/clas_project):  bash scripts/setup_cluster.sh
set -euo pipefail

REPO="$HOME/clas_project"
SCR="${CLAS_SCRATCH:-/N/lustre/scratch/$USER/clas-data}"
VENV="$HOME/.venvs/clas"
OFFICIAL="$REPO/official"

export PATH="$HOME/.local/bin:$PATH"

echo "== 1/6 folders"
mkdir -p "$SCR/hf_cache" "$SCR/data" "$SCR/generations" "$SCR/vectors" \
         "$SCR/uv_cache" "$SCR/vllm_cache" "$REPO/logs/slurm"
export UV_CACHE_DIR="$SCR/uv_cache"

echo "== 2/6 uv and a home-folder Python (GPU nodes cannot see /usr/bin/python3.11)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export UV_PYTHON_PREFERENCE=only-managed
uv python install 3.11

echo "== 3/6 official CLaS repo -> $OFFICIAL"
if [ ! -d "$OFFICIAL/.git" ]; then
  git clone https://github.com/d-gurgurov/CLaS-SemEval-2027 "$OFFICIAL"
else
  git -C "$OFFICIAL" pull --ff-only || true
fi

echo "== 4/6 virtual environment -> $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.11 "$VENV"
fi
uv pip install --python "$VENV/bin/python" -r "$OFFICIAL/requirements.txt"
uv pip install --python "$VENV/bin/python" pandas matplotlib
uv pip install --python "$VENV/bin/python" "numpy==1.26.4" "pyarrow>=15,<20" "datasets==3.6.0"
echo "== 5/6 environment file ~/.clas_env"
cat > "$HOME/.clas_env" <<ENV
# CLaS project environment (written by scripts/setup_cluster.sh)
export CLAS_ROOT=$REPO
export CLAS_OFFICIAL=$OFFICIAL
export CLAS_SCRATCH=$SCR
export CLAS_DATA=$SCR/data
export CLAS_GEN=$SCR/generations
export CLAS_VECTORS=$SCR/vectors
export HF_HOME=$SCR/hf_cache
export UV_CACHE_DIR=$SCR/uv_cache
export VLLM_CACHE_ROOT=$SCR/vllm_cache
export CLAS_PY=$VENV/bin/python
export PYTHONPATH=$OFFICIAL:$REPO
export VIRTUAL_ENV=$VENV
export PATH=$VENV/bin:\$HOME/.local/bin:\$PATH
ENV
grep -q "alias clas=" "$HOME/.bashrc" || \
  echo "alias clas='source ~/.clas_env && cd ~/clas_project'" >> "$HOME/.bashrc"

echo "== 6/6 check"
echo "venv python -> $(readlink -f "$VENV/bin/python")"
"$VENV/bin/python" -c "import vllm, transformers, torch, fasttext; print('vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__)"
echo
echo "Done. Now run:  source ~/.bashrc   then   clas"
