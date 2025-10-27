#!/bin/bash
#SBATCH --job-name=lyricwhiz              # Job name
#SBATCH --gpus=1                          # Request 1 GPU
#SBATCH --gres=min-vram:30g               # Request GPU with at least 30GB VRAM
#SBATCH --cpus-per-task=4                 # 4 CPU cores
#SBATCH --mem=24G                         # 24 GB RAM
#SBATCH --time=24:00:00                   # 10 hours
#SBATCH --output=lyricwhiz_%j.log         # Log file
#SBATCH --gres=min-cuda-cc:90             # Compute capability ≥ 9.0 (e.g. H100)

# ----------------------------
#   Job environment setup
# ----------------------------

echo "============================================"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "GPU Info:"
nvidia-smi
echo "============================================"

# 1️⃣ Load conda/mamba and activate environment
# (Use `source` instead of smart quotes — yours had curly ones)
eval "$(mamba shell hook --shell bash)"
mamba activate clarity

# 2️⃣ Redirect caches to /scratch (to save /home quota)
export TRANSFORMERS_CACHE=/scratch/work/mohanv1/.cache/huggingface
export HF_HOME=/scratch/work/mohanv1/.cache/huggingface
export TORCH_HOME=/scratch/work/mohanv1/.cache/torch
export XDG_CACHE_HOME=/scratch/work/mohanv1/.cache
export MPLCONFIGDIR=/scratch/work/mohanv1/.cache/matplotlib

mkdir -p $TRANSFORMERS_CACHE $TORCH_HOME $XDG_CACHE_HOME $MPLCONFIGDIR

# 3️⃣ Optional: improve PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5️⃣ Run the main script
echo "Starting compute_lyricwhiz.py..."
python compute_lyricwhiz.py split=train baseline.system=lyricwhiz
python compute_lyricwhiz.py split=valid baseline.system=lyricwhiz

echo "============================================"
echo "Job finished on $(date)"
echo "============================================"
