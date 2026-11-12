#!/usr/bin/env bash
#SBATCH --job-name=rlvla-v1
#SBATCH --partition=h800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/s2250018/project/rlvla-v1-%j.out

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rlvla_env
cd /public/home/s2250018/project/RL4VLA

export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/maniskill-assets}"
export VLA_PATH="${VLA_PATH:-$HF_HOME/models/openvla-7b-rlvla-warmup}"
export WANDB_MODE=offline
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export RUN_ROOT="${RUN_ROOT:-experiments/sdpo-v1}"
export SEEDS='0 1'

nvidia-smi
bash SimplerEnv/scripts/run_sdpo_v1_comparison.sh
