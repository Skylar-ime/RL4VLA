#!/usr/bin/env bash
#SBATCH --job-name=rlvla-v2-eval
#SBATCH --partition=h800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/s2250018/project/rlvla-v2-eval-%j.out
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rlvla_env
cd "$HOME/project/RL4VLA"
export PYTHONNOUSERSITE=1
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/maniskill-assets}"
export VLA_PATH="${VLA_PATH:-$HOME/hf-cache/models/openvla-7b-rlvla-warmup}"
export EVAL_ROOT="${EVAL_ROOT:-experiments/sdpo-v2-10000/evaluation}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export EVAL_EPISODE_START=0 EVAL_EPISODES=60 EVAL_NUM_ENVS=6

arm="${1:?Usage: sbatch $0 ppo|sdpo}"
if [[ "$arm" != ppo && "$arm" != sdpo ]]; then
    echo "Arm must be ppo or sdpo" >&2
    exit 2
fi
train_run="${TRAIN_RUN:?Set TRAIN_RUN to the exact offline-run directory}"
if [[ ! -d "$train_run/glob" ]]; then
    echo "Missing training run: $train_run" >&2
    exit 2
fi

if [[ "$arm" == ppo ]]; then
    EVAL_LABEL=warmup-validation-60-h800 VLA_LOAD_PATH='' \
        bash SimplerEnv/scripts/run_sdpo_v2_10000_eval.sh
fi

for steps in 480 2400 5000 10000; do
    printf -v checkpoint 'step_%06d' "$steps"
    checkpoint_dir="$train_run/glob/$checkpoint"
    if [[ ! -s "$checkpoint_dir/environment_steps.json" || ! -s "$checkpoint_dir/adapter_model.safetensors" ]]; then
        echo "Incomplete checkpoint: $checkpoint_dir" >&2
        exit 1
    fi
    EVAL_LABEL="$arm-$checkpoint-validation-60-h800" VLA_LOAD_PATH="$checkpoint_dir" \
        bash SimplerEnv/scripts/run_sdpo_v2_10000_eval.sh
done
