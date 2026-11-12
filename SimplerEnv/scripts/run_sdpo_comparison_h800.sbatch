#!/usr/bin/env bash
#SBATCH --job-name=rlvla-sdpo
#SBATCH --partition=h800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=rlvla-sdpo-%j.out

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rlvla_env
cd "$HOME/RL4VLA"

export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/maniskill-assets}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'

case "${1:-matrix}" in
    smoke)
        python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
            --alg-name sdpo \
            --sdpo-feedback-mode correct \
            --env-id PutOnPlateInScene25Main-v3 \
            --vla-path gen-robot/openvla-7b-rlvla-warmup \
            --vla-load-path '' \
            --seed 0 \
            --num-envs 1 \
            --episode-len 8 \
            --steps-max 8 \
            --buffer-minibatch 1 \
            --alg-gradient-accum 1 \
            --eval-episodes 2 \
            --name h800-sdpo-mechanism-smoke \
            --no-save-render
        latest_run="$(find wandb -maxdepth 1 -type d -name 'offline-run-*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d ' ' -f 2-)"
        checkpoint="$latest_run/glob/steps_0000"
        test -f "$checkpoint/sdpo_protocol.json"
        python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
            --alg-name sdpo \
            --env-id PutOnPlateInScene25Main-v3 \
            --vla-path gen-robot/openvla-7b-rlvla-warmup \
            --vla-load-path "$checkpoint" \
            --seed 0 \
            --num-envs 1 \
            --episode-len 8 \
            --eval-only \
            --eval-episodes 2 \
            --eval-obj-set test \
            --name h800-sdpo-checkpoint-restore-smoke \
            --no-save-render
        ;;
    matrix)
        seeds="0 1 2"
        if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
            seeds="$SLURM_ARRAY_TASK_ID"
        fi
        SEEDS="$seeds" \
        NUM_ENVS=2 \
        EPISODE_LEN=80 \
        STEPS_MAX=480 \
        INTERVAL_EVAL=1 \
        EVAL_EPISODES=2 \
        INTERVAL_SAVE=40 \
        SAVE_RENDER=false \
        WANDB_MODE="$WANDB_MODE" \
            bash SimplerEnv/scripts/run_sdpo_comparison.sh
        ;;
    *)
        echo "Usage: sbatch $0 [smoke|matrix]" >&2
        exit 2
        ;;
esac
