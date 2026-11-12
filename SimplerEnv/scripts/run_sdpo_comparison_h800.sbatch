#!/usr/bin/env bash
#SBATCH --job-name=rlvla-sdpo
#SBATCH --partition=h800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/s2250018/project/rlvla-sdpo-%j.out

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rlvla_env
cd "$HOME/project/RL4VLA"

export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/maniskill-assets}"
export VLA_PATH="${VLA_PATH:-$HF_HOME/models/openvla-7b-rlvla-warmup}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'

case "${1:-matrix}" in
    smoke)
        smoke_dir="${SLURM_TMPDIR:-/tmp}/rlvla-smoke-${SLURM_JOB_ID:-manual}"
        mkdir -p "$smoke_dir/wandb"
        export WANDB_DIR="$smoke_dir/wandb"
        trap 'rm -rf -- "$smoke_dir"' EXIT
        python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
            --alg-name sdpo \
            --sdpo-feedback-mode correct \
            --env-id PutOnPlateInScene25Main-v3 \
            --vla-path "$VLA_PATH" \
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
        latest_run="$(find "$WANDB_DIR/wandb" -maxdepth 1 -type d -name 'offline-run-*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d ' ' -f 2-)"
        checkpoint="$latest_run/glob/steps_0000"
        test -f "$checkpoint/sdpo_protocol.json"
        python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
            --alg-name sdpo \
            --env-id PutOnPlateInScene25Main-v3 \
            --vla-path "$VLA_PATH" \
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
        IFS=, read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm did not set CUDA_VISIBLE_DEVICES}"
        if (( ${#allocated_gpus[@]} != 2 )); then
            echo "Expected exactly 2 allocated GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
            exit 2
        fi
        run_root="$HOME/project/RL4VLA/experiments/${SLURM_JOB_ID:-manual}"
        mkdir -p "$run_root"
        run_seed() {
            local seed="$1"
            local gpu_index="$2"
            local output_dir="$run_root/seed${seed}"
            mkdir -p "$output_dir"
            CUDA_VISIBLE_DEVICES="${allocated_gpus[$gpu_index]}" \
            WANDB_DIR="$output_dir" \
            SEEDS="$seed" \
            NUM_ENVS=2 \
            EPISODE_LEN=80 \
            STEPS_MAX=480 \
            INTERVAL_EVAL=1 \
            EVAL_EPISODES=2 \
            INTERVAL_SAVE=40 \
            SAVE_RENDER=false \
            WANDB_MODE="$WANDB_MODE" \
                bash SimplerEnv/scripts/run_sdpo_comparison.sh
        }
        run_seed 0 0 & worker_a=$!
        run_seed 1 1 & worker_b=$!
        wait "$worker_a"
        wait "$worker_b"
        run_seed 2 0
        ;;
    *)
        echo "Usage: sbatch $0 [smoke|matrix]" >&2
        exit 2
        ;;
esac
