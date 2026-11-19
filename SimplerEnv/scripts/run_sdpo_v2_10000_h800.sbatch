#!/usr/bin/env bash
#SBATCH --job-name=rlvla-v2-10k
#SBATCH --partition=h800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --output=/public/home/s2250018/project/rlvla-v2-10k-%j.out
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rlvla_env
cd "$HOME/project/RL4VLA"
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/maniskill-assets}"
export VLA_PATH="${VLA_PATH:-$HOME/hf-cache/models/openvla-7b-rlvla-warmup}"
export EXTERNAL_FEEDBACK_MODEL="${EXTERNAL_FEEDBACK_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
PY

run_root="${RUN_ROOT:-$HOME/project/RL4VLA/experiments/sdpo-v2-10000}"
mkdir -p "$run_root"
arm="${1:-sdpo}"
steps_max="${STEPS_MAX:-10000}"
checkpoint_steps="${CHECKPOINT_STEPS:-480,2400,5000,10000}"
case "$arm" in
  sdpo)
    : "${EXTERNAL_FEEDBACK_URL:?Set endpoint reachable from H800 compute node}"
    export EXTERNAL_FEEDBACK_API_KEY="$(cat "$HOME/.config/rlvla/qwen_v2.key")"
    arm_args=(--alg-name sdpo --sdpo-feedback-mode correct
      --sdpo-feedback-version multimodal-correction-v2
      --external-feedback-url "$EXTERNAL_FEEDBACK_URL"
      --external-feedback-model "$EXTERNAL_FEEDBACK_MODEL"
      --external-feedback-timeout "${EXTERNAL_FEEDBACK_TIMEOUT:-180}")
    minibatch=1
    ;;
  ppo)
    arm_args=(--alg-name ppo --sdpo-feedback-mode none)
    minibatch=2
    ;;
  *) echo "Usage: sbatch $0 [sdpo|ppo]" >&2; exit 2 ;;
esac
run_name="${RUN_NAME:-${arm}-v2-seed0-${steps_max}steps}"
WANDB_DIR="$run_root" python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
  "${arm_args[@]}" \
  --env-id PutOnPlateInScene25Main-v3 --vla-path "$VLA_PATH" --vla-load-path '' \
  --seed 0 --num-envs "${NUM_ENVS:-8}" --episode-len 80 --steps-max "$steps_max" \
  --checkpoint-steps "$checkpoint_steps" --interval-eval 0 --interval-save 0 \
  --buffer-inferbatch "${BUFFER_INFERBATCH:-8}" --buffer-minibatch "$minibatch" \
  --alg-gradient-accum 1 --alg-ppo-epoch 1 --name "$run_name" --no-save-render
