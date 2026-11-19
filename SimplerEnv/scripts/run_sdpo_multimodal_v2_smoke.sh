#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${EXTERNAL_FEEDBACK_URL:?Set EXTERNAL_FEEDBACK_URL to a Qwen2.5-VL OpenAI-compatible /v1 endpoint}"
: "${MS_ASSET_DIR:=/home/skylar/@lab/hf-cache/maniskill}"
: "${VLA_PATH:=/home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03}"
: "${RUN_ROOT:=experiments/sdpo-multimodal-v2}"
: "${EXTERNAL_FEEDBACK_MODEL:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${EXTERNAL_FEEDBACK_TIMEOUT:=120}"
: "${SEED:=0}"
: "${EPISODE_LEN:=4}"
: "${STEPS_MAX:=8}"
: "${WANDB_MODE:=offline}"

export MS_ASSET_DIR WANDB_MODE PYTHONNOUSERSITE=1
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"

if (( EPISODE_LEN < 1 || STEPS_MAX != 2 * EPISODE_LEN )); then
    echo "Smoke requires STEPS_MAX=2*EPISODE_LEN" >&2
    exit 2
fi

name="multimodal-v2-sdpo-seed${SEED}-${STEPS_MAX}steps-smoke"
run_dir="$RUN_ROOT/$name"
if [[ -e "$run_dir/wandb" ]]; then
    echo "Run directory already exists: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"

WANDB_DIR="$run_dir" conda run --no-capture-output -n rlvla_env python -u \
    SimplerEnv/simpler_env/train_ms3_ppo.py \
    --alg-name sdpo \
    --sdpo-feedback-mode correct \
    --sdpo-feedback-version multimodal-correction-v2 \
    --external-feedback-url "$EXTERNAL_FEEDBACK_URL" \
    --external-feedback-model "$EXTERNAL_FEEDBACK_MODEL" \
    --external-feedback-timeout "$EXTERNAL_FEEDBACK_TIMEOUT" \
    --env-id PutOnPlateInScene25Main-v3 \
    --vla-path "$VLA_PATH" \
    --vla-load-path '' \
    --seed "$SEED" \
    --num-envs 2 \
    --episode-len "$EPISODE_LEN" \
    --steps-max "$STEPS_MAX" \
    --interval-eval 1 \
    --interval-save 1 \
    --eval-episodes 2 \
    --buffer-inferbatch 2 \
    --buffer-minibatch 1 \
    --alg-gradient-accum 1 \
    --alg-ppo-epoch 1 \
    --name "$name" \
    --no-save-render
