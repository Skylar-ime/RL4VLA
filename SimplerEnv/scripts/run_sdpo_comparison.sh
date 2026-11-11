#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${MS_ASSET_DIR:=/home/skylar/@lab/hf-cache/maniskill}"
: "${VLA_PATH:=/home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03}"
: "${ENV_ID:=PutOnPlateInScene25Main-v3}"
: "${SEEDS:=0 1 2}"
: "${EPISODE_LEN:=4}"
: "${STEPS_MAX:=8}"
: "${NUM_ENVS:=2}"
: "${WANDB_MODE:=online}"

export MS_ASSET_DIR WANDB_MODE
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"

if (( NUM_ENVS < 2 || STEPS_MAX % (EPISODE_LEN * NUM_ENVS) != 0 )); then
    echo "NUM_ENVS must be at least 2, and STEPS_MAX must be divisible by EPISODE_LEN * NUM_ENVS" >&2
    exit 2
fi

run_arm() {
    local seed="$1"
    local alg="$2"
    local feedback_mode="$3"
    local name="${alg}-${feedback_mode}-seed${seed}-${STEPS_MAX}steps"
    local extra=()
    if [[ "$alg" == "sdpo" ]]; then
        extra+=(--sdpo-feedback-mode "$feedback_mode")
    fi

    conda run -n rlvla_env python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
        --alg-name "$alg" \
        --env-id "$ENV_ID" \
        --vla-path "$VLA_PATH" \
        --vla-load-path '' \
        --seed "$seed" \
        --num-envs "$NUM_ENVS" \
        --episode-len "$EPISODE_LEN" \
        --steps-max "$STEPS_MAX" \
        --buffer-minibatch 2 \
        --alg-gradient-accum 1 \
        --alg-ppo-epoch 1 \
        --name "$name" \
        "${extra[@]}"
}

for seed in $SEEDS; do
    run_arm "$seed" ppo none
    run_arm "$seed" sdpo correct
    run_arm "$seed" sdpo shuffled
    run_arm "$seed" sdpo none
done
