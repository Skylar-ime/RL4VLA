#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${MS_ASSET_DIR:=/home/skylar/@lab/hf-cache/maniskill}"
: "${VLA_PATH:=/home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03}"
: "${RUN_ROOT:=experiments/sdpo-v1}"
: "${SEEDS:=0 1}"
: "${STEPS_MAX:=2400}"
: "${NUM_ENVS:=2}"
: "${EPISODE_LEN:=80}"
: "${WANDB_MODE:=online}"

export MS_ASSET_DIR WANDB_MODE
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"

if (( NUM_ENVS < 2 || EPISODE_LEN < 1 || STEPS_MAX % (NUM_ENVS * EPISODE_LEN) != 0 )); then
    echo "Invalid NUM_ENVS, EPISODE_LEN, or STEPS_MAX" >&2
    exit 2
fi

run_arm() {
    local seed="$1" alg="$2" feedback_mode="$3" feedback_version="$4"
    local name="v1-${alg}-${feedback_mode}-${feedback_version}-seed${seed}-${STEPS_MAX}steps"
    local run_dir="$RUN_ROOT/$name"
    local last_episode=$((STEPS_MAX / (NUM_ENVS * EPISODE_LEN) - 1))
    local final_checkpoint
    printf -v final_checkpoint 'steps_%04d' "$last_episode"
    if [[ -n "$(find "$run_dir/wandb" -path "*/glob/$final_checkpoint/sdpo_protocol.json" -print -quit 2>/dev/null)" ]]; then
        echo "Already complete: $name"
        return
    fi
    if [[ "$alg" == "ppo" && -n "$(find "$run_dir/wandb" -path "*/glob/$final_checkpoint/dataset_statistics.json" -print -quit 2>/dev/null)" ]]; then
        echo "Already complete: $name"
        return
    fi
    mkdir -p "$run_dir"
    WANDB_DIR="$run_dir" conda run -n rlvla_env python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
        --alg-name "$alg" \
        --sdpo-feedback-mode "$feedback_mode" \
        --sdpo-feedback-version "$feedback_version" \
        --env-id PutOnPlateInScene25Main-v3 \
        --vla-path "$VLA_PATH" \
        --vla-load-path '' \
        --seed "$seed" \
        --num-envs "$NUM_ENVS" \
        --episode-len "$EPISODE_LEN" \
        --steps-max "$STEPS_MAX" \
        --interval-eval 15 \
        --interval-save 5 \
        --eval-episodes 2 \
        --buffer-inferbatch 2 \
        --buffer-minibatch 2 \
        --alg-gradient-accum 1 \
        --alg-ppo-epoch 1 \
        --name "$name" \
        --no-save-render
}

for seed in $SEEDS; do
    run_arm "$seed" ppo none relspatial-v2
    run_arm "$seed" sdpo correct relspatial-v2
    run_arm "$seed" sdpo correct relspatial-action-v1
    run_arm "$seed" sdpo shuffled relspatial-action-v1
done
