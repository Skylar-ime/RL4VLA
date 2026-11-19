#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${MS_ASSET_DIR:=/home/skylar/@lab/hf-cache/maniskill}"
: "${VLA_PATH:=/home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03}"
: "${RUN_ROOT:=experiments/sdpo-outcome-v2}"
: "${SEEDS:=0 1}"
: "${WANDB_MODE:=offline}"

export MS_ASSET_DIR WANDB_MODE PYTHONNOUSERSITE=1
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"

for seed in $SEEDS; do
    name="outcome-v2-sdpo-correct-seed${seed}-2400steps"
    run_dir="$RUN_ROOT/$name"
    if [[ -n "$(find "$run_dir/wandb" -path '*/glob/steps_0014/sdpo_protocol.json' -print -quit 2>/dev/null)" ]]; then
        echo "Already complete: $name"
        continue
    fi
    if [[ -e "$run_dir/wandb" ]]; then
        echo "Incomplete run directory exists: $run_dir" >&2
        exit 2
    fi
    mkdir -p "$run_dir"

    WANDB_DIR="$run_dir" conda run --no-capture-output -n rlvla_env python -u \
        SimplerEnv/simpler_env/train_ms3_ppo.py \
        --alg-name sdpo \
        --sdpo-feedback-mode correct \
        --sdpo-feedback-version relspatial-outcome-v2 \
        --env-id PutOnPlateInScene25Main-v3 \
        --vla-path "$VLA_PATH" \
        --vla-load-path '' \
        --seed "$seed" \
        --num-envs 2 \
        --episode-len 80 \
        --steps-max 2400 \
        --interval-eval 15 \
        --interval-save 5 \
        --eval-episodes 2 \
        --buffer-inferbatch 2 \
        --buffer-minibatch 2 \
        --alg-gradient-accum 1 \
        --alg-ppo-epoch 1 \
        --name "$name" \
        --no-save-render
done
