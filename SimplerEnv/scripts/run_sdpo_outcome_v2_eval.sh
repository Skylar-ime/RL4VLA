#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${MS_ASSET_DIR:=/home/skylar/@lab/hf-cache/maniskill}"
: "${VLA_PATH:=/home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03}"
: "${RUN_ROOT:=experiments/sdpo-outcome-v2}"
: "${SEEDS:=0 1}"

export MS_ASSET_DIR WANDB_MODE=offline PYTHONNOUSERSITE=1
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"

for seed in $SEEDS; do
    name="outcome-v2-sdpo-correct-seed${seed}-2400steps"
    for episode in 0004 0009 0014; do
        steps=$(( (10#$episode + 1) * 160 ))
        label="${name}-steps${steps}"
        eval_dir="$RUN_ROOT/evaluation/$label"
        complete=$(find "$eval_dir/wandb" -name eval_only.jsonl -type f -print -quit 2>/dev/null || true)
        if [[ -n "$complete" && $(wc -l < "$complete") -eq 5 ]]; then
            echo "Already evaluated: $label"
            continue
        fi
        if [[ -e "$eval_dir/wandb" ]]; then
            echo "Incomplete evaluation directory exists: $eval_dir" >&2
            exit 2
        fi

        checkpoint=$(find "$RUN_ROOT/$name/wandb" -path "*/glob/steps_$episode/sdpo_protocol.json" -print -quit 2>/dev/null || true)
        if [[ -z "$checkpoint" ]]; then
            echo "Missing checkpoint: $label" >&2
            exit 2
        fi
        checkpoint_dir=$(dirname "$checkpoint")
        mkdir -p "$eval_dir"
        WANDB_DIR="$eval_dir" conda run --no-capture-output -n rlvla_env python -u \
            SimplerEnv/simpler_env/train_ms3_ppo.py \
            --alg-name ppo \
            --env-id PutOnPlateInScene25Main-v3 \
            --vla-path "$VLA_PATH" \
            --vla-load-path "$checkpoint_dir" \
            --seed 0 \
            --num-envs 6 \
            --episode-len 80 \
            --eval-only \
            --eval-episodes 30 \
            --eval-obj-set test \
            --name "$label" \
            --no-save-render > "$eval_dir/eval.log" 2>&1

        complete=$(find "$eval_dir/wandb" -name eval_only.jsonl -type f -print -quit)
        if [[ -z "$complete" || $(wc -l < "$complete") -ne 5 ]]; then
            echo "Incomplete 30-episode evaluation: $label" >&2
            exit 1
        fi
        echo "Evaluated: $label"
    done
done
