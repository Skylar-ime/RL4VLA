#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
: "${MS_ASSET_DIR:=/home/skylar/@lab/hf-cache/maniskill}"
: "${VLA_PATH:=/home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03}"
: "${EVAL_ROOT:=experiments/sdpo-v2-10000/evaluation}"
: "${EVAL_LABEL:?Set unique EVAL_LABEL}"
: "${EVAL_EPISODES:=60}"
: "${EVAL_EPISODE_START:=0}"
: "${EVAL_NUM_ENVS:=6}"
: "${VLA_LOAD_PATH:=}"
: "${WANDB_MODE:=offline}"

if (( EVAL_EPISODES < 1 || EVAL_EPISODES % EVAL_NUM_ENVS != 0 )); then
    echo "EVAL_EPISODES must be a positive multiple of EVAL_NUM_ENVS" >&2
    exit 2
fi
run_dir="$EVAL_ROOT/$EVAL_LABEL"
if [[ -e "$run_dir" ]]; then
    echo "Evaluation directory already exists: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
export MS_ASSET_DIR WANDB_MODE PYTHONNOUSERSITE=1
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"

WANDB_DIR="$run_dir" conda run --no-capture-output -n rlvla_env python -u \
    SimplerEnv/simpler_env/train_ms3_ppo.py \
    --alg-name ppo --env-id PutOnPlateInScene25Main-v3 \
    --vla-path "$VLA_PATH" --vla-load-path "$VLA_LOAD_PATH" \
    --seed 0 --num-envs "$EVAL_NUM_ENVS" --episode-len 80 \
    --eval-only --eval-episodes "$EVAL_EPISODES" \
    --eval-episode-start "$EVAL_EPISODE_START" --eval-obj-set test \
    --name "$EVAL_LABEL" --no-save-render > "$run_dir/eval.log" 2>&1

result=$(find "$run_dir" -name eval_only.jsonl -type f -print -quit)
if [[ -z "$result" || $(wc -l < "$result") -ne $((EVAL_EPISODES / EVAL_NUM_ENVS)) ]]; then
    echo "Incomplete evaluation: $EVAL_LABEL" >&2
    exit 1
fi
echo "Evaluation complete: $EVAL_LABEL ($result)"
