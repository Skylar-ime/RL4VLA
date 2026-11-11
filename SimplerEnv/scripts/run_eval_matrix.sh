#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${MS_ASSET_DIR:=/home/skylar/@lab/hf-cache/maniskill}"
: "${VLA_PATH:=/home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03}"
: "${ENV_IDS:=PutOnPlateInScene25Main-v3}"
: "${NUM_ENVS:=2}"
: "${EPISODE_LEN:=80}"
: "${EVAL_EPISODES:=6}"
: "${EVAL_OBJ_SET:=test}"
: "${SEED:=0}"
: "${WANDB_MODE:=online}"
: "${CHECKPOINT_MANIFEST:=docs/sdpo_eval_matrix.tsv}"

export MS_ASSET_DIR WANDB_MODE
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"

if (( NUM_ENVS < 1 || EVAL_EPISODES < 1 || EVAL_EPISODES % NUM_ENVS != 0 )); then
    echo "EVAL_EPISODES must be a positive multiple of NUM_ENVS" >&2
    exit 2
fi

if [[ "$CHECKPOINT_MANIFEST" != "-" && ! -f "$CHECKPOINT_MANIFEST" ]]; then
    echo "checkpoint manifest not found: $CHECKPOINT_MANIFEST" >&2
    exit 2
fi

manifest_input="$CHECKPOINT_MANIFEST"
[[ "$manifest_input" == "-" ]] && manifest_input=/dev/stdin

while IFS=$'\t' read -r label checkpoint; do
    [[ -z "$label" || "$label" == \#* ]] && continue
    if [[ ! -d "$checkpoint" ]]; then
        echo "checkpoint not found for $label: $checkpoint" >&2
        exit 2
    fi
    for env_id in $ENV_IDS; do
        conda run -n rlvla_env python -u SimplerEnv/simpler_env/train_ms3_ppo.py \
            --alg-name ppo \
            --env-id "$env_id" \
            --vla-path "$VLA_PATH" \
            --vla-load-path "$checkpoint" \
            --seed "$SEED" \
            --num-envs "$NUM_ENVS" \
            --episode-len "$EPISODE_LEN" \
            --eval-only \
            --eval-episodes "$EVAL_EPISODES" \
            --eval-obj-set "$EVAL_OBJ_SET" \
            --name "${label}-${env_id}-${EVAL_OBJ_SET}"
    done
done < "$manifest_input"
