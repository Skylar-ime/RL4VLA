#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${TRAIN_JOB_ID:?Set TRAIN_JOB_ID to the robot Slurm training job ID}"
: "${ROBOT_HOST:=robot}"
: "${ROBOT_REPO:=/public/home/s2250018/project/RL4VLA}"
: "${RUN_ROOT:=experiments/sdpo-v1}"
: "${EVAL_ROOT:=$RUN_ROOT/evaluation}"
: "${MS_ASSET_DIR:=/home/skylar/@lab/hf-cache/maniskill}"
: "${VLA_PATH:=/home/skylar/.cache/huggingface/hub/models--gen-robot--openvla-7b-rlvla-warmup/snapshots/735fd5d1f4472c4c9f3f998f0e89aa641af5ad03}"
: "${EVAL_NUM_ENVS:=6}"

if (( EVAL_NUM_ENVS < 1 || 30 % EVAL_NUM_ENVS != 0 )); then
    echo "EVAL_NUM_ENVS must divide 30" >&2
    exit 2
fi
expected_batches=$((30 / EVAL_NUM_ENVS))

export MS_ASSET_DIR WANDB_MODE=offline
export PYTHONPATH="SimplerEnv:ManiSkill:openvla${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$EVAL_ROOT"

for seed in 0 1; do
    for arm in \
        'ppo none relspatial-v2' \
        'sdpo correct relspatial-v2' \
        'sdpo correct relspatial-action-v1' \
        'sdpo shuffled relspatial-action-v1'; do
        read -r algorithm mode version <<< "$arm"
        name="v1-${algorithm}-${mode}-${version}-seed${seed}-2400steps"
        marker=dataset_statistics.json
        [[ "$algorithm" == sdpo ]] && marker=sdpo_protocol.json

        for episode in 0004 0009 0014; do
            steps=$(( (10#$episode + 1) * 160 ))
            label="${name}-steps${steps}"
            eval_dir="$EVAL_ROOT/$label"
            completed=$(find "$eval_dir/wandb" -name eval_only.jsonl -type f -print -quit 2>/dev/null || true)
            if [[ -n "$completed" && $(wc -l < "$completed") -eq "$expected_batches" ]]; then
                echo "Already evaluated: $label"
                continue
            fi

            remote_run="$ROBOT_REPO/$RUN_ROOT/$name/wandb"
            while :; do
                remote_marker=$(ssh "$ROBOT_HOST" "find '$remote_run' -path '*/glob/steps_$episode/$marker' -print -quit 2>/dev/null" || true)
                [[ -n "$remote_marker" ]] && break
                state=$(ssh "$ROBOT_HOST" "/opt/gridview/slurm/bin/squeue -j '$TRAIN_JOB_ID' -h -o %T")
                if [[ -z "$state" ]]; then
                    echo "Training job ended before $label checkpoint appeared" >&2
                    exit 1
                fi
                sleep 30
            done

            checkpoint_dir=$(dirname "$remote_marker")
            copy_dir=$(mktemp -d "$EVAL_ROOT/.checkpoint.XXXXXX")
            trap 'find "$copy_dir" -depth -delete' EXIT
            mkdir -p "$copy_dir/checkpoint" "$eval_dir"
            rsync -a --partial "$ROBOT_HOST:$checkpoint_dir/" "$copy_dir/checkpoint/"
            test -s "$copy_dir/checkpoint/adapter_model.safetensors"
            test -s "$copy_dir/checkpoint/training_state.pt"
            test -s "$copy_dir/checkpoint/$marker"

            WANDB_DIR="$eval_dir" conda run --no-capture-output -n rlvla_env python -u \
                SimplerEnv/simpler_env/train_ms3_ppo.py \
                --alg-name ppo \
                --env-id PutOnPlateInScene25Main-v3 \
                --vla-path "$VLA_PATH" \
                --vla-load-path "$copy_dir/checkpoint" \
                --seed 0 \
                --num-envs "$EVAL_NUM_ENVS" \
                --episode-len 80 \
                --eval-only \
                --eval-episodes 30 \
                --eval-obj-set test \
                --name "$label" \
                --no-save-render >"$eval_dir/eval.log" 2>&1

            completed=$(find "$eval_dir/wandb" -name eval_only.jsonl -type f -print -quit)
            if [[ -z "$completed" || $(wc -l < "$completed") -ne "$expected_batches" ]]; then
                echo "Incomplete 30-episode evaluation: $label" >&2
                exit 1
            fi
            echo "Evaluated: $label"
            find "$copy_dir" -depth -delete
            trap - EXIT
        done
    done
done
