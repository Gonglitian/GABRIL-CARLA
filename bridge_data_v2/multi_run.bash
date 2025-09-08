#!/usr/bin/env bash

set -euo pipefail

export DATA_ROOT=/data3/vla-reasoning/dataset/bdv2_tfrecord
export SAVE_DIR=/data3/vla-reasoning/exp/bdv2_runs

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT path does not exist: $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$SAVE_DIR"

ALGOS=(
  "bc"
  "gc_bc"
  "gc_ddpm_bc"
)

TASKS=(
  # "open_microwave"
  # "put_in_pot_lid"
  "remove_pot_lid"
)

EXTRA_ARGS=${EXTRA_ARGS:-}

echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] SAVE_DIR=$SAVE_DIR"

# Tunables with environment overrides
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1000}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
NUM_STEPS=${NUM_STEPS:-60000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
SAVE_INTERVAL=${SAVE_INTERVAL:-15000}
LOG_INTERVAL=${LOG_INTERVAL:-30}
WARMUP_STEPS=${WARMUP_STEPS:-3000}
# Optional: cap number of eval batches per eval
EVAL_BATCHES=${EVAL_BATCHES:-}

# Decay schedule controls (per algorithm)
DECAY_STEPS_BC=${DECAY_STEPS_BC:-1000000}
DECAY_STEPS_GC_BC=${DECAY_STEPS_GC_BC:-1000000}
ACTOR_DECAY_STEPS_GC_DDPM_BC=${ACTOR_DECAY_STEPS_GC_DDPM_BC:-1000000}

echo "[INFO] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE VAL_BATCH_SIZE=${VAL_BATCH_SIZE}"
echo "[INFO] NUM_STEPS=$NUM_STEPS EVAL_INTERVAL=$EVAL_INTERVAL SAVE_INTERVAL=$SAVE_INTERVAL LOG_INTERVAL=$LOG_INTERVAL"
if [[ -n "${EVAL_BATCHES}" ]]; then
  echo "[INFO] EVAL_BATCHES=$EVAL_BATCHES"
fi

for algo in "${ALGOS[@]}"; do
  for task in "${TASKS[@]}"; do
    run_name="${algo}_${task}_bdv2"

    # Build per‑algo decay flag before echo/use
    DECAY_FLAG=""
    case "$algo" in
      bc)
        DECAY_FLAG="--config.agent_kwargs.decay_steps=${DECAY_STEPS_BC}"
        ;;
      gc_bc)
        DECAY_FLAG="--config.agent_kwargs.decay_steps=${DECAY_STEPS_GC_BC}"
        ;;
      gc_ddpm_bc)
        DECAY_FLAG="--config.agent_kwargs.actor_decay_steps=${ACTOR_DECAY_STEPS_GC_DDPM_BC}"
        ;;
      *)
        DECAY_FLAG=""
        ;;
    esac

    # Optional eval batches flag
    EVAL_BATCHES_FLAG=""
    if [[ -n "${EVAL_BATCHES}" ]]; then
      EVAL_BATCHES_FLAG="--config.eval_batches=${EVAL_BATCHES}"
    fi

    echo "[RUN ] algo=${algo} task=${task} name=${run_name} ${DECAY_FLAG} ${EXTRA_ARGS}"

    python experiments/train.py \
      --name "${run_name}" \
      --config "experiments/configs/train_config.py:${algo}" \
      --bridgedata_config "experiments/configs/data_config.py:${task}" \
      --config.data_path "$DATA_ROOT" \
      --config.save_dir "$SAVE_DIR" \
      --config.batch_size ${TRAIN_BATCH_SIZE} \
      --config.val_batch_size ${VAL_BATCH_SIZE} \
      --config.num_steps=${NUM_STEPS} \
      --config.eval_interval=${EVAL_INTERVAL} \
      --config.save_interval=${SAVE_INTERVAL} \
      --config.log_interval=${LOG_INTERVAL} \
      --config.agent_kwargs.warmup_steps=${WARMUP_STEPS} \
      ${DECAY_FLAG} \
      ${EVAL_BATCHES_FLAG} \
      ${EXTRA_ARGS}

    echo "[DONE] ${run_name} Finished"
  done
done

echo "[ALL DONE] All experiments finished"
