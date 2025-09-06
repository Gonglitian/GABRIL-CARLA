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
  # "gc_bc"
  # "gc_ddpm_bc"
)

TASKS=(
  # "open_microwave"
  "put_in_pot_lid"
  # "remove_pot_lid"
)

EXTRA_ARGS=${EXTRA_ARGS:-}

echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] SAVE_DIR=$SAVE_DIR"

for algo in "${ALGOS[@]}"; do
  for task in "${TASKS[@]}"; do
    run_name="${algo}_${task}_bdv2"
    echo "[RUN ] algo=${algo} task=${task} name=${run_name} ${DECAY_FLAG} ${EXTRA_ARGS}"

    DECAY_FLAG=""
    case "$algo" in
      bc|gc_bc)
        DECAY_FLAG="--config.agent_kwargs.decay_steps=200000"
        ;;
      gc_ddpm_bc)
        DECAY_FLAG="--config.agent_kwargs.actor_decay_steps=200000"
        ;;
      *)
        DECAY_FLAG=""
        ;;
    esac

    python experiments/train.py \
      --name "${run_name}" \
      --config "experiments/configs/train_config.py:${algo}" \
      --bridgedata_config "experiments/configs/data_config.py:${task}" \
      --config.data_path "$DATA_ROOT" \
      --config.save_dir "$SAVE_DIR" \
      --config.batch_size 64 \
      --config.num_steps=100000 \
      --config.save_interval=10000 \
      ${DECAY_FLAG} \
      ${EXTRA_ARGS}

    echo "[DONE] ${run_name} Finished"
  done
done

echo "[ALL DONE] All experiments finished"


