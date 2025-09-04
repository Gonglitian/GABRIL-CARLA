#!/usr/bin/env bash
set -euo pipefail

# DDP launcher for Behavior Cloning (BC)
# 单/多实验一体化脚本
# 用法示例：
#   bash vlm_gaze/train/run_ddp_train_bc.bash
#   NPROC=4 MASTER_PORT=29512 bash vlm_gaze/train/run_ddp_train_bc.bash data.batch_size=128 optimizer.lr=3e-4
#   MULTI_RUN=1 bash vlm_gaze/train/run_ddp_train_bc.bash  # 运行内置多组组合
#   MULTI_RUN=1 METHOD_PAIRS="None:GMD,Reg:GMD,ViSaRL:None" bash vlm_gaze/train/run_ddp_train_bc.bash  # 覆盖组合

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# GPU selection and comm envs
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Torchrun params
NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-29502}
export MASTER_ADDR=127.0.0.1

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
# Reduce C++ backtrace symbolization spam and lower C++ log level
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-0}
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
# Prefer loopback IPv4 to avoid IPv6 localhost issues
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
# Additional network settings for stability
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
## Quiet down NCCL logging (INFO -> WARN)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,ENV}
# Remove deprecated var if present to silence warnings
unset NCCL_ASYNC_ERROR_HANDLING || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Silence Python warnings unless explicitly overridden
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}



# 如果 NPROC 大于可见 GPU 数量，自动下调，避免所有进程落到同一张卡
IFS=',' read -r -a __gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
__num_visible_gpus=${#__gpu_arr[@]}
if [[ ${NPROC} -gt ${__num_visible_gpus} ]]; then
  echo "[WARN] NPROC(${NPROC}) > visible GPUs(${__num_visible_gpus}); set NPROC=${__num_visible_gpus}"
  NPROC=${__num_visible_gpus}
fi

echo "Launching DDP BC training on GPUs: ${CUDA_VISIBLE_DEVICES} (nproc_per_node=${NPROC})"
echo "Master address: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Environment setup complete, starting training..."

# 定义单次运行函数
run_one() {
  local extra_overrides=("$@")
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    vlm_gaze/train/train_bc.py \
    --config-name=train_bc_confounded \
    "${extra_overrides[@]}"
}

# 多实验组合（可通过环境变量 METHOD_PAIRS 覆盖，逗号分隔；默认内置组合）
__DEFAULT_METHOD_PAIRS=(
  "None:GMD" 
  "ViSaRL:None" 
  "GRIL:None" 
  "None:None" 
  "AGIL:None"
  "Reg:GMD"
  "Reg:None"
)

if [[ "${MULTI_RUN:-1}" == "1" ]]; then
  # 如果用户提供 METHOD_PAIRS 以逗号分隔，则解析为数组
  METHOD_PAIRS_ARR=()
  if [[ -n "${METHOD_PAIRS:-}" ]]; then
    IFS=',' read -r -a METHOD_PAIRS_ARR <<< "${METHOD_PAIRS}"
  else
    METHOD_PAIRS_ARR=("${__DEFAULT_METHOD_PAIRS[@]}")
  fi

  echo "=== Multi-run DDP BC Training ==="
  echo "Will run ${#METHOD_PAIRS_ARR[@]} method pairs"
  for pair in "${METHOD_PAIRS_ARR[@]}"; do
    IFS=':' read -r gaze_method dropout_method <<< "${pair}"
    echo "  - gaze.method=${gaze_method}, dropout.method=${dropout_method}"
  done
  echo ""

  __run_counter=0
  for pair in "${METHOD_PAIRS_ARR[@]}"; do
    IFS=':' read -r gaze_method dropout_method <<< "${pair}"
    __run_counter=$((__run_counter + 1))
    echo "=== Run ${__run_counter}/${#METHOD_PAIRS_ARR[@]}: gaze.method=${gaze_method}, dropout.method=${dropout_method} ==="
    if ! run_one gaze.method="${gaze_method}" dropout.method="${dropout_method}" "$@"; then
      echo "ERROR: Run ${__run_counter} failed (gaze=${gaze_method}, dropout=${dropout_method})"
      echo "Continuing with next experiment..."
      sleep 5
      continue
    fi
    echo "=== Run ${__run_counter} completed successfully ==="
    echo ""
    sleep 5
  done
  echo "=== All experiments completed! ==="
else
  # 单次运行：支持通过外部传入 Hydral overrides（例如 gaze.method/ dropout.method 等）
  run_one "$@"
fi
