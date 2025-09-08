## 🚀 Quick Overview

本项目使用 JAX + Flax 进行训练，数据以 BridgeData TFRecord 组织。近期我们修复了若干依赖/运行时问题，并完善了多实验启动流程。以下文档汇总环境搭建、数据处理与训练步骤，并标注关键变更与排障建议。

---

## 🔧 环境准备（JAX/TF/显卡）

建议在独立虚拟环境中安装依赖。以下版本已验证互相兼容：

```bash
# 1) 安装 JAX (CUDA 12 pip wheels)
pip install --upgrade "jax[cuda12_pip]==0.4.13" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 2) 科学计算基础
pip install "scipy>=1.10,<1.12"

# 3) 解决 tensorstore ↔ ml-dtypes 兼容问题（需要 float8_e3m4）
pip install -U "ml-dtypes>=0.5.0"   # 我们使用到的是 0.5.3

# 4) 解决 distrax/TFP 与 TF 2.15 的签名冲突
pip install -U "tensorflow-probability==0.22.1"

# 5) 提供与 jaxlib 对齐的 NVIDIA 运行时库（CUDA12 系）
#    这些库来自 pip，无需系统 CUDA Toolkit 即可运行
pip install -U \
  "nvidia-cuda-runtime-cu12==12.2.*" \
  "nvidia-cublas-cu12==12.2.*" \
  "nvidia-cudnn-cu12==8.9.*" \
  "nvidia-cusolver-cu12==11.5.*" \
  "nvidia-cusparse-cu12==12.1.*" \
  "nvidia-nvjitlink-cu12==12.9.*"
```

提示：我们默认让 TensorFlow 使用 CPU，JAX 使用 GPU，以避免 TF/GPU 与 JAX/CUDA 库的冲突（训练脚本已内置设置）。

---

## 🧩 代码改动与配置变更

本次为稳定训练，做了以下关键改动：

- 文件 `jaxrl_m/common/encoding.py`
  - 修复观测编码与本体感觉（proprio）拼接时的维度不一致问题。
  - 当 `observations["proprio"]` 形如 `(B, 1, P)` 或 `(B, T, P)` 时，会自动展平成 `(B, T*P)` 后再与编码 `(B, F)` 拼接，修复错误：
    - Cannot concatenate arrays with different numbers of dimensions: got (B, F), (B, 1, P)

- 文件 `experiments/multi_train.py`
  - 新增自动发现并注入 pip 提供的 NVIDIA 运行库目录到 `LD_LIBRARY_PATH`（cuDNN/cuBLAS/NVJitLink 等），避免系统 CUDA Toolkit 版本不匹配。
  - 支持从 YAML 传入 `XLA_FLAGS`（修复某些机器的 nvlink 报错），`JAX_PLATFORM_NAME`（可强制 CPU/GPU）。

- 文件 `experiments/configs/multi_train.yaml`
  - 新增全局字段：
    - `xla_flags: "--xla_gpu_force_compilation_parallelism=1"`（⚠️ 用于规避 nvlink linking API 并行 bug）。
    - `use_pip_cuda_libs: true`（优先使用 pip 提供的 CUDA 运行库）。
    - 保留 `cuda_visible_devices: "0"`，默认使用单卡，避免 batch size 整除问题。

---

## 🧱 数据处理（TASL/LIRA 两套路径）

TASL 机器示例：

```bash
python data_processing/bridgedata_raw_to_numpy.py \
  --input_path /data3/vla-reasoning/dataset/bdv2 \
  --output_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --depth 2 --num_workers 8 --train_proportion 0.9 --im_size 256

python data_processing/bridgedata_numpy_to_tfrecord.py \
  --input_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --output_path /data3/vla-reasoning/dataset/bdv2_tfrecord \
  --depth 2 --num_workers 1 --im_size 256
```

LIRA 机器示例：

```bash
python data_processing/bridgedata_raw_to_numpy.py \
  --input_path /scr/litian/dataset/bdv2 \
  --output_path /scr/litian/dataset/bdv2_numpy \
  --depth 2 --num_workers 8 --train_proportion 0.99 --im_size 256

python data_processing/bridgedata_numpy_to_tfrecord.py \
  --input_path /scr/litian/dataset/bdv2_numpy \
  --output_path /scr/litian/dataset/bdv2_tfrecord \
  --depth 2 --num_workers 1
```

---

## 🧪 训练（单/多实验）

单次多实验启动：

```bash
python experiments/multi_train.py \
  --config /home/litian/proj/GABRIL-CARLA/bridge_data_v2/experiments/configs/multi_train.yaml
```

后台运行（nohup）：

```bash
nohup python experiments/multi_train.py \
  --config /home/litian/proj/GABRIL-CARLA/bridge_data_v2/experiments/configs/multi_train.yaml \
  > log.txt 2>&1 &
```

YAML 关键字段解释（`experiments/configs/multi_train.yaml` → `global:`）：

- `data_root` / `save_dir`：数据与保存路径。
- `train_batch_size` / `val_batch_size`：训练/验证 batch；注意需满足“能整除设备数”的约束。
- `cuda_visible_devices`：默认 `"0"` 保持单卡，避免 batch 整除问题。多卡时请把 `train_batch_size` 设为设备数的倍数。
- `xla_flags`：默认 `--xla_gpu_force_compilation_parallelism=1` 以稳定 CUDA linking。
- `use_pip_cuda_libs`：默认开启；自动把 pip 的 CUDA 运行库注入 `LD_LIBRARY_PATH`。
- 可选 `jax_platform_name`：强制 `cpu` 或 `gpu`（需时可以临时切到 CPU 验证）。

---

## 🩺 常见问题排查（Troubleshooting）

- ImportError: tensorstore 初始化失败，或 `ml_dtypes` 提示缺少 `float8_e3m4`
  - 运行：`pip install -U ml-dtypes>=0.5.0`

- TFP 报 `Arg specs do not match`（与 TF 的 `tf.ones_like` 签名冲突）
  - 运行：`pip install -U tensorflow-probability==0.22.1`（匹配 TF 2.15）

- `DNN library initialization failed`（cuDNN 初始化失败）
  - 确保安装并注入 pip 的 CUDA 运行库（见上文第 5 步）。
  - 若仍有问题，临时切 CPU：在 YAML 里设置 `jax_platform_name: "cpu"` 或 `cuda_visible_devices: ""`。

- `nvlink fatal : Input file ... newer than toolkit (129 vs 124)`
  - 安装 `nvidia-nvjitlink-cu12==12.9.*` 并设置 `xla_flags: --xla_gpu_force_compilation_parallelism=1`。

- `Cannot concatenate arrays ... (B, F) vs (B, 1, P)`
  - 已在 `jaxrl_m/common/encoding.py` 统一展平 proprio；请更新代码后重试。

---

## 🧭 小贴士

- 我们在训练脚本里默认让 TensorFlow 只用 CPU，以避免与 JAX/CUDA 冲突；JAX 仍使用 GPU。
- 多卡训练时请确认 `train_batch_size % local_device_count == 0`。
- Weights & Biases 会自动初始化并记录运行；如果需要离线可用 `wandb offline`。

祝训练顺利！🎯
