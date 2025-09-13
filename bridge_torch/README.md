# Bridge Torch (PyTorch Training Port)

本目录提供使用 PyTorch 训练 BridgeData 的 BC / GC-BC / GC-DDPM-BC 算法的完整流程。本文档包含：环境配置、数据准备（从 raw → numpy）、配置与启动训练、分布式与日志、常见问题等。

注意：本实现仅支持 numpy 数据管线（无需 TensorFlow）：
- numpy 管线：读取 `<task>/{train,val}/out.npy`（由 raw 数据转换而来），无需 TF 依赖，效率高、部署简单。

关键入口：
- `bridge_torch/experiments/train_hydra.py`: 训练入口（Hydra 配置系统）。
- `bridge_torch/conf/`: Hydra 配置目录（algo、bridgedata 分组）。

---

## 1. 环境准备

推荐使用 Conda 独立环境（Python 3.10 测试通过）：

```bash
conda create -n bridge_torch python=3.10 -y
conda activate bridge_torch

# 安装 PyTorch (按你的 CUDA 版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖（仓库根目录）
pip install -r requirements.txt

# 可选：如果需要使用 TFRecord 管线（非必须）
pip install tensorflow-cpu==2.13.*

# 日志与可视化
pip install wandb
wandb login  # 登录你的 W&B 账号
```

验证环境：

```bash
python -c "import torch; print('Torch:', torch.__version__)"
```

---

## 2. 数据准备（raw → numpy）

BridgeData raw → numpy 的转换脚本位于：`bridge_data_v2/data_processing/bridgedata_raw_to_numpy.py`。

原始目录结构（示例）：

```
bridgedata_raw/
  rss/toykitchen2/set_table/00/2022-01-01_00-00-00/
    collection_metadata.json
    config.json
    raw/traj_group*/traj*/
      obs_dict.pkl
      policy_out.pkl
      images0/im_0.jpg ...
```

运行转换（将 raw 转到 numpy，输出为 `<task>/{train,val}/out.npy`）：

```bash
# 示例：处理所有任务（depth=5），输出到 /path/to/bdv2_numpy
python bridge_data_v2/data_processing/bridgedata_raw_to_numpy.py \
  --input_path /path/to/bridgedata_raw \
  --output_path /path/to/bdv2_numpy \
  --depth 2 \
  --train_proportion 0.99 \
  --num_workers 8 \
  --im_size 256 \
  --saliency

# 转换完成后：
# /path/to/bdv2_numpy/<task>/train/out.npy
# /path/to/bdv2_numpy/<task>/val/out.npy
```

已有数据的常见路径示例（本机）：
- numpy: `/scr/litian/dataset/bdv2_numpy/lift_carrot_100/{train,val}/out.npy`

训练脚本依赖 numpy 管线（必须存在 `out.npy`）。

### 2.1 Torch 如何读取 numpy 数据用于训练

本仓库在 `bridge_torch/data/bridge_numpy.py` 中实现了纯 PyTorch 的数据读取与打包逻辑：

- 核心类：`BridgeNumpyDataset`
  - 输入：若干个 `out.npy` 路径列表（通常来自多个任务的 `train/out.npy` 或 `val/out.npy`）。
  - 文件内容：`out.npy` 为一个 Python 对象数组（`allow_pickle=True`），其中每个元素表示一条轨迹（trajectory）字典：
    - `observations`: 每步观测的字典，至少包含 `images0`（HWC uint8）与可选 `state`（proprio）。
    - `next_observations`: 同上，表示下一步观测。
    - `actions`: 每步动作（长度与 `observations` 对齐）。
  - 采样：每个 batch 会随机抽取若干条轨迹 + 每条轨迹内的一个时间步（transition），从而构建一个大小为 `batch_size` 的独立样本集合。

- 图像转换：
  - 将 `images0`（HWC, uint8）转换为 `CHW, float32 ∈ [0,1]`，并堆叠为 `torch.Tensor` 形状 `(B, C, H, W)`。
  - 同时对 goal 图像执行相同转换（见下文“目标重标”）。

- proprio（本体状态）：
  - 若 `observations` 中存在 `state`，会被当作 proprio 特征；若为 `(T,P)` 或更高维，将展平成二维 `(B, *)` 并与编码器特征拼接。

- 目标重标（goals）：
  - 默认策略为“uniform future”：对每个样本，随机从同一轨迹的未来某一步选择 `next_observations.images0` 作为 `goals.image`。
  - 也可根据需要扩展更多策略（与 TF 版保持一致）。

- 动作形状与 DDPM：
  - 若 `actions` 为二维 `(B, A)`，在 GC-DDPM-BC 中会自动视为单步序列 `(B, 1, A)`，匹配 `act_pred_horizon=1` 的默认设定。
  - 若已是三维 `(B, T, A)`，将按序列处理（T 为动作预测序列长度）。

- 迭代器：
  - `BridgeNumpyDataset.iterator()` 持续产出字典 batch：
    - `{"observations": {"image": (B,3*obs_horizon,H,W), "proprio": (可选)}, "goals": {"image": (B,3*obs_horizon,H,W)}, "actions": (B,A) 或 (B,T,A)}`
  - 训练脚本中通过统一接口 `iter_torch_batches(...)` 获取 batch，算法无感知底层数据源。

---

## 3. 训练配置（Hydra）

训练入口：`bridge_torch/experiments/train_hydra.py`

示例命令：

```bash
# 单次训练
python -m bridge_torch.experiments.train_hydra \
  algo=bc bridgedata=lift_carrot_100 \
  data_path=/path/to/bdv2_numpy \
  save_dir=/path/to/runs \
  batch_size=256 num_steps=10000

# 多实验（Hydra Multirun）
python -m bridge_torch.experiments.train_hydra -m \
  algo=bc,gc_bc \
  bridgedata=lift_carrot_100,pull_pot_100 \
  seed=1,2,3 \
  data_path=/path/to/bdv2_numpy save_dir=/path/to/runs

# 覆盖继承的 algo 预设中的键（以 BC 的 saliency 为例）
python -m bridge_torch.experiments.train_hydra \
  algo=bc bridgedata=lift_carrot_100 \
  agent_kwargs.policy_kwargs.saliency.enabled=true \
  agent_kwargs.policy_kwargs.saliency.weight=0.2 \
  encoder_kwargs.arch=resnet50
```

配置位于 `bridge_torch/conf/`：
- `algo/*`: 算法预设（bc/gc_bc/gc_ddpm_bc）
- `bridgedata/*`: 任务/数据集选择（例如 `lift_carrot_100`）

---

## 4. 启动训练与保存

输出目录结构：
- 权重与配置：`<save_dir>/bridgedata_torch/<exp_name_时间戳>/ckpt_*.pt`, `config.json`
- W&B: `project=bridgedata_torch`（可在 `conf/config.yaml` 中设置 `wandb.enabled=false` 关闭）

检查输出：
- 权重与配置：`<save_dir>/bridgedata_torch/<exp_name_时间戳>/ckpt_*.pt`, `config.json`
- W&B: 自动创建 `project=bridgedata_torch` 的 run（可 `wandb offline` 关闭同步）。

---

## 5. 算法说明与实现要点

- 编码器：`resnetv1-34-bridge`（torchvision resnet34），支持 `add_spatial_coordinates`、`avg pool` 等；忽略 JAX `act` 参数。
- 算法：
  - BC（`agents/bc.py`）：高斯策略，支持固定或状态相关方差；使用 AdamW，梯度裁剪；支持 proprio 融合。
  - GC-BC（`agents/gc_bc.py`）：目标条件策略，支持 early concat 或 late add 的特征融合；与 BC 同步优化/日志接口。
  - GC-DDPM-BC（`agents/gc_ddpm_bc.py`）：时间嵌入 + 条件编码 + MLP-ResNet 反向网络；训练为噪声预测 MSE；采样使用 DDPM 逆扩散。动态 MLP/ResNet 在首次前向后完成构建与设备对齐，再初始化目标网络以确保 Polyak 参数维度一致。
- 学习率调度：
  - 线性 warmup（LinearLR，start_factor=1e-6 → 1.0），可选余弦退火（CosineAnnealingLR）。
  - 每步 chainable `scheduler.step()`。

---

## 5.1 图像增强（Augmentation）与时间窗口（Horizons）

- 增强开关与库: 图像增强基于 Albumentations 实现；在训练配置的 `dataset_kwargs` 中设置 `augment: true` 生效；增强参数与顺序通过 `augment_kwargs` 配置，默认与 `bridge_data_v2/experiments/configs/train_config.py` 保持一致。
  - 支持的增强（按 `augment_order` 顺序应用）：
    - `random_resized_crop`: 对应 `A.RandomResizedCrop(height=H,width=W, scale, ratio)`，裁剪后回缩至原始分辨率。
    - `random_brightness`: `A.ColorJitter(brightness=...)`（仅启用亮度）。
    - `random_contrast`: `A.ColorJitter(contrast=...)`（仅启用对比度）。
    - `random_saturation`: `A.ColorJitter(saturation=...)`（仅启用饱和度）。
    - `random_hue`: `A.ColorJitter(hue=...)`（仅启用色相）。
  - 时序一致性：对于 `obs_horizon>1` 的多帧观测，使用 `ReplayCompose` 机制对所有帧复用同一组随机参数；当 `augment_next_obs_goal_differently: false` 时，目标图也复用这组参数以保证空间一致性。

- 时间窗口配置（Horizons）:
  - `obs_horizon` (>=1): 从时间步 `t` 向前收集最近 `obs_horizon` 帧 `[t-obs_horizon+1, ..., t]`，在通道维度拼接为 `(B, 3*obs_horizon, H, W)` 后送入编码器；编码器第一层会在首次前向时自动自适配输入通道数（动态替换 `conv1`）。若算法使用共享目标编码器（如 GC-BC/GC-DDPM-BC 的 `shared_goal_encoder=True`），数据管线会在通道维度上将目标图像重复 `obs_horizon` 次，使得目标与观测输入通道一致。
  - `act_pred_horizon`:
    - 当 >1 时，数据集会返回动作序列 `(B, T, A)`，用于 GC-DDPM-BC；若输入为二维 `(B, A)`，GC-DDPM-BC 会自动视为 `T=1`。
    - 对于 BC / GC-BC，动作必须为单步 `(B, A)`，因此该值应保持为 `1`。

示例（开启增强并指定顺序）：

```python
ConfigDict(dict(
  dataset_kwargs=dict(
    augment=True,
    augment_next_obs_goal_differently=False,
    augment_kwargs=dict(
      random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
      random_brightness=[0.2],
      random_contrast=[0.8, 1.2],
      random_saturation=[0.8, 1.2],
      random_hue=[0.1],
      augment_order=[
        "random_resized_crop",
        "random_brightness",
        "random_contrast",
        "random_saturation",
        "random_hue",
      ],
    ),
    # horizons
    obs_horizon=2,   # 双帧：输入 (B,6,H,W)
    act_pred_horizon=1,
  ),
))
```

注意：
- `obs_horizon` 增大会线性增加显存/算力与 IO；增强（尤其是裁剪）在所有帧上复用同一参数，保证时序一致性。
- 由于编码器 `conv1` 会根据首个 batch 的输入通道数自适配，建议训练/评估阶段保持相同的 `obs_horizon`；若确需切换，新的前向会自动完成一次轻量自适配。

---

## 6. 常见问题（FAQ）

- 仍看到 TensorFlow 的警告？
  - 当前版本不依赖 TensorFlow。如果你的环境中另有 TF 相关输出，可忽略或卸载 TF。
- 维度/设备不匹配报错？
  - 我们已将编码器与动态网络在首次前向后对齐到 `device`，并兼容 `actions` 为 (B,A) 或 (B,T,A)。如仍遇到，请记录异常处张量维度并提 Issue。
- W&B 同步过慢/不想联网？
  - `wandb offline`，或在 YAML `extra_args` 传入 `--debug True`（会以 disabled 模式记录本地文件）。
- 只用单卡/CPU 测试？
  - 将 `global.ddp.enabled: false`；或直接运行单次训练命令（不经 torchrun）。

---

## 7. 参考命令速查

```bash
# 单次训练（BC）
python -m bridge_torch.experiments.train_hydra \
  algo=bc bridgedata=lift_carrot_100 \
  data_path=/scr/litian/dataset/bdv2_numpy \
  save_dir=/scr/litian/torch_runs \
  batch_size=64 num_steps=2000 eval_interval=200 save_interval=500

# Multirun 同时扫多个算法/任务/种子
python -m bridge_torch.experiments.train_hydra -m \
  algo=bc,gc_bc \
  bridgedata=lift_carrot_100,pull_pot_100 \
  seed=1,2,3 \
  data_path=/scr/litian/dataset/bdv2_numpy save_dir=/scr/litian/torch_runs

# DDP（假设 4 卡），与 Hydra 一起使用
torchrun --nproc_per_node=4 -m bridge_torch.experiments.train_hydra \
  algo=gc_ddpm_bc bridgedata=lift_carrot_100 \
  data_path=/scr/litian/dataset/bdv2_numpy save_dir=/scr/litian/torch_runs \
  batch_size=128 ddp.enabled=true
```

---

如需我为你的数据根或 YAML 生成一份专用配置，请告知实际路径与期望 GPU 数，我可以顺手调整并验证一次运行。

---

## 8. Saliency 训练变体（Reg，仅当提供 saliency_map 时可用）

本分支支持“Saliency 辅助正则（Reg）”训练：从编码器最后一层卷积特征图 `z_map (B,C,Hf,Wf)` 生成上采样的注意力掩码（`get_gaze_mask(z_map, beta, (H,W)) → (B,1,H,W)`），与数据集中提供的单帧 `saliency`（同维度 `(B,1,H,W)`）做 MSE 作为 `reg_loss`，并以 `total = base_loss + weight * reg_loss` 优化。

启用步骤概览：
- 数据准备：
  - 运行 saliency pipeline 生成每条轨迹的 `saliency_map.pkl`（形状 `(T,1,480,640)`）。
  - 运行 numpy 转换时带上 `--saliency`，将单帧 saliency（时间步 t）并入样本：
    ```bash
    python bridge_torch/data/bdv2_to_numpy.py \
      --input_path <bdv2_raw_root> \
      --output_path <bdv2_numpy_root> \
      --depth 4 --im_size 128 --saliency | cat
    ```
  - 注意：即使 `obs_horizon>1`，saliency 也不做时间堆叠，仅使用采样时间步 t 的单帧；saliency 不做任何图像增强或几何变换。

- 训练端开关（按算法）：
  - BC / GC-BC：在 `agent_kwargs.policy_kwargs.saliency` 下配置：
    ```yaml
    agent_kwargs:
      policy_kwargs:
        saliency:
          enabled: true   # 开启 Reg 分支
          weight: 0.2     # reg_loss 权重
          beta: 1.0       # get_gaze_mask 的温度参数
    ```
  - GC-DDPM-BC：在 `agent_kwargs.saliency` 下配置：
    ```yaml
    agent_kwargs:
      saliency:
        enabled: true
        weight: 0.2
        beta: 1.0
    ```

- 日志指标：
  - `saliency_reg`: MSE 正则项
  - `total_loss`: 含权重后的总损失

实现要点（已内置，无需改源码）：
- 数据侧：`bdv2_to_numpy.py` 读取 `saliency_map.pkl`，写入样本的 `observations.saliency`（HWC float32，C=1）；`bridge_numpy.py` 将其转换为张量 `(B,1,H,W)`，不做增强、不做时间堆叠。
- 模型侧：`agents/{bc,gc_bc,gc_ddpm_bc}.py` 注册 forward hook 捕获编码器最后的 4D 卷积特征图，调用 `get_gaze_mask` 上采样并与 batch 中的 `observations.saliency` 做 MSE，加权至总损失。
# Bridge Torch — Clean Hydra + Torch BC

This repository is a streamlined, PyTorch-only pipeline for training Behavior Cloning (BC) policies on BridgeData v2 (bdv2) numpy datasets. It follows a clean Hydra configuration pattern, uses a standard PyTorch `Dataset`/`DataLoader`, and keeps advanced runtime options (DDP, AMP, torch.compile, W&B) while removing legacy configuration nesting and deprecated algorithms.

What’s included:

- Behavior Cloning (BC) with Gaussian policy and ResNet-based encoder
- Optional saliency regularization (from gaze maps) integrated into the training loss
- Clean Hydra schema (flat, discoverable, composable) for model/optimizer/data
- Torch-style bdv2 `Dataset` and `DataLoader` with augmentation
- DDP via `torchrun`, AMP (bf16/fp16), optional `torch.compile`
- W&B logging and structured checkpoints

What’s removed in this refactor:

- GC-BC and GC-DDPM-BC (and related configs) to keep the codebase simple and focused on BC

If you need GC variants, please use a prior commit/branch before this refactor.


## Quick Start

1) Install dependencies (PyTorch per your CUDA setup, plus libs below). Example with pip:

```bash
pip install -e .
pip install hydra-core omegaconf albumentations opencv-python pillow tqdm imageio wandb
# Install torch/torchvision from the official site for your CUDA version
```

2) Prepare data in bdv2 numpy format (see Data section). Then run training:

```bash
python experiments/train_hydra.py \
  name=my_bc_run \
  bridgedata=lift_carrot_100 \
  data_path=/path/to/bdv2_numpy
```

Common overrides:

```bash
# Model
model.hidden_dims=[512,512] model.dropout_rate=0.1
model.use_proprio=true

# Data
data.obs_horizon=2 data.augment=true

# Saliency
saliency.enabled=true saliency.weight=5 saliency.beta=1.0 saliency.alpha=1.0

# Optimizer/Scheduler
optimizer.lr=3e-4 scheduler.type=warmup_cosine scheduler.warmup_steps=2000
```


## Configuration

Hydra entry: `conf/config.yaml` merges the BC preset `conf/algo/bc.yaml` and a dataset preset under `conf/bridgedata/*`.

- Top-level commonly used keys:
  - `name`: experiment name, used for run directory names and W&B
  - `data_path`: bdv2 numpy root (contains `task/train/out.npy` and `task/val/out.npy`)
  - `batch_size`, `val_batch_size`, `num_steps`, `log_interval`, `eval_interval`, `save_interval`
  - `encoder` + `encoder_kwargs`: visual backbone configuration (ResNet v1 bridge)
  - `saliency`: `{enabled, weight, beta, alpha}`
  - `data`: dataset-related knobs (replaces old `dataset_kwargs`)
  - `model`, `optimizer`, `scheduler`: BC model/opt configuration (replaces old `agent_kwargs/*`)

Preset `conf/algo/bc.yaml` defines:

- `model`: `hidden_dims`, `dropout_rate`, `tanh_squash_distribution`, `state_dependent_std`, `fixed_std`, `use_proprio`
- `optimizer`: `lr`, `weight_decay`
- `scheduler`: `{type, warmup_steps}`; `decay_steps` defaults to `${num_steps}` from root config
- `data`: `obs_horizon`, `act_pred_horizon`, `augment`, `augment_kwargs`, `saliency_alpha`, etc.

Dataset preset `conf/bridgedata/*` specifies which tasks to include (via `include`) and ships basic metadata (`_base.yaml`).


## Data

The pipeline expects bdv2 numpy format with this structure:

```
<data_path>/
  <task_name>/
    train/out.npy
    val/out.npy
```

Each `out.npy` is an array of trajectory dicts with fields like:

- `observations` / `next_observations`: image(s) in subkeys like `images0` (HWC uint8), optional `state` (proprio)
- `actions`: float arrays
- Optional `saliency` per observation (HWC float32, C=1, in [0,1])

If your data is raw BridgeData, convert to bdv2 using the included converter:

```bash
python data/bdv2_to_numpy.py --help
```

See in-script docs for expected raw directory structure.


## Training (single GPU)

```bash
python experiments/train_hydra.py \
  name=lift_carrot_100_bc \
  bridgedata=lift_carrot_100 \
  data_path=/data/bdv2_numpy \
  batch_size=2000 num_steps=30000
```

Useful flags:

- `amp.enabled=true amp.dtype=bf16` for mixed precision
- `compile.enabled=true` to try `torch.compile`
- `wandb.enabled=false` to disable W&B


## Training (multi-GPU, DDP)

```bash
torchrun --nproc_per_node=8 experiments/train_hydra.py \
  ddp.enabled=true \
  bridgedata=lift_carrot_100 data_path=/data/bdv2_numpy
```

Notes:

- The script auto-initializes process group and sets `DistributedSampler` for train/val loaders.
- Use `--nproc_per_node` equal to your GPU count.


## Evaluation

`experiments/eval.py` loads a saved run from disk and instantiates a BC agent using the stored `config.json` and the latest checkpoint.

Example:

```bash
python experiments/eval.py --save_dir /path/to/run_dir
```


## Saliency Regularization

- Data-side saliency maps (if present) are aggregated temporally when `data.obs_horizon>1` using `alpha` from `saliency.alpha`.
- Model-side regularization computes a predicted spatial mask from the encoder’s last 4D feature map (captured via a forward hook) and trains it towards the provided saliency target using MSE:
  - Implementation: `common/gaze.py:get_gaze_mask`
  - Controls: `saliency.enabled`, `saliency.weight`, `saliency.beta` (temperature)


## Project Structure

```
agents/
  bc.py                # GaussianPolicy + BCAgent
common/
  gaze.py              # get_gaze_mask for saliency regularization
conf/
  algo/bc.yaml         # BC preset (model/optimizer/data)
  bridgedata/*.yaml    # dataset presets
  config.yaml          # root config
data/
  bridge_numpy.py      # BridgeDataset + DataLoader builder
  bdv2_to_numpy.py     # raw -> bdv2 converter
experiments/
  train_hydra.py       # training entry
  eval.py              # evaluation entry
models/
  encoders.py          # ResNetV1-Bridge encoder
```


## Tips & Troubleshooting

- Devices: set `device=cuda` (default) or `device=cpu` in `conf/config.yaml`.
- DataLoader workers: tune under `dataloader.*` in the root config.
- Batch sizes with DDP: global `batch_size` is divided across ranks automatically.
- Checkpoints: saved under `save_dir/<project>/<name>_<time>/ckpt_*.pt` with a copy of the full resolved config.


## Changelog (Refactor)

- Flattened Hydra schema: `model`, `optimizer`, `scheduler`, `data` (replacing nested `agent_kwargs/*`, `dataset_kwargs`)
- Torch-style `BridgeDataset` and `make_dataloader` replace custom iterable samplers
- Removed GC-BC and GC-DDPM-BC to simplify codebase and configs
- Kept DDP/AMP/compile/W&B integration


## Citation

If you use BridgeData or related components, please cite the original BridgeData work. This repo focuses on a PyTorch training stack and does not change dataset licensing or attribution.
