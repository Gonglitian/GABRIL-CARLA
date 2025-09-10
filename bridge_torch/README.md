# Bridge Torch (PyTorch Training Port)

本目录提供使用 PyTorch 训练 BridgeData 的 BC / GC-BC / GC-DDPM-BC 算法的完整流程。本文档包含：环境配置、数据准备（从 raw → numpy）、配置与启动训练、分布式与日志、常见问题等。

注意：本实现仅支持 numpy 数据管线（无需 TensorFlow）：
- numpy 管线：读取 `<task>/{train,val}/out.npy`（由 raw 数据转换而来），无需 TF 依赖，效率高、部署简单。

关键入口：
- `bridge_torch/experiments/train.py`: 单次训练入口。
- `bridge_torch/experiments/multi_train.py`: 基于 YAML 的多实验启动器。
- `bridge_torch/experiments/configs/train_config.py`: 算法超参（bc, gc_bc, gc_ddpm_bc）。
- `bridge_torch/experiments/configs/data_config.py`: 任务/数据集选择。

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

## 3. 训练配置

单次训练的通用入口：`bridge_torch/experiments/train.py`，算法配置位于：
- 训练超参：`bridge_torch/experiments/configs/train_config.py`
- 数据选择：`bridge_torch/experiments/configs/data_config.py`

多实验矩阵式提交（推荐）：`bridge_torch/experiments/multi_train.py`，YAML 配置：

`bridge_torch/experiments/configs/multi_train.yaml` 关键项：
- `global.data_root`: 数据根目录（numpy 或 TFRecord 的根）。
- `global.save_dir`: 日志与权重保存目录。
- `matrix.algos`: 选择算法，例如 `["bc", "gc_bc", "gc_ddpm_bc"]`。
- `matrix.tasks`: 选择任务名，`data_config.py` 中已内置如下键：
  - `all`, `bdv2_local`, `remove_pot_lid_150`, `lift_carrot_100`, `lift_carrot_100_confounded`, `pull_pot_100`, `put_carrot_in_pot_100`, `put_in_pot_lid_100`
- 分布式训练（DDP）：`global.ddp.enabled: true`, `nproc_per_node: 4`（按 GPU 数调整）。
- `global.dataset`: 数据集相关覆盖项，会被展开到 `config.dataset_kwargs.*`，适用于所有算法（单个 run 可用 `runs[*].dataset` 覆盖）：
  - `obs_horizon`：按通道堆叠为 `(B,3*obs_horizon,H,W)` 并自动适配编码器输入通道。
  - `act_pred_horizon`：仅 `gc_ddpm_bc` 支持 `>1`；对 `bc/gc_bc` 将在调度器侧强制为 `1` 以保障兼容。
  - `augment`、`augment_next_obs_goal_differently`、`augment_kwargs`：增强开关与参数（Albumentations），`augment_order` 控制顺序。

示例（YAML 片段）：

```yaml
global:
  dataset:
    obs_horizon: 2
    act_pred_horizon: 1
    augment: true
    augment_next_obs_goal_differently: false
    augment_kwargs:
      random_resized_crop:
        scale: [0.8, 1.0]
        ratio: [0.9, 1.1]
      random_brightness: [0.2]
      random_contrast: [0.8, 1.2]
      random_saturation: [0.8, 1.2]
      random_hue: [0.1]
      augment_order:
        - random_resized_crop
        - random_brightness
        - random_contrast
        - random_saturation
        - random_hue
```

示例（使用 numpy 数据根，4 卡 DDP，小步验证）：

```yaml
global:
  data_root: /scr/litian/dataset/bdv2_numpy
  save_dir:  /scr/litian/torch_runs
  train_batch_size: 20
  val_batch_size:   20
  num_steps: 100
  eval_interval: 10
  save_interval: 10
  log_interval: 10
  warmup_steps: 10
  ddp:
    enabled: true
    nproc_per_node: 4

matrix:
  algos: ["gc_ddpm_bc"]
  tasks: ["lift_carrot_100"]
```

---

## 4. 启动训练

多实验矩阵：

```bash
# Dry run 预览命令
python bridge_torch/experiments/multi_train.py --config bridge_torch/experiments/configs/multi_train.yaml --dry_run

# 正式运行（会根据 ddp 设置自动用 torchrun 启动）
python bridge_torch/experiments/multi_train.py --config bridge_torch/experiments/configs/multi_train.yaml
```

单次训练（不经由 YAML）：

```bash
python -m bridge_torch.experiments.train \
  --name gc_ddpm_bc_lift_carrot_100 \
  --config bridge_torch/experiments/configs/train_config.py:gc_ddpm_bc \
  --bridgedata_config bridge_torch/experiments/configs/data_config.py:lift_carrot_100 \
  --config.data_path /scr/litian/dataset/bdv2_numpy \
  --config.save_dir  /scr/litian/torch_runs \
  --config.batch_size 64 --config.val_batch_size 64 \
  --config.num_steps 50000 --config.eval_interval 1000 --config.save_interval 2000
```

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
# 仅预览命令
python bridge_torch/experiments/multi_train.py --config bridge_torch/experiments/configs/multi_train.yaml --dry_run

# 仅单卡、numpy 数据根、小步测试（DDPM）
python -m bridge_torch.experiments.train \
  --name debug_np \
  --config bridge_torch/experiments/configs/train_config.py:gc_ddpm_bc \
  --bridgedata_config bridge_torch/experiments/configs/data_config.py:lift_carrot_100 \
  --config.data_path /scr/litian/dataset/bdv2_numpy \
  --config.save_dir  /scr/litian/torch_runs \
  --config.batch_size 8 --config.val_batch_size 8 \
  --config.num_steps 200 --config.eval_interval 20 --config.save_interval 50
```

---

如需我为你的数据根或 YAML 生成一份专用配置，请告知实际路径与期望 GPU 数，我可以顺手调整并验证一次运行。

## 8. TODO（体检报告）

以下事项可能影响性能/稳定性或可改进：

- **学习率调度（BC/GC-BC）**：为 BC/GC-BC 加入 Linear warmup + Cosine；使 `warmup_steps/decay_steps` 生效，并在每步调用 `scheduler.step()`（DDPM 已具备）。
- **动作/本体归一化（numpy 管线）**：将 `ACTION_PROPRIO_METADATA` 应用于 `actions`/`proprio` 标准化；训练回归标准化目标，推理阶段反标准化，避免量纲不匹配导致的不稳定。
- **动作有界化（BC/GC-BC）**：实现 `tanh_squash_distribution`（TanhTransform 与 log_prob 修正）或在采样/输出处 clamp，与环境动作范围对齐（DDPM 采样已含 clamp）。
- **多帧观测 `obs_horizon`**：核查图像多帧堆叠是否在通道维度生效并贯穿训练/评估流程；目标图像与观测在通道数对齐（重复策略）保持一致，必要时补充单元测试。
- **多任务采样权重**：在 numpy 管线支持 `sample_weights` 以控制多任务混合比例，避免被忽略。
- **DDP 数据分片（numpy）**：为 numpy 迭代器增加按 `rank` 的无重叠分片，减少样本重叠并提升有效吞吐。
- **Checkpoint 扩充**：同时保存/恢复 optimizer（及 scheduler）状态，便于中断恢复与复现实验。
- **全局随机种子**：设置 `numpy/torch/cuda` 种子与 `cudnn` 相关标志，提升复现性。
- **性能优化（可选）**：AMP 混合精度；引入 DataLoader/多进程预处理或并行预取；调整日志/评估频率与 I/O 的权衡。
- **文档/配置提示**：在 README/YAML 明示上述开关的默认行为与建议配置，降低踩坑概率。

---

## 9. Saliency 训练变体（Reg，仅当提供 saliency_map 时可用）

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
