### 训练指南（BridgeData V2，本地数据结构适配）

本文档说明如何基于你的本地数据结构与路径完成数据预处理与训练。你的数据根目录示例：

```json
{
  "dataset_root": "/data3/vla-reasoning/dataset/bdv2",
  "samples": [
    {
      "task": "open_microwave",
      "traj_path": "/data3/vla-reasoning/dataset/bdv2/open_microwave/2022-03-12_14-48-28/raw/traj_group0/traj0",
      "images.cameras_found": ["images0", "images1", "images2"],
      "observations.key_shapes": {
        "full_state": [39, 7],
        "time_stamp": [39]
      },
      "actions": {"step_count": 38, "action_dim": 7}
    }
  ]
}
```

---

### 1. 环境准备

- Python 3.10（建议使用 Conda 虚拟环境）
- 安装依赖：
```bash
conda create -n jaxrl python=3.10 -y
conda activate jaxrl
pip install -e .
pip install -r requirements.txt
# GPU（CUDA11 示例）：
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
注意：训练脚本会禁止 TensorFlow 使用 GPU（仅 JAX 用 GPU），无需改动。

---

### 2. 数据预处理（原始 → NumPy → TFRecord）

项目使用自定义 `tf.data` 数据加载器（`jaxrl_m/data/bridge_dataset.py`）。训练脚本读取 TFRecord，每条 Example 是一段轨迹。数据需分两步转换：

#### 2.1 原始数据 → NumPy（`data_processing/bridgedata_raw_to_numpy.py`）

你的原始数据路径类似：`/data3/vla-reasoning/dataset/bdv2/<task>/<timestamp>/raw/traj_group*/traj*`。
- 对应脚本参数 `--depth` 的语义：在 `--input_path` 下，匹配 `dir_1/.../dir_{depth-1}/<dated_dir>` 的模式。
- 你的结构为 `<task>/<dated_dir>`，因此设定 `--depth=2`（即在 `bdv2/*/<dated_dir>` 层级下寻找）。
- 默认将图像缩放为 128×128（可用 `--im_size` 修改）。

示例命令：
```bash
python data_processing/bridgedata_raw_to_numpy.py \
  --input_path /data3/vla-reasoning/dataset/bdv2 \
  --output_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --depth 2 \
  --train_proportion 0.9 \
  --num_workers 8 \
  --im_size 128
```
输出示例结构：
```
/data3/vla-reasoning/dataset/bdv2_numpy/
  open_microwave/
    train/out.npy
    val/out.npy
  <other_task>/
    train/out.npy
    val/out.npy
```

#### 2.2 NumPy → TFRecord（`data_processing/bridgedata_numpy_to_tfrecord.py`）

- `--depth` 的语义：在 `--input_path` 下匹配 `{.../dir_{depth-1}}/(train|val)/out.npy`。
- 对应你的结构，仍设 `--depth=2`（即 `bdv2_numpy/*/(train|val)/out.npy`）。

示例命令：
```bash
python data_processing/bridgedata_numpy_to_tfrecord.py \
  --input_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --output_path /data3/vla-reasoning/dataset/bdv2_tfrecord \
  --depth 2 \
  --num_workers 8
```
输出示例结构：
```
/data3/vla-reasoning/dataset/bdv2_tfrecord/
  open_microwave/
    train/out.tfrecord
    val/out.tfrecord
  <other_task>/
    train/out.tfrecord
    val/out.tfrecord
```
说明：
- 转换脚本仅写入 `images0` 与 `state`（多相机会在原始→NumPy阶段收集，但 TFRecord 仅使用 `images0`）。
- `actions` 期望维度为 7，其中第 7 维为连续夹爪信号；训练时会自动离散化为开/合二值。

---

### 3. 训练配置与启动

训练入口：`experiments/train.py`。
- 方法选择见 `experiments/configs/train_config.py`：`gc_bc`、`gc_ddpm_bc`、`gc_iql`、`contrastive_rl_td`、`lc_bc`（语言条件）。
- 数据包含/排除与动作归一化在 `experiments/configs/data_config.py`。

#### 3.1 为你的本地 TFRecord 目录添加一个数据配置项

默认的 `data_config.py:all` 使用更深层目录模式（如 `bridge_data_v2/?*/?*/?*`），与你的本地目录不一致。建议在 `experiments/configs/data_config.py` 新增一个条目（例如 `bdv2_local`）：

```python
# 在 possible_structures 字典中新增：
"bdv2_local": ml_collections.ConfigDict({
    "include": [["?*"]],  # 匹配 <task> 这一层目录
    "exclude": [],
    "sample_weights": None,
    "action_proprio_metadata": ACTION_PROPRIO_METADATA,
})
```
- `include` 的模式与 `--config.data_path` 拼接后，脚本会在该目录下追加 `train/out.tfrecord` 与 `val/out.tfrecord`。
- 如需禁用动作/本体归一化，可把 `action_proprio_metadata` 置为 `None`。

#### 3.2 启动训练（以 GCBC 为例）

```bash
python experiments/train.py \
  --config experiments/configs/train_config.py:gc_bc \
  --bridgedata_config experiments/configs/data_config.py:bdv2_local \
  --name gcbc_bdv2_local \
  --config.save_dir /data3/vla-reasoning/exp_out/bridge_runs \
  --config.data_path /data3/vla-reasoning/dataset/bdv2_tfrecord \
  --config.batch_size 256 \
  --config.num_steps 2000000
```
关键说明：
- 批大小需能被本机 `jax.local_devices()` 数量整除（多卡均分）。
- Checkpoint 保存目录：`${save_dir}/jaxrl_m_bridgedata/${exp_descriptor}_${unique_identifier}`。
- WANDB：默认启用，需在环境中配置 `WANDB_API_KEY`（或在离线/禁用环境配置中使用 debug/offline）。

#### 3.3 其他可选方法

- GCIQL：`--config experiments/configs/train_config.py:gc_iql`
- 对比强化学习：`--config experiments/configs/train_config.py:contrastive_rl_td`
- 语言条件 BC：`--config experiments/configs/train_config.py:lc_bc`（需要语言标签；若无语言，使用 `gc_bc`）。

---

### 4. 常见问题与排查

- 批大小断言失败：`assert FLAGS.config.batch_size % num_devices == 0`。请把 `--config.batch_size` 设为设备数的整数倍。
- 找不到数据（空匹配或断言失败）：
  - 确认 TFRecord 目录为 `--config.data_path`；
  - 确认 `bridgedata_config.include` 与目录层级一致（本指南建议 `?*` 即任务一级目录）；
  - 转换脚本的 `--depth` 应为 2（对应 `<task>/<dated_dir>` → NumPy；`<task>/(train|val)` → TFRecord）。
- 显存不足：
  - 降低 `--config.batch_size`；
  - 关闭图像增强：把所选方法在 `train_config.py` 中的 `dataset_kwargs.augment=False`；
  - 使用更小分辨率（原始→NumPy阶段设 `--im_size 128`）。
- 动作/本体归一化不合适：
  - 在 `data_config.py` 中自定义 `ACTION_PROPRIO_METADATA`（均值/方差、min/max）；或设为 `None` 关闭归一化。
- 恢复训练：
  - 通过 `--config.resume_path /path/to/checkpoint_xxx` 指向最近的 checkpoint 目录。

---

### 5. 目录与键值对照（供核对）

- TFRecord 内部键（训练读取）：
  - `observations/images0`（uint8，T×H×W×C）→ `observations.image`
  - `observations/state`（float32，T×7）→ `observations.proprio`
  - `next_observations/images0`、`next_observations/state` 同上
  - `actions`（float32，T×7）
  - `terminals`（bool，T），`truncates`（bool，T）
  - 若语言条件：`language`（可选，字符串数组）
- 你的原始数据中多相机（`images1`、`images2`）不会被写入 TFRecord（当前管线仅使用 `images0`）。

---

### 6. 最小可复现清单

1) 原始 → NumPy：
```bash
python data_processing/bridgedata_raw_to_numpy.py \
  --input_path /data3/vla-reasoning/dataset/bdv2 \
  --output_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --depth 2 --im_size 128 --num_workers 8
```
2) NumPy → TFRecord：
```bash
python data_processing/bridgedata_numpy_to_tfrecord.py \
  --input_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --output_path /data3/vla-reasoning/dataset/bdv2_tfrecord \
  --depth 2 --num_workers 8
```
3) 训练（GCBC 示例）：
```bash
python experiments/train.py \
  --config experiments/configs/train_config.py:gc_bc \
  --bridgedata_config experiments/configs/data_config.py:bdv2_local \
  --name gcbc_bdv2_local \
  --config.save_dir /data3/vla-reasoning/exp_out/bridge_runs \
  --config.data_path /data3/vla-reasoning/dataset/bdv2_tfrecord \
  --config.batch_size 256 --config.num_steps 2000000
```

如需，我可以直接为你添加 `bdv2_local` 配置并提供一键运行脚本。

