# BridgeData V2 训练指南（基于本仓库）

本指南面向你给出的 BDV2 数据结构与路径，例如：
`/data3/vla-reasoning/dataset/bdv2/<task>/<timestamp>/raw/traj_group*/traj*/{images*,obs_dict.pkl,policy_out.pkl,agent_data.pkl,lang.txt}`。

整体流程：Raw 数据 → NumPy → TFRecord → 训练。仓库中已提供完整脚本与训练入口，本文给出一步到位的可复制命令与建议配置。

---

## 环境准备
- Python 建议 3.10。
- 先安装项目依赖：
  - `pip install -e .`
  - `pip install -r requirements.txt`
- GPU 上安装合适的 JAX/JAXLIB（按本机 CUDA 版本选择）：
  - CUDA 11.x：`pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
  - CUDA 12.x：`pip install --upgrade "jax[cuda12_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- WANDB 可离线运行：`export WANDB_MODE=offline`（或配置你的云端账号）。

注意：训练时 batch size 必须被本机可见加速卡数整除（JAX 会做设备分片）。

---

## 数据预处理
本仓库默认使用自带的 `tf.data` 加载器（`jaxrl_m/data/bridge_dataset.py`）。训练脚本期望的数据是 TFRecord（每个样本为一条轨迹）。`data_processing` 目录提供了两步式转换脚本：

1) Raw → NumPy：`data_processing/bridgedata_raw_to_numpy.py`
2) NumPy → TFRecord：`data_processing/bridgedata_numpy_to_tfrecord.py`

结合你的目录（任务在 `bdv2/<task>/<timestamp>`，时间戳层级在第 2 层），`--depth` 应该为 2。

示例命令（请按需修改路径）：

```bash
# 1) Raw -> NumPy（默认将图像压缩到 128x128，可用 --im_size 调整）
python data_processing/bridgedata_raw_to_numpy.py \
  --input_path /data3/vla-reasoning/dataset/bdv2 \
  --output_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --depth 2 --num_workers 8 --train_proportion 0.9

# 2) NumPy -> TFRecord
python data_processing/bridgedata_numpy_to_tfrecord.py \
  --input_path /data3/vla-reasoning/dataset/bdv2_numpy \
  --output_path /data3/vla-reasoning/dataset/bdv2_tfrecord \
  --depth 2 --num_workers 8
```

转换完成后，数据大致如下：
```
/data3/vla-reasoning/dataset/bdv2_tfrecord/
  open_microwave/
    train/out.tfrecord
    val/out.tfrecord
  put_in_pot_lid/
    train/out.tfrecord
    val/out.tfrecord
  ...
```

- 如果原始轨迹目录包含 `lang.txt`，转换脚本会自动把语言标签写入样本（LC 任务可用）。
- 如果你的原图较大、显存足够，可以在 Raw→NumPy 时设置更大的 `--im_size`，但要和模型编码器的输入分辨率相匹配（默认 128）。

---

## 训练所需配置
训练脚本入口：`experiments/train.py`，需要两个配置：
- 训练配置：`experiments/configs/train_config.py`（选择算法与超参）
- 数据配置：包含需要训练的任务路径模式、动作/本体归一化信息等

仓库自带的数据配置 `experiments/configs/data_config.py:all` 假设目录层级较深（如 `bridge_data_v2/?*/?*/?*`），与你的目录不完全一致。建议新增一个“本地 BDV2 配置”，例如在 `experiments/configs/my_bdv2_config.py` 新建：

```python
# experiments/configs/my_bdv2_config.py
import ml_collections
from experiments.configs.data_config import ACTION_PROPRIO_METADATA

def get_config(config_string="default"):
    return ml_collections.ConfigDict({
        # 一组任务列表（可只写你想训练的任务名）
        # 也可写通配符："?*" 表示所有一级子目录
        "include": [["open_microwave", "put_in_pot_lid"]],
        "exclude": [],
        "sample_weights": None,  # 多任务时可自定义采样权重
        # 用于动作/本体归一化的统计量（先用默认即可，有需要再替换为你数据的统计量）
        "action_proprio_metadata": ACTION_PROPRIO_METADATA,
    })
```

说明：
- `include` 是“列表的列表”，外层用于“多组子数据集”采样，内层为一组目录匹配；训练脚本会在这些目录下自动拼接 `train/out.tfrecord` 与 `val/out.tfrecord`。
- 如需全量匹配，可以写成 `"include": [["?*"]]`。

---

## 选择算法与关键开关
`experiments/configs/train_config.py` 内置若干方法（传入 `:方法名` 选择）：
- `gc_bc`：目标图像条件 BC（无语言）；
- `gc_ddpm_bc`：扩散策略的目标条件 BC（支持 `obs_horizon`/`act_pred_horizon` 序列）；
- `gc_iql`：目标条件 IQL；
- `contrastive_rl_td`：对比式 RL；
- `lcbc`/`lc_bc`：语言条件 BC（若数据含 `lang.txt`）。

数据相关开关（均在 `dataset_kwargs` 下）：
- `augment`/`augment_kwargs`：图像增广；
- `relabel_actions=True`：使用相邻状态差分重标注动作，并将抓手动作二值化；
- `load_language` 与 `skip_unlabeled`（LCBC 中已默认开启）。

---

## 启动训练（命令模板）
将 `DATA_ROOT` 指向 TFRecord 根目录，将 `SAVE_DIR` 指向你的输出目录：

```bash
export DATA_ROOT=/data3/vla-reasoning/dataset/bdv2_tfrecord
export SAVE_DIR=/data3/vla-reasoning/exp/bdv2_runs

# 例 1：语言条件 BC（若存在 lang.txt）
python experiments/train.py \
  --name lcbc_bdv2 \
  --config experiments/configs/train_config.py:lc_bc \
  --bridgedata_config experiments/configs/my_bdv2_config.py:default \
  --config.data_path $DATA_ROOT \
  --config.save_dir $SAVE_DIR \
  --config.batch_size 64

# 例 2：目标条件 BC（纯视觉）
python experiments/train.py \
  --name gcbc_bdv2 \
  --config experiments/configs/train_config.py:gc_bc \
  --bridgedata_config experiments/configs/my_bdv2_config.py:default \
  --config.data_path $DATA_ROOT \
  --config.save_dir $SAVE_DIR \
  --config.batch_size 64

# 例 3：扩散策略（短时序预测）
python experiments/train.py \
  --name ddpm_bdv2 \
  --config experiments/configs/train_config.py:gc_ddpm_bc \
  --bridgedata_config experiments/configs/my_bdv2_config.py:default \
  --config.data_path $DATA_ROOT \
  --config.save_dir $SAVE_DIR \
  --config.batch_size 64 \
  --config.dataset_kwargs.obs_horizon 1 \
  --config.dataset_kwargs.act_pred_horizon 1
```

注意事项：
- 批大小需被 GPU 数整除（例如 2 卡时可用 64、128…）。
- 训练中会自动保存到：`$SAVE_DIR/jaxrl_m_bridgedata/<name>_<uuid>/checkpoint_*`。
- 恢复训练：在命令中额外加 `--config.resume_path /path/to/checkpoint_XXXXX`。

---

## 评估与导出
- 机器人线上评估脚本：`experiments/eval.py`（支持 GC/LC 两种条件方式）。
- 需要配合 WidowX 控制端，详见 `README.md` 与对应注释。
- 仅做离线验证时，你也可以直接加载 checkpoint 后用 `agents[...]` 的 `sample_actions` 在你的数据上跑前向（见 `experiments/eval.py` 的 `load_checkpoint` 逻辑）。

---

## 真机 Eval 评估指南（WidowX）

本节给出“真机”联机评估的完整流程与命令。脚本入口在 `experiments/eval.py`，既支持目标图像条件（GC）也支持语言条件（LC）。

### 0. 前置准备
- 硬件：WidowX 机械臂、相机、急停按钮；建议先在空台无易碎物品环境下联调。
- 控制栈：在机器人侧安装并启动 `bridge_data_robot` 的控制端（推荐 server-client 方式）。参见其文档启动服务，默认端口 `5556`。
- 相机话题与工作空间：脚本默认相机话题为 `/blue/image_raw`（`experiments/eval.py:71`）、工作空间边界（`experiments/eval.py:70`）。若你的环境不同，请修改对应常量后再运行。
- Checkpoint 与配置 JSON：
  - 使用你训练得到的 `checkpoint_xxxxx` 与配套 `*_config.json`，或使用 README 中“Provided Checkpoints”的公开权重（均自带 JSON）。
  - 图像分辨率 `--im_size` 必须与该 JSON/权重匹配（例如 `*_256_config.json` 对应 `--im_size 256`）。

常用可调参数（均在 `experiments/eval.py` 内定义）：
- `--ip`/`--port`：机器人服务地址与端口（见 `experiments/eval.py:58` 和 `experiments/eval.py:59`）。
- `--initial_eep`/`--goal_eep`：起始位姿与采集目标图像时的位姿（单位米，XYZ）。
- `--num_timesteps`：总步数；`--act_exec_horizon`：每次下发的动作序列长度。
- `--show_image`：显示取流画面；无显示环境请关闭。
- `--video_save_path`：保存评估视频。
- 其他安全/行为开关可在 `experiments/eval.py:66`（步时）、`experiments/eval.py:67`/`experiments/eval.py:68`（自由度约束）等位置调整。

### 1. 推荐方案：Server-Client（在评估机上跑 eval.py）
1) 在机器人侧启动 `bridge_data_robot` 的服务端（参见其文档）。确保可通过 `<robot_ip>:5556` 访问。

2) 在评估机（本机/远端均可）运行以下命令（按需修改路径与 IP）：

```bash
# 指向你的 checkpoint 目录
export CHECKPOINT_DIR=/path/to/checkpoint_dir

# 目标图像条件（GCBC/GC*）
python experiments/eval.py \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_300000 \
  --checkpoint_config_path $CHECKPOINT_DIR/gcbc_256_config.json \
  --im_size 256 --goal_type gc \
  --ip <robot_ip> --port 5556 \
  --show_image --blocking \
  --initial_eep [0.30,0.00,0.15] \
  --goal_eep [0.33,0.00,0.12] \
  --num_timesteps 120 \
  --video_save_path /tmp/eval_videos

# 语言条件（LCBC/LC*）
python experiments/eval.py \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_145000 \
  --checkpoint_config_path $CHECKPOINT_DIR/lcbc_256_config.json \
  --im_size 256 --goal_type lc \
  --ip <robot_ip> --port 5556 \
  --show_image --blocking \
  --initial_eep [0.30,0.00,0.15] \
  --num_timesteps 120 \
  --video_save_path /tmp/eval_videos
```

3) 交互流程与提示：
- 多权重同时传入时，程序会列出多个 policy 供选择（`select policy:`）。
- GC：首次会提示是否“采集新的目标图像”。选择 `y` 后，机械臂将张开夹爪并移动到 `--goal_eep`；检查画面后按回车采集目标图；再按回车开始执行。
- LC：会提示输入语言指令（`Instruction?`），回车确认后再次回车开始执行。
- 运行过程中，可加 `--show_image` 观察画面（需要 GUI 环境）。
- 若设置了 `--video_save_path`，脚本会以 `时间戳_策略名_sticky_*.mp4` 保存视频。

备注：
- 联机初始化关键调用见 `experiments/eval.py:229`（`WidowXClient(host=--ip, port=--port)`）。
- 观测与图像显示示例见 `experiments/eval.py:261`；动作下发示例见 `experiments/eval.py:362`。

### 2. 可选方案：Docker Compose（在容器内跑 eval_gc.py / eval_lc.py）
若你使用 `bridge_data_robot` 的 docker-compose 方案，可进入 `bridge_data_v2` 容器内直接运行：

```bash
# GC 评估（容器内）
python experiments/eval_gc.py \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_300000 \
  --checkpoint_config_path $CHECKPOINT_DIR/gcbc_256_config.json \
  --im_size 256 --blocking --num_timesteps 120 \
  --initial_eep [0.30,0.00,0.15] \
  --goal_eep [0.33,0.00,0.12] \
  --video_save_path /tmp/eval_videos

# LC 评估（容器内）
python experiments/eval_lc.py \
  --checkpoint_weights_path $CHECKPOINT_DIR/checkpoint_145000 \
  --checkpoint_config_path $CHECKPOINT_DIR/lcbc_256_config.json \
  --im_size 256 --blocking --num_timesteps 120 \
  --initial_eep [0.30,0.00,0.15] \
  --video_save_path /tmp/eval_videos
```

两脚本的摄像头与工作空间默认值见文件顶部常量，若与你环境不一致需相应修改：
- `experiments/eval_gc.py`：相机话题 `CAMERA_TOPICS=[IMTopic("/blue/image_raw")]`，边界 `WORKSPACE_BOUNDS`。
- `experiments/eval_lc.py`：同上。

### 3. 常见问题与排查
- 一直打印 “Waiting for observations…” 或无图像：
  - 确认机器人端服务已启动、`--ip/--port` 正确无防火墙阻断；
  - 相机话题名称是否为 `/blue/image_raw`，否则需修改脚本常量（`experiments/eval.py:71`）。
- 连接被拒绝/超时：多为服务未启动或端口错误；在机器人端复核服务日志。
- `ValueError: Unknown goal type`：确认传入了 `--goal_type gc` 或 `--goal_type lc`。
- 维度/形状不匹配：
  - `--im_size` 与 checkpoint JSON 分辨率不一致；
  - 使用了 GC 的 JSON 去跑 LC（或反之），导致缺少 `text_processor` 等键（见 `experiments/eval.py:144`）。
- OpenCV 弹窗失败：无图形界面环境，去掉 `--show_image` 或配置 DISPLAY。
- 视频未保存：确认 `--video_save_path` 有写权限；保存逻辑见 `experiments/eval.py:374`。

### 4. 安全与建议
- 首次联机务必在空台、低速下测试（可适当增大 `STEP_DURATION`，见 `experiments/eval.py:66`）。
- 留意工作空间边界（`WORKSPACE_BOUNDS`，见 `experiments/eval.py:70`），不要越界；必要时缩小范围。
- 若需要限制姿态自由度，可在代码中开启 `NO_PITCH_ROLL` / `NO_YAW`（`experiments/eval.py:66-73`）。
- 评估前统一设定 `--initial_eep`，确保每次起始姿态一致，便于对比。

## 常见问题（基于你的数据结构）
- 目录层级与 `include` 不匹配：如果你像示例一样只有一级 `task` 目录，把 `include` 设成 `[["?*"]]` 或手写具体任务名即可。
- 没有语言标签：使用 `gc_bc`/`gc_iql`/`contrastive_rl_td`；有 `lang.txt` 才使用 `lc_bc`。
- 内存/显存不足：
  - 降低 `--config.batch_size`；
  - 使用默认 128 分辨率（或在 Raw→NumPy 时调低 `--im_size`）。
- JAX 设备相关错误：确保 `batch_size % GPU数 == 0`，并且已安装与你 CUDA 版本匹配的 `jax`。

---

## 文件索引（便于快速定位）
- 预处理脚本：`data_processing/bridgedata_raw_to_numpy.py`，`data_processing/bridgedata_numpy_to_tfrecord.py`
- 数据加载器：`jaxrl_m/data/bridge_dataset.py`
- 训练入口：`experiments/train.py`
- 训练配置：`experiments/configs/train_config.py`
- 数据配置（示例）：`experiments/configs/my_bdv2_config.py`

如需我帮你生成 `my_bdv2_config.py` 或根据你的 GPU/任务数量微调参数，请告诉我你的偏好（语言/目标条件、分辨率、GPU 数等）。
