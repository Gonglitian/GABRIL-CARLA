## 评估指南（GC/LC/服务端）

本文档介绍如何使用本仓库的评估脚本，对训练好的策略进行在线评估与录像保存。对应脚本位于 `experiments/`：
- 视觉目标条件（GC）：`experiments/eval_gc.py`
- 语言条件（LC）：`experiments/eval_lc.py`
- 机器人服务端评估（本地/远端控制）：`experiments/eval.py`

请根据你的策略训练方式与实际机器人接入方式，选择相应入口。

### 1. 前置准备
- 安装依赖并确保能导入本仓库模块。
- 准备以下输入：
  - 已训练的权重文件：`--checkpoint_weights_path`（支持传入多个）
  - 训练时保存的配置 JSON：`--checkpoint_config_path`（与权重一一对应）
  - 相机输入尺寸：`--im_size`（需与实时相机/录制尺寸一致）
  - 可选：视频保存目录 `--video_save_path`
- 若使用服务端脚本 `experiments/eval.py`：
  - 指定 `--goal_type` 为 `gc` 或 `lc`
  - 指定机器人服务地址：`--ip`、`--port`

建议将路径写为绝对路径。例如：`/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_data_v2/...`

### 2. 评估入口与交互流程

#### 2.1 GC（目标图像条件）— `experiments/eval_gc.py`
运行后流程：
- 程序会让你为每次 rollout 采集目标图像（从相机抓拍或用 `--goal_image_path` 提供）。
- 如果传入多个 checkpoint，会在开始时提示选择策略。
- 程序每步调用策略输出 7 维动作（末位为夹爪开合），并执行粘滞夹爪逻辑和可选自由度约束。
- 如提供 `--video_save_path`，会在结束时保存视频（目标图+观察图拼接）。

关键内部参数（如需修改请直接改脚本）：
- `STEP_DURATION=0.2`：控制时间步时长与视频 FPS。
- `STICKY_GRIPPER_NUM_STEPS=1`：夹爪动作粘滞步数。
- `NO_PITCH_ROLL`/`NO_YAW`：如需去掉对应自由度可置为 `True`。

#### 2.2 LC（语言条件）— `experiments/eval_lc.py`
运行后流程：
- 程序会提示你输入一条语言指令；内部会用 `text_processors[...]` 将其编码为向量。
- 其余交互与 GC 基本一致，可多策略选择、可保存视频。

#### 2.3 服务端评估（GC/LC 统一）— `experiments/eval.py`
适用于通过 `WidowXClient` 服务进行控制的场景：
- 使用 `--goal_type` 明确是 `gc` 还是 `lc`。
- 指定 `--ip`、`--port` 与机器人控制服务连接。
- GC 模式下可交互采集目标图；LC 模式下会提示输入指令并编码。

注意：三个脚本都会从配置 JSON 中加载 `encoder`、`agent`、`dataset_kwargs`、`bridgedata_config.action_proprio_metadata` 等关键信息；动作会按 `action_mean/std` 反归一化。

### 3. 运行示例（使用绝对路径）

以下示例统一以仓库根目录 `/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_data_v2` 为前缀。

#### 3.1 单策略 GC 评估
```bash
python /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_data_v2/experiments/eval_gc.py \
  --checkpoint_weights_path /abs/path/to/ckpt_200000 \
  --checkpoint_config_path /abs/path/to/config.json \
  --im_size 128 \
  --goal_image_path /abs/path/to/goal.png \
  --video_save_path /abs/path/to/eval_videos \
  --num_timesteps 120 \
  --act_exec_horizon 1 \
  --deterministic True
```

#### 3.2 单策略 LC 评估
```bash
python /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_data_v2/experiments/eval_lc.py \
  --checkpoint_weights_path /abs/path/to/ckpt_200000 \
  --checkpoint_config_path /abs/path/to/config.json \
  --im_size 128 \
  --video_save_path /abs/path/to/eval_videos \
  --num_timesteps 120 \
  --act_exec_horizon 1 \
  --deterministic True
```

运行时会提示输入指令，例如 “把锅盖放进锅里”。

#### 3.3 服务端（GC）评估
```bash
python /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_data_v2/experiments/eval.py \
  --checkpoint_weights_path /abs/path/to/ckpt_200000 \
  --checkpoint_config_path /abs/path/to/config.json \
  --goal_type gc \
  --im_size 128 \
  --ip localhost --port 5556 \
  --video_save_path /abs/path/to/eval_videos \
  --num_timesteps 120 \
  --act_exec_horizon 1 \
  --deterministic True
```

#### 3.4 多策略对比（传入多个 checkpoint）
可多次传入 `--checkpoint_weights_path` 与 `--checkpoint_config_path`，例如：
```bash
python /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_data_v2/experiments/eval_gc.py \
  --checkpoint_weights_path /abs/runA/ckpt_200000 \
  --checkpoint_weights_path /abs/runB/ckpt_250000 \
  --checkpoint_config_path /abs/runA/config.json \
  --checkpoint_config_path /abs/runB/config.json \
  --im_size 128 --num_timesteps 120 --deterministic True
```
程序会在开始时列出策略列表，供你交互选择。

### 4. 常用参数说明

- `--checkpoint_weights_path` / `--checkpoint_config_path`：可重复传入，数量须一致。
- `--im_size`：输入图像尺寸，需与相机/数据一致。
- `--goal_image_path`（GC）：可直接提供目标图像，跳过交互拍摄。
- `--num_timesteps`：每次 rollout 的时间步数。
- `--act_exec_horizon`：若策略输出动作序列，此值控制每步执行的序列长度（通常为 1）。
- `--deterministic`：是否使用 `argmax` 动作；若为 `False` 则从策略分布采样。
- `--video_save_path`：若提供则保存评估视频。
- `--ip` / `--port` / `--goal_type`（仅 `experiments/eval.py`）：服务端连接与评估模式设置。

内部常量（如需更改请编辑脚本）：`STEP_DURATION`、`NO_PITCH_ROLL`、`NO_YAW`、`STICKY_GRIPPER_NUM_STEPS`、`WORKSPACE_BOUNDS`、`FIXED_STD`。

### 5. 常见问题排查

- 维度/形状不匹配：
  - 确认 `--im_size` 与相机/数据一致；
  - 若训练使用了 `obs_horizon`/`act_pred_horizon`，对应配置需在 JSON 中正确保存。
- KeyError/字段缺失：
  - `config.json` 需包含 `encoder`、`encoder_kwargs`、`agent`、`agent_kwargs`、`dataset_kwargs`、`bridgedata_config.action_proprio_metadata` 等字段。
- 控制失败或环境异常：
  - 检查相机话题与边界设置（`WORKSPACE_BOUNDS`）；
  - 服务端脚本需确保 `--ip` `--port` 可连接。
- 轨迹中夹爪频繁抖动：
  - 适当增大 `STICKY_GRIPPER_NUM_STEPS`；或将 `FIXED_STD` 设为 0 以避免额外噪声。

### 6. 输出与记录

- 若设置 `--video_save_path`，脚本将以时间戳与策略名命名输出 MP4 文件；GC 模式保存“目标图+观察图”拼接视频。
- 运行日志会打印到标准输出，包含策略选择、环境复位、每步执行与异常栈等信息。

如需扩展评估逻辑（例如自动化多目标批量评估、指标统计），建议在上述脚本基础上新增入口，保留动作反归一化与观测堆叠逻辑，以确保与训练配置一致。

