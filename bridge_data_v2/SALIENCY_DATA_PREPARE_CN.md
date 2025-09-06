```bash
# 1. 将原始数据转换为numpy格式
python ./saliency_data_processing/saliency_raw_to_numpy.py --input_path /data3/vla-reasoning/dataset/bdv2  --output_path /data3/vla-reasoning/dataset/bdv2_numpy_gaze --depth 2 --im_size 128 --bbox_source no_filter --gaze_k 2 --gaze_alpha 0.7 --gaze_beta 0.7 --gaze_gamma_frac 0.06 --num_workers 8

# 2. 将numpy数据转换为tfrecord格式
python ./saliency_data_processing/saliency_numpy_to_tfrecord.py \
--input_path /data3/vla-reasoning/dataset/bdv2_numpy_gaze \
--output_path /data3/vla-reasoning/dataset/bdv2_tfrecord_gaze \
--depth 2 \
--num_workers 8
```

## 显著性（凝视热图）数据处理使用说明

本文档说明如何使用 `bridge_data_v2/saliency_data_processing/` 目录下的两个脚本，将 BridgeData 原始数据集转换为带“凝视/显著性”热图的 Numpy 与 TFRecord 格式：

- `saliency_raw_to_numpy.py`：从原始数据目录读取轨迹，按需生成每帧的显著性热图（来自 VLM 过滤/未过滤的 bbox pickle），并导出为 numpy 文件。
- `saliency_numpy_to_tfrecord.py`：将上述 numpy 数据转换为 TFRecord，保留可选的 `gaze` 字段。

脚本路径：
- `bridge_data_v2/saliency_data_processing/saliency_raw_to_numpy.py`
- `bridge_data_v2/saliency_data_processing/saliency_numpy_to_tfrecord.py`

建议使用绝对路径运行，下面示例统一以仓库根目录 `/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA` 为前缀。

---

### 1. 前置条件与数据要求

- 依赖安装：参考 `bridge_data_v2/requirements.txt` 并确保可导入 `tensorflow`、`absl-py`、`tqdm`、`numpy`、`Pillow` 等。
  - 如缺少 Pillow：`pip install pillow`
- 原始数据目录需为 BridgeData v2 的标准布局（与 `data_processing/bridgedata_raw_to_numpy.py` 相同），每条轨迹应包含：
  - `obs_dict.pkl`，`policy_out.pkl`
  - 图像目录：如 `images0/im_0.jpg, im_1.jpg, ...`（脚本会将图像缩放到 `--im_size`）
  - 语言可选：`lang.txt`
- 额外要求（用于显著性热图）：每条轨迹目录下需有以下其一（可两者都有）：
  - `filter.pkl`：VLM 过滤后的 bbox
  - `no_filter.pkl`：未过滤的 bbox

两种 bbox 文件的格式相同：按帧的列表，元素为 `[x1, y1, x2, y2]`，坐标范围归一化到 `[0, 1]`。

示例输入目录层级（省略不相关文件）：
```
bridgedata_raw/
  rss/
    toykitchen2/
      set_table/
        00/
          2022-01-01_00-00-00/
            raw/
              traj_group0/
                traj0/
                  obs_dict.pkl
                  policy_out.pkl
                  images0/
                    im_0.jpg
                    im_1.jpg
                    ...
                  filter.pkl      # 可选：VLM 过滤 bbox
                  no_filter.pkl   # 可选：未过滤 bbox
              ...
        01/
        ...
```

---

### 2. 步骤一：原始数据 + bbox → Numpy（带 gaze 热图）

脚本：`bridge_data_v2/saliency_data_processing/saliency_raw_to_numpy.py`

功能概述：
- 读取原始数据目录，生成每条轨迹的 `observations` 与 `next_observations`；
- 从 `filter.pkl` 或 `no_filter.pkl` 读取每帧 bbox，按“时序多模态高斯”算法将 bbox 中心转为凝视热图，并与图像同尺寸对齐；
- 将热图写入 `observations/gaze` 与 `next_observations/gaze`（形状 `[H, W, 1]`，`float32`）；
- 按 `--train_proportion` 将轨迹划分为 train/val，并分别导出为 `out.npy`。

常用参数：
- `--input_path`：原始数据根目录。
- `--output_path`：输出根目录。
- `--depth`：目录深度控制。若 `--depth=5`，脚本会匹配 `input_path/*/*/*/*` 这些叶目录，将其视作“直接包含日期文件夹”的父目录进行处理。
- `--im_size`：图像缩放尺寸（方形），gaze 热图也会按此尺寸生成，需与后续训练配置一致。
- `--bbox_source`：`filter` 或 `no_filter`，选择使用哪份 bbox pickle。
- `--train_proportion`：训练集占比，默认 0.9（注意：本脚本按遍历顺序划分，未随机打乱）。
- `--num_workers`：并行进程数。
- `--overwrite`：如目标输出已存在，是否覆盖。

凝视热图（时序多模态高斯）超参：
- `--gaze_k`：时间窗口半径 k（整数，默认 2）。对第 i 帧，累加 i±j 帧（j∈[0,k]）的高斯核。
- `--gaze_alpha`：强度衰减系数 α ∈ (0,1)（默认 0.7），权重为 `α^{|j|}`。
- `--gaze_beta`：半径扩张系数 β ∈ (0,1)（默认 0.7），标准差随时间偏移为 `σ_j = γ · β^{-abs(j)}`。
- `--gaze_gamma_frac`：基础标准差 γ 的像素比例（默认 0.06），即 `γ = gaze_gamma_frac × im_size`。

算法说明：对第 i 帧，记 bbox 中心为注视点 `(x, y)`（归一化到 `[0,1]` 后再映射到像素坐标），在每个时间偏移 `j ∈ [-k, k]` 上累加二维高斯核：

`g_i = Σ_{j=-k..k} α^{|j|} · N([x_{i+j}, y_{i+j}], (γ^2 · β^{-2|j|}) I)`，并对每帧进行 [0,1] 的 min-max 归一化。

实现细节：
- 多个 bbox 则视作多模态高斯的叠加；
- 使用归一化二维高斯 PDF，随后按帧归一化，兼顾“半径扩张 + 强度衰减”；
- 若某帧缺少 bbox，仍可从邻近帧贡献（窗口内 j≠0 的项）。

运行示例：
```bash
python /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_data_v2/saliency_data_processing/saliency_raw_to_numpy.py \
  --input_path /abs/path/to/bridgedata_raw \
  --output_path /abs/path/to/bridgedata_numpy_gaze \
  --depth 5 \
  --im_size 128 \
  --bbox_source filter \
  --gaze_k 2 --gaze_alpha 0.7 --gaze_beta 0.7 --gaze_gamma_frac 0.06 \
  --num_workers 8
```

输出目录结构（会复用输入目录的尾部路径）：
```
bridgedata_numpy_gaze/
  toykitchen2/
    set_table/
      00/
        train/out.npy
        val/out.npy
```

每条轨迹（numpy 元素）包含字段：
- `observations` / `next_observations`：字典列表，键包括 `images0`、`state`、`time_stamp`；若存在还包含 `gaze`（`float32`，形状 `[H, W, 1]`），其数值为按上文“时序多模态高斯”算法生成并按帧归一化的热图。
- `actions`：`float32` 列表。
- `terminals`：与 `actions` 等长的 0/1 序列（末两步会标为 1）。
- `language`：字符串列表（若无 `lang.txt` 则为空字符串占位）。

注意：如对应轨迹缺少 `filter.pkl`/`no_filter.pkl`，脚本会打印 warning 并跳过写入 `gaze` 键，其余数据仍会正常保存。若窗口内的其它帧也缺少 bbox，热图将退化为零图。

---

### 3. 步骤二：Numpy → TFRecord（保留 gaze 字段）

脚本：`bridge_data_v2/saliency_data_processing/saliency_numpy_to_tfrecord.py`

功能概述：
- 扫描上一步生成的 `train/out.npy` 与 `val/out.npy`；
- 为每条轨迹构造 `tf.train.Example` 并写入 `out.tfrecord`；
- 若 numpy 轨迹包含 `gaze`，会额外写入：
  - `observations/gaze`（`float32`，[T, H, W, 1]）
  - `next_observations/gaze`（`float32`，[T, H, W, 1]）

常用参数：
- `--input_path`：上一步的 numpy 输出根目录（例如 `/abs/path/to/bridgedata_numpy_gaze`）。
- `--output_path`：TFRecord 输出根目录。
- `--depth`：与上一步含义一致，用于复用尾部路径层级并在对应位置写出 `out.tfrecord`。
- `--num_workers`：并行进程数。
- `--overwrite`：如已存在 `out.tfrecord` 是否覆盖。

运行示例：
```bash
python /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_data_v2/saliency_data_processing/saliency_numpy_to_tfrecord.py \
  --input_path /abs/path/to/bridgedata_numpy_gaze \
  --output_path /abs/path/to/bridgedata_tfrecord_gaze \
  --depth 5 \
  --num_workers 8
```

输出目录结构：
```
bridgedata_tfrecord_gaze/
  toykitchen2/
    set_table/
      00/
        train/out.tfrecord
        val/out.tfrecord
```

TFRecord 中的主要键：
- `observations/images0`（uint8）
- `observations/state`（float32）
- `next_observations/images0`（uint8）
- `next_observations/state`（float32）
- `language`（字符串列表，序列化后写入）
- `actions`（float32）
- `terminals`（bool）
- `truncates`（bool，末步为 True）
- 可选：`observations/gaze`、`next_observations/gaze`（float32）

---

### 4. 参数与路径深度小贴士

- `--depth` 定义了“复用多少层尾部目录结构到输出”。
  - 例如：若原始数据位于 `bridgedata_raw/rss/toykitchen2/set_table/00/2022-...`，常用 `--depth=5` 并将 `--input_path` 设为 `bridgedata_raw`，这样脚本会匹配 `rss/toykitchen2/set_table/00` 这些父目录并遍历其中的日期文件夹。
  - 若只处理某一子集（如 `bridgedata_raw/rss/toykitchen2`），可设 `--input_path` 为该子目录，并将 `--depth` 相应减小（例如 3）。
- `--im_size` 请与训练配置一致；热图与图像都会被缩放到该尺寸，保持像素级对齐。
- 调参与建议：`gaze_k=2~3` 一般足够，`gaze_alpha≈0.7`、`gaze_beta≈0.7` 可体现“过去/未来的强度衰减 + 半径扩张”，`gaze_gamma_frac≈0.05~0.08` 随任务做微调。
- 训练/验证划分：`saliency_raw_to_numpy.py` 按遍历顺序前 `train_proportion` 比例划入训练集，不做随机打乱；如需随机性，可自行在输入目录层级上打乱或调整脚本。

---

### 5. 常见问题排查

- 找不到 `filter.pkl`/`no_filter.pkl`：
  - 轨迹仍会导出，但不会包含 `gaze` 键；请检查 pickle 放置路径是否为每条 `traj*/` 目录下，且内容为按帧的 bbox 列表，范围 `[0,1]`。
- 形状/维度不匹配：
  - 确认 `--im_size` 与后续训练一致；
  - `gaze` 的形状应为 `[T, H, W, 1]`，若为 `[T, H, W]`，转换脚本会自动扩展通道。
- 输出已存在被跳过：
  - 需要重跑时添加 `--overwrite`；或删除对应输出目录后再运行。
- 性能与并行：
  - 可增大 `--num_workers` 提升吞吐，但注意磁盘与内存瓶颈。
  - 大 `gaze_k` 会显著增加计算量；如需更高效可考虑预生成可平移的高斯模板并裁剪（参考 `arxived/importrant_files/gaze/gaze_to_mask.py` 的做法）。

---

### 6. 快速命令汇总

1）原始数据到 Numpy（带 gaze）：
```bash
python bridge_data_v2/saliency_data_processing/saliency_raw_to_numpy.py \
  --input_path /abs/path/to/bridgedata_raw \
  --output_path /abs/path/to/bridgedata_numpy_gaze \
  --depth 5 --im_size 128 --bbox_source filter --num_workers 8
```

2）Numpy 到 TFRecord（保留 gaze）：
```bash
python bridge_data_v2/saliency_data_processing/saliency_numpy_to_tfrecord.py \
  --input_path /abs/path/to/bridgedata_numpy_gaze \
  --output_path /abs/path/to/bridgedata_tfrecord_gaze \
  --depth 5 --num_workers 8
```

如需进一步与训练脚本集成，请参考 `bridge_data_v2/TRAINING_GUIDE_CN.md` 与 `bridge_data_v2/train_guide.md` 中的数据加载配置说明。
