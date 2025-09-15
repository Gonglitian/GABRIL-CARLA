import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 默认颜色映射（可按需扩展/修改）
DEFAULT_METHOD_COLORS: Dict[str, str] = {
    "GMD": "#ED784A",           
    "ViSaRL": "#2E86AB",
    "GRIL": "#6C5B7B",
    "BC": "#7A7D7D",
    "AGIL": "#27AE60",
    "GABRIL+GMD": "#C0392B",
    "GABRIL": "#F39C12",
}


def make_style(ax: plt.Axes) -> None:
    """应用背景与网格样式。"""
    ax.set_facecolor("#F8F8F8")
    ax.yaxis.grid(True, which="major", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def determine_ylabel(metric_name: Optional[str]) -> str:
    if not metric_name:
        return ""
    lower = metric_name.lower()
    if "return" in lower:
        return "Return"
    if "success" in lower or "score" in lower:
        return "Success Rate (%)"
    if "completion" in lower:
        return "Route Completion (%)"
    return metric_name


def choose_color(method_name: str, color_map: Dict[str, str], default_cycle: List[str], idx: int) -> str:
    if method_name in color_map:
        return color_map[method_name]
    return default_cycle[idx % len(default_cycle)]


def nice_category_labels(categories: List[str]) -> List[str]:
    # 将诸如 "Seen_Human" → "Seen\nHuman"
    return [c.replace("_", "\n") for c in categories]


def darken_color(hex_color: str, factor: float = 0.8) -> str:
    """返回略深的颜色。factor 越小越深，取值 (0,1]。"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return f"#{hex_color}"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02X}{g:02X}{b:02X}"


def plot_overlay_from_csv(
    csv_path: Path,
    output_dir: Path,
    title: Optional[str] = None,
    figsize: tuple = (7.2, 3.2),
    color_map: Optional[Dict[str, str]] = None,
    hatch: str = "////",
    dpi: int = 300,
    title_size: int = 16,
    axis_label_size: int = 13,
    tick_size: int = 11,
    legend_size: int = 10,
    annot_size: int = 8,
) -> Path:
    """按方法分组：每个方法绘制两根柱（Seen/Unseen）。

    - 底柱：Human（灰色实心）
    - 叠加柱：Δ = VLM − Human（带阴影，正向上、负向下）
    - 细边框：VLM 的最终高度（仅轮廓）
    """
    df = pd.read_csv(csv_path)
    if "Method" not in df.columns:
        raise ValueError(f"CSV 缺少 'Method' 列: {csv_path}")

    df = df[df["Method"].astype(str).str.lower() != "average"].copy()

    for c in ["Seen_Human", "Seen_VLM", "Unseen_Human", "Unseen_VLM"]:
        if c not in df.columns:
            raise ValueError(f"CSV 缺少列: {c}")

    methods = df["Method"].astype(str).tolist()
    n = len(methods)
    x = np.arange(n, dtype=float)

    # 每个方法四根柱：1 Seen-H, 2 Seen-VLM, 3 Unseen-H, 4 Unseen-VLM
    inner_w = 0.16
    gap = inner_w * 1.4

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    make_style(ax)

    color_map = color_map or DEFAULT_METHOD_COLORS
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    all_values: List[float] = []

    for i, row in enumerate(df.itertuples(index=False)):
        method_name = str(getattr(row, 'Method'))
        base_color = choose_color(method_name, color_map, default_cycle, i)
        dark_color = darken_color(base_color, factor=0.8)

        seen_h = float(getattr(row, 'Seen_Human'))
        unseen_h = float(getattr(row, 'Unseen_Human'))
        seen_v = float(getattr(row, 'Seen_VLM'))
        unseen_v = float(getattr(row, 'Unseen_VLM'))

        all_values.extend([seen_h, seen_v, unseen_h, unseen_v])

        # 四柱位置
        x1 = x[i] - 1.5 * gap
        x2 = x[i] - 0.5 * gap
        x3 = x[i] + 0.5 * gap
        x4 = x[i] + 1.5 * gap

        # 画四根柱：Seen-H, Seen-VLM, Unseen-H, Unseen-VLM（Unseen 使用斜线）
        b1 = ax.bar(x1, seen_h, inner_w, color=base_color, edgecolor=dark_color, linewidth=0.8, label="Human" if i == 0 else None)
        b2 = ax.bar(x2, seen_v, inner_w, color=dark_color, edgecolor=dark_color, linewidth=0.8, label="VLM" if i == 0 else None)
        b3 = ax.bar(x3, unseen_h, inner_w, color=base_color, edgecolor=dark_color, linewidth=0.8, hatch=hatch)
        # Unseen VLM：底层深色柱（保留深色边框），叠加一层仅斜线（白色、无边框）
        b4 = ax.bar(
            x4,
            unseen_v,
            inner_w,
            color=dark_color,
            edgecolor=dark_color,
            linewidth=0.8,
        )
        _b4_hatch = ax.bar(
            x4,
            unseen_v,
            inner_w,
            color="none",
            edgecolor="#F8F8F8",
            linewidth=0.0,
            hatch=hatch,
        )

        # 绝对值标签（两位小数）
        for bars in (b1, b2, b3, b4):
            bar = bars.patches[0]
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.6, f"{val:.2f}", ha='center', va='bottom', fontsize=annot_size)

        #（差值括号标注已移除，应用户要求仅保留柱顶端绝对数值）

    # x 轴与标签
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=tick_size, fontweight='bold')
    ax.set_ylabel("Score (%)", fontweight="bold", fontsize=axis_label_size)
    ax.tick_params(axis='y', labelsize=tick_size)

    # 让 y 轴范围包含可能的负向 Overlay
    all_vals = np.array(all_values, dtype=float)
    ymin = max(0.0, float(np.nanmin(all_vals)) - 6.0)
    ymax = float(np.nanmax(all_vals)) + 10.0
    ax.set_ylim(ymin, ymax)

    # 标题
    if not title:
        env_name = csv_path.stem
        title = f"CARLA-{env_name}"
    ax.set_title(title, fontsize=title_size, pad=18)

    # 参考线与图例
    handles, labels = ax.get_legend_handles_labels()
    # 追加“Unseen（带斜线）”的图例说明
    hatch_proxy = Patch(facecolor='white', edgecolor='#666666', hatch=hatch, label='Unseen (hatched)')
    handles.append(hatch_proxy)
    labels.append('Unseen (hatched)')
    ax.legend(handles, labels, frameon=True, fontsize=legend_size, loc='upper left', bbox_to_anchor=(0, 1), ncol=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{csv_path.stem}_overlay.pdf"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)
    return out_path

def plot_delta_from_csv(
    csv_path: Path,
    output_dir: Path,
    title: Optional[str] = None,
    figsize: tuple = (10, 5),
) -> Path:
    """保留占位（向后兼容），内部改为调用 overlay 以生成差值更直观的叠加图。"""
    return plot_overlay_from_csv(csv_path, output_dir, title=title, figsize=figsize)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CARLA overlay charts (Human + Δ(VLM)) from CSV and export PDF.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="一个或多个 CSV 文件路径（每个文件生成一个 PDF）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="PDF 输出目录（默认与各自 CSV 同目录）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="覆盖默认标题（默认：CARLA Overlay: Human + Δ(VLM) - <csv文件名>）",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[7.2, 3.2],
        help="图形尺寸 (英寸，宽度 高度)，默认 7.2 3.2（ICRA 双栏宽度）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="导出分辨率 DPI，默认 600（适合印刷/投稿）",
    )
    parser.add_argument("--title-size", type=int, default=16, help="标题字号，默认16")
    parser.add_argument("--axis-label-size", type=int, default=13, help="坐标轴标签字号，默认13")
    parser.add_argument("--tick-size", type=int, default=11, help="刻度字号，默认11")
    parser.add_argument("--legend-size", type=int, default=10, help="图例字号，默认10")
    parser.add_argument("--annot-size", type=int, default=8, help="数值标注字号，默认8")
    # 仅保留 overlay 模式，无需额外开关

    args = parser.parse_args()

    csv_paths = [Path(p).expanduser().resolve() for p in args.input]
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = None

    saved: List[Path] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到 CSV: {csv_path}")
        out_dir = output_dir if output_dir else csv_path.parent
        pdf_path = plot_overlay_from_csv(
            csv_path=csv_path,
            output_dir=out_dir,
            title=args.title,
            figsize=tuple(args.figsize),
            dpi=int(args.dpi),
            title_size=int(args.title_size),
            axis_label_size=int(args.axis_label_size),
            tick_size=int(args.tick_size),
            legend_size=int(args.legend_size),
            annot_size=int(args.annot_size),
        )
        saved.append(pdf_path)

    # 打印结果路径，便于在终端查看
    for p in saved:
        print(str(p))


if __name__ == "__main__":
    main()
