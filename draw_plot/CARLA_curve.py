import argparse
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 全局字体与基础样式，与 CARLA_bar.py 保持一致
plt.rcParams['font.family'] = 'Times New Roman'


def make_style(ax: plt.Axes) -> None:
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
        return "Score (%)"
    if "completion" in lower:
        return "Route Completion (%)"
    return metric_name


def parse_x_from_header(columns: List[str]) -> List[float]:
    # 例如列名：["0%","10%","25%","50%","75%","100%"] → [0,10,25,50,75,100]
    xs: List[float] = []
    for c in columns:
        s = str(c).strip().rstrip("%")
        try:
            xs.append(float(s))
        except Exception:
            # 若无法解析，退化为顺序索引
            return list(range(len(columns)))
    return xs


def plot_curve_from_csv(
    csv_path: Path,
    output_dir: Path,
    metric_name: str = "Score (%)",
    title: Optional[str] = None,
    figsize: tuple = (3.5, 2.6),
    dpi: int = 600,
    title_size: int = 14,
    axis_label_size: int = 11,
    tick_size: int = 9,
    legend_size: int = 9,
    annot_size: int = 7,
) -> Path:
    """从 CSV 绘制曲线图，风格参考 CARLA_bar.py。

    预期 CSV 结构（table3.csv）：
      - 第一列：Settings（如 Seen Original、Seen Confounded、Unseen Original、Unseen Confounded、Average）
      - 其余列：不同比例（如 0%、10%、25%、50%、75%、100%）对应的分数（百分比）
    我们将为每个 Setting 绘制一条折线，过滤掉 Average 行。
    """
    df = pd.read_csv(csv_path)
    if "Settings" not in df.columns:
        raise ValueError(f"CSV 缺少 'Settings' 列: {csv_path}")

    # 过滤 Average 行
    df = df[df["Settings"].astype(str).str.lower() != "average"].copy()

    series_cols = [c for c in df.columns if c != "Settings"]
    if not series_cols:
        raise ValueError(f"CSV 未发现可用的数据列: {csv_path}")

    # x 轴数值与标签
    x_vals = parse_x_from_header(series_cols)
    x_labels = [str(c) for c in series_cols]

    # 定义颜色与线型：Seen 使用蓝色，Unseen 使用红色；Original 实线，Confounded 虚线
    seen_color = "#2E86AB"
    unseen_color = "#C0392B"

    def style_for(name: str):
        n = name.lower()
        color = seen_color if "seen" in n and not "unseen" in n else unseen_color
        linestyle = "-" if "original" in n else "--"
        marker = "o" if "original" in n else "s"
        return color, linestyle, marker

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    make_style(ax)

    # 绘制每条曲线
    for _, row in df.iterrows():
        name = str(row["Settings"]) if not pd.isna(row["Settings"]) else "Series"
        try:
            y = [float(row[c]) for c in series_cols]
        except Exception as e:
            raise ValueError(f"解析数值失败（{name}）: {e}")

        color, linestyle, marker = style_for(name)
        ax.plot(
            x_vals,
            y,
            label=name,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker=marker,
            markersize=5,
            markerfacecolor="#FFFFFF",
            markeredgewidth=1.0,
        )

    # 轴与标签
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels, fontsize=tick_size, fontweight='bold')

    ylabel = determine_ylabel(metric_name)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=axis_label_size)
    ax.tick_params(axis='y', labelsize=tick_size)

    # y 轴范围
    try:
        all_vals = df[series_cols].astype(float).values.flatten()
        vmin = float(pd.Series(all_vals).min())
        vmax = float(pd.Series(all_vals).max())
        margin = max(3.0, 0.1 * (vmax - vmin + 1e-6))
        ax.set_ylim(max(0.0, vmin - margin), vmax + margin)
    except Exception:
        pass

    # 图例样式与边框
    ax.legend(
        frameon=True,
        fontsize=legend_size,
        loc='upper left',
        bbox_to_anchor=(0, 1),
        ncol=2,
        framealpha=0.9,
        fancybox=True,
        shadow=True,
    )

    # 边框微调，与 bar 图一致
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{csv_path.stem}_curve.pdf"
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CARLA-style curve charts from CSV and export PDF (ICRA single column friendly).")
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
        "--metric-name",
        type=str,
        default="success",
        help="用于推断 y 轴标签的指标名，含 'success'、'score' 或 'completion' 关键词",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="覆盖默认标题（默认：CARLA Curve - <csv文件名>）",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[3.5, 2.6],
        help="图形尺寸 (英寸，宽度 高度)，默认 3.5 2.6（ICRA 单栏宽度）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="导出分辨率 DPI，默认 600",
    )
    parser.add_argument("--title-size", type=int, default=14, help="标题字号，默认14")
    parser.add_argument("--axis-label-size", type=int, default=11, help="坐标轴标签字号，默认11")
    parser.add_argument("--tick-size", type=int, default=9, help="刻度字号，默认9")
    parser.add_argument("--legend-size", type=int, default=9, help="图例字号，默认9")
    parser.add_argument("--annot-size", type=int, default=7, help="数值标注字号，默认7")

    args = parser.parse_args()

    csv_paths = [Path(p).expanduser().resolve() for p in args.input]
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    saved: List[Path] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到 CSV: {csv_path}")
        out_dir = output_dir if output_dir else csv_path.parent
        pdf_path = plot_curve_from_csv(
            csv_path=csv_path,
            output_dir=out_dir,
            metric_name=args.metric_name,
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

    for p in saved:
        print(str(p))


if __name__ == "__main__":
    main()
