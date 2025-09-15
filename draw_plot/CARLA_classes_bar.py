import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
# Match the general styling used in CARLA_bar.py
plt.rcParams['font.family'] = 'Times New Roman'

# 可配置的排除类别列表 - 在此处添加不想显示的类别
EXCLUDE_CATEGORIES = [
    "vehicles",
    "pedestrian",
    "streetlights",
    "fences",
    "lane markings pedestrian crosswalk",
    "vehicle bicyclist",
    "street light",
    "road sign",
    "traffic sign"
]
def make_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#F8F8F8")
    ax.yaxis.grid(True, which="major", linestyle="-", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
def load_summary(summary_path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    overall = data.get('overall', {})
    det = overall.get('detection_classes', {}) or {}
    vlm = overall.get('vlm_filtered_classes', {}) or {}
    # Coerce keys to str and values to int
    det = {str(k): int(v) for k, v in det.items()}
    vlm = {str(k): int(v) for k, v in vlm.items()}
    return det, vlm
def pick_top_classes(det_classes: Dict[str, int], vlm_classes: Dict[str, int], top_k: int) -> List[str]:
    # Sort by detection count desc, tie-break by name
    # 先选出top k个类别
    items = sorted(det_classes.items(), key=lambda x: (-x[1], x[0]))
    top_k_classes = [k for k, _ in items[:top_k]]
    
    # 然后从top k中排除指定的类别
    exclude_lower = [exc.lower() for exc in EXCLUDE_CATEGORIES]
    filtered_classes = [cls for cls in top_k_classes if cls.lower() not in exclude_lower]
    
    return filtered_classes
def plot_class_bars(
    det_counts: Dict[str, int],
    vlm_counts: Dict[str, int],
    classes: List[str],
    out_path: Path,
    title: str = "Class-wise Detections vs VLM",
    figsize: Tuple[float, float] = (14, 4.6),
    dpi: int = 300,
    detection_color: str = "#87CEEB",  # light blue (sky blue)
    vlm_color: str = "#FFA07A",        # light red (light salmon)
    annot: bool = False,
    bar_width: float = 0.35,
    bar_gap: float = 0.05,
    left_margin: float = 0.4,
) -> None:
    n = len(classes)
    x = np.arange(n, dtype=float)
    
    # 使用相同位置让bar重叠，通过bar_gap控制各组之间的间距
    x_pos = x * (bar_width + bar_gap) + left_margin
    
    det_vals = np.array([det_counts.get(c, 0) for c in classes], dtype=float)
    vlm_vals = np.array([vlm_counts.get(c, 0) for c in classes], dtype=float)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    make_style(ax)
    
    # Detection bars: 浅蓝色，作为底层
    ax.bar(
        x_pos,
        det_vals,
        bar_width,
        color=detection_color,
        edgecolor="none",  # 移除边框
        linewidth=0,
        label="Detections",
        zorder=2,
    )
    # VLM bars: 浅红色，重叠在上层
    bars_vlm = ax.bar(
        x_pos,
        vlm_vals,
        bar_width,
        color=vlm_color,
        edgecolor="none",  # 移除边框
        linewidth=0,
        label="VLM Filtered",
        zorder=3,
    )
    # Optional annotations atop overlay bars (less clutter)
    if annot:
        for rect, val in zip(bars_vlm.patches, vlm_vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + max(det_vals.max() * 0.005, 50),
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    # 设置x轴刻度在每组bar的中心
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, rotation=50, ha='right', fontsize=10)
    ax.set_ylabel("Count",fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(title, fontsize=15, pad=14)
    # Y-limits with headroom
    ymax = float(max(det_vals.max(), vlm_vals.max()) * 1.15 + 10)
    ax.set_ylim(0, ymax)
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, fontsize=10, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=dpi)
def main():
    parser = argparse.ArgumentParser(description="Plot top-K class-wise detections vs VLM-filtered from summary JSON.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("analysis_summary.json"),
        help="Path to analysis_summary.json",
    )
    parser.add_argument(
        "--per-route",
        type=Path,
        default=Path("per_route_seed_counts.csv"),
        help="Path to per_route_seed_counts.csv (optional, not required)",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Top-K classes by detection count")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/class_counts_overlay.pdf"),
        help="Output PDF/PNG path",
    )
    parser.add_argument("--annot", action="store_true", help="Annotate VLM counts above bars")
    parser.add_argument("--bar-width", type=float, default=0.35, help="Width of each bar (default: 0.35)")
    parser.add_argument("--bar-gap", type=float, default=0.05, help="Gap between detection and VLM bars (default: 0.05)")
    parser.add_argument("--left-margin", type=float, default=0.4, help="Left margin from y-axis to first bar (default: 0.4)")
    parser.add_argument("--detection-color", type=str, default="#87CEEB", help="Color for detection bars (default: #87CEEB)")
    parser.add_argument("--vlm-color", type=str, default="#FFA07A", help="Color for VLM bars (default: #FFA07A)")
    
    args = parser.parse_args()
    det, vlm = load_summary(args.summary)
    classes = pick_top_classes(det, vlm, args.top_k)
    title = "Top {} Classes: Detections vs VLM".format(len(classes))
    plot_class_bars(
        det, 
        vlm, 
        classes, 
        args.output, 
        title=title, 
        annot=args.annot,
        bar_width=args.bar_width,
        bar_gap=args.bar_gap,
        left_margin=args.left_margin,
        detection_color=args.detection_color,
        vlm_color=args.vlm_color
    )
if __name__ == "__main__":
    main()
