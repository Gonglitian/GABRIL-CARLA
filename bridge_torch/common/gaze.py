from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def get_gaze_mask(
    spatial_features: torch.Tensor,
    beta: float = 1.0,
    out_hw: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """Compute a saliency map from last spatial feature maps.

    对齐 vlm_gaze 的实现：通道求绝对值和 → 以 beta 为温度的 softmax → 上采样 → 归一化到 [0,1]。

    Args:
        spatial_features: (B, C, H, W) encoder 空间激活
        beta: softmax 温度（越小越尖锐；与对侧一致）
        out_hw: 可选输出尺寸 (H_out, W_out)

    Returns:
        (B, 1, H*, W*)，范围 [0,1]
    """
    if not isinstance(spatial_features, torch.Tensor):
        raise TypeError("spatial_features must be a torch.Tensor")
    if spatial_features.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(spatial_features.shape)}")

    # 1) 通道聚合（与 vlm_gaze 相同：abs 后在通道维求和）
    z = spatial_features.abs().sum(dim=1)  # (B, H, W)

    # 2) 温度化 softmax（按像素展平后做 softmax，再还原形状）
    B, Hf, Wf = z.shape
    zf = z.view(B, Hf * Wf)
    # 与对侧一致：使用 z / beta 作为温度化 softmax 输入
    beta_safe = float(beta) if isinstance(beta, (int, float)) else 1.0
    zf = torch.nn.functional.softmax(zf / max(beta_safe, 1e-8), dim=-1)
    sal = zf.view(B, 1, Hf, Wf)  # (B,1,Hf,Wf)

    # 3) 上采样到目标分辨率（若指定）
    if out_hw is not None:
        Ho, Wo = int(out_hw[0]), int(out_hw[1])
        if Ho > 0 and Wo > 0 and (Ho != Hf or Wo != Wf):
            sal = F.interpolate(sal, size=(Ho, Wo), mode="bicubic", align_corners=False)

    # 4) 再次做稳健归一化到 [0,1]
    flat = sal.view(B, 1, -1)
    mx = flat.max(dim=-1).values.view(B, 1, 1, 1)
    mn = flat.min(dim=-1).values.view(B, 1, 1, 1)
    sal = (sal - mn) / (mx - mn + 1e-8)
    sal = sal.clamp_(0.0, 1.0)
    return sal


