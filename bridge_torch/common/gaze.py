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

    Args:
        spatial_features: Tensor of shape (B, C, H, W), encoder spatial activations.
        beta: Sharpness/temperature parameter (>0). Values >1 sharpen, <1 smooth.
        out_hw: Optional (H_out, W_out). If provided, result is resized to this size.

    Returns:
        Tensor of shape (B, 1, H*, W*) with values in [0, 1].
    """
    if not isinstance(spatial_features, torch.Tensor):
        raise TypeError("spatial_features must be a torch.Tensor")
    if spatial_features.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(spatial_features.shape)}")

    # Aggregate channel responses to a single-channel importance map per sample
    # Use mean absolute activation as a simple, stable proxy
    sal = spatial_features.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)

    # Normalize per-sample to [0,1]
    mn = sal.amin(dim=(-2, -1), keepdim=True)
    mx = sal.amax(dim=(-2, -1), keepdim=True)
    sal = (sal - mn) / (mx - mn + 1e-8)

    # Apply sharpness control
    try:
        b = float(beta)
    except Exception:
        b = 1.0
    if b > 0 and abs(b - 1.0) > 1e-6:
        sal = sal.pow(b)

    # Optional resize to target spatial size
    if out_hw is not None:
        H, W = int(out_hw[0]), int(out_hw[1])
        if H > 0 and W > 0 and (H != sal.shape[-2] or W != sal.shape[-1]):
            sal = F.interpolate(sal, size=(H, W), mode="bilinear", align_corners=False)

    # Ensure [0,1]
    sal = sal.clamp_(0.0, 1.0)
    return sal


