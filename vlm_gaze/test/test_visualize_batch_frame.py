#!/usr/bin/env python3
"""
Visualize one frame from a batch: original image, preprocessed gaze heatmap, and overlay.

Usage (as a test or standalone):
  - As test: python -m vlm_gaze.test.test_visualize_batch_frame
  - CLI args:
        --hdf5-path      Path to robomimic HDF5 (defaults to the same as other tests)
        --output-dir     Directory to save images (default: /tmp/vlm_gaze_vis)
        --batch-size     Batch size to fetch (default: 4)
        --sample-index   Which sample in batch to visualize (default: 0)
        --frame-index    Which frame index in stack/sequence to visualize (-1 = last)
        --config-name    Which config to load (default: train_gaze)
"""

import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from hydra import compose, initialize_config_dir

from vlm_gaze.train.common.data import build_obs_specs, build_dataset, build_dataloader
from vlm_gaze.data_utils.data_loader_robomimic import GazePreprocessor


def _to_uint8_image(img: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 [H, W, 3] for saving."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def _extract_frame_image(sample_img: torch.Tensor, frame_index: int) -> np.ndarray:
    """
    Extract a single frame image from a per-sample image tensor.
    Supported shapes:
      - [stack, H, W, C]
      - [H, W, C]
      - [C, H, W] or [stack*C, H, W] (falls back best-effort)
    Returns numpy array [H, W, C], float in [0,1] if original was float.
    """
    with torch.no_grad():
        if sample_img.dim() == 4:
            # Likely [stack, H, W, C]
            stack, H, W, C = sample_img.shape
            idx = frame_index if frame_index >= 0 else (stack - 1)
            idx = max(0, min(idx, stack - 1))
            frame = sample_img[idx]
            img = frame
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            return img.cpu().numpy()
        elif sample_img.dim() == 3:
            # Could be [H, W, C] or [C, H, W] or [stack*C, H, W]
            if sample_img.shape[-1] in (1, 3):
                # [H, W, C]
                img = sample_img
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                return img.cpu().numpy()
            else:
                # [C, H, W] or [stack*C, H, W]
                ch = sample_img.shape[0]
                H, W = sample_img.shape[1:]
                # Heuristic: assume 3-channel color if divisible by 3
                if ch % 3 == 0:
                    channels_per_frame = 3
                else:
                    channels_per_frame = 1
                num_frames = ch // channels_per_frame if channels_per_frame > 0 else 1
                idx = frame_index if frame_index >= 0 else (num_frames - 1)
                idx = max(0, min(idx, max(0, num_frames - 1)))
                start = idx * channels_per_frame
                end = start + channels_per_frame
                slice_c = sample_img[start:end]
                if channels_per_frame == 1:
                    slice_c = slice_c.squeeze(0)
                    img = slice_c.permute(1, 2, 0)  # [H, W] -> will expand later
                else:
                    img = slice_c.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                return img.cpu().numpy()
        else:
            # Unexpected, try to squeeze to 2D or 3D
            img = sample_img.squeeze()
            if img.dim() == 2:
                arr = img.cpu().numpy()
                if img.dtype != torch.uint8:
                    arr = np.clip(arr, 0.0, 1.0)
                return arr
            elif img.dim() == 3:
                return _extract_frame_image(img, frame_index)
            raise ValueError(f"Unsupported image tensor shape: {tuple(sample_img.shape)}")


def _prepare_gaze_for_frame(sample_gaze: torch.Tensor, frame_index: int, maxpoints: int) -> torch.Tensor:
    """Prepare gaze coords to shape [1, 1, maxpoints*2] or [1, 1, maxpoints, 2] for the preprocessor."""
    with torch.no_grad():
        if sample_gaze.dim() == 3:
            # [T, P, 2] or [T, P*2]
            T = sample_gaze.shape[0]
            idx = frame_index if frame_index >= 0 else (T - 1)
            idx = max(0, min(idx, T - 1))
            g = sample_gaze[idx]
            if g.dim() == 2 and g.shape[-1] == 2:
                g = g.unsqueeze(0).unsqueeze(0)  # [1,1,P,2]
            else:
                g = g.view(1, 1, maxpoints * 2)  # [1,1,P*2]
            return g
        elif sample_gaze.dim() == 2:
            # [T, P*2]
            T = sample_gaze.shape[0]
            idx = frame_index if frame_index >= 0 else (T - 1)
            idx = max(0, min(idx, T - 1))
            g = sample_gaze[idx].view(1, 1, maxpoints * 2)
            return g
        elif sample_gaze.dim() == 1:
            # [P*2]
            return sample_gaze.view(1, 1, maxpoints * 2)
        else:
            raise ValueError(f"Unsupported gaze tensor shape: {tuple(sample_gaze.shape)}")


def visualize_one_batch_frame(
    hdf5_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 4,
    sample_index: int = 0,
    frame_index: int = -1,
    config_name: str = 'train_gaze',
) -> bool:
    """
    Fetch one batch from the dataset and visualize one frame: original image, gaze heatmap, overlay.
    Returns True on success.
    """
    try:
        # Resolve defaults
        default_hdf5 = "/data3/vla-reasoning/dataset/bench2drive220_robomimic_large_chunk.hdf5"
        hdf5_path = hdf5_path or os.environ.get("HDF5_PATH", default_hdf5)
        out_dir = Path(output_dir or "/tmp/vlm_gaze_vis")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        config_dir = Path('/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/vlm_gaze/configs').absolute()
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=config_name)
        # Override minimal fields
        cfg.data.hdf5_path = hdf5_path
        cfg.data.batch_size = batch_size
        cfg.data.num_workers = 0
        cfg.data.cache_mode = getattr(cfg.data, 'cache_mode', 'low_dim')

        # Build dataset and loader
        build_obs_specs()
        dataset = build_dataset(cfg.data)
        loader = build_dataloader(dataset, cfg.data, sampler=None, grad_accum_steps=1)

        # Fetch one batch
        data_iter = iter(loader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # Select sample
        bsz = batch['obs']['image'].shape[0]
        sidx = max(0, min(sample_index, bsz - 1))

        # Extract image tensor for the sample
        img_tensor = batch['obs']['image'][sidx]
        # If stacked, try to get shape [stack, H, W, C] for robust extraction
        # Convert [stack, H, W, C] if needed
        if img_tensor.dim() == 5:
            # [T, stack, H, W, C] -> choose last T then proceed
            img_tensor = img_tensor[-1]
        img_np = _extract_frame_image(img_tensor, frame_index)

        # Prepare gaze coords for this frame
        gaze_tensor = batch['obs']['gaze_coords'][sidx]
        gaze_maxpoints = int(getattr(cfg.gaze, 'max_points', 5))
        gaze_for_pre = _prepare_gaze_for_frame(gaze_tensor, frame_index, gaze_maxpoints)

        # Create heatmap via preprocessor (CPU by default)
        H = int(getattr(cfg.data, 'img_height', img_np.shape[0]))
        W = int(getattr(cfg.data, 'img_width', img_np.shape[1]))
        pre = GazePreprocessor(
            img_height=H,
            img_width=W,
            gaze_sigma=float(getattr(cfg.gaze, 'mask_sigma', 30.0)),
            gaze_coeff=float(getattr(cfg.gaze, 'mask_coeff', 0.8)),
            maxpoints=gaze_maxpoints,
            device='cpu'
        )
        heatmap = pre(gaze_for_pre).squeeze().detach().cpu().numpy()
        # heatmap shape could be [H, W] or [1, H, W]; squeeze handles it

        # Normalize image to [0,1] float
        if img_np.dtype == np.uint8:
            img_float = img_np.astype(np.float32) / 255.0
        else:
            img_float = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        if img_float.ndim == 2:
            img_float = np.stack([img_float, img_float, img_float], axis=-1)
        if img_float.shape[-1] == 1:
            img_float = np.repeat(img_float, 3, axis=-1)

        # Colorize heatmap using matplotlib
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('jet')
            heat_color = cmap(np.clip(heatmap, 0.0, 1.0))[:, :, :3]  # [H, W, 3]
        except Exception:
            # Fallback: grayscale to RGB
            heat_color = np.stack([heatmap, heatmap, heatmap], axis=-1)

        # Resize heat_color if needed
        if heat_color.shape[:2] != img_float.shape[:2]:
            # simple nearest resize via torch (avoid cv2)
            th = torch.from_numpy(heat_color).permute(2, 0, 1).unsqueeze(0)
            th = torch.nn.functional.interpolate(th, size=img_float.shape[:2], mode='bilinear', align_corners=False)
            heat_color = th.squeeze(0).permute(1, 2, 0).numpy()

        # Blend overlay
        alpha = 0.4
        overlay = np.clip((1.0 - alpha) * img_float + alpha * heat_color, 0.0, 1.0)

        # Save outputs
        fi = frame_index if frame_index >= 0 else -1
        base = f"sample{sidx}_frame{fi}"
        out_img = out_dir / f"{base}_image.png"
        out_heat = out_dir / f"{base}_heatmap.png"
        out_overlay = out_dir / f"{base}_overlay.png"

        from imageio import imwrite
        imwrite(out_img, _to_uint8_image(img_float))
        imwrite(out_heat, _to_uint8_image(heat_color))
        imwrite(out_overlay, _to_uint8_image(overlay))

        print(f"Saved:\n  {out_img}\n  {out_heat}\n  {out_overlay}")
        return True
    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualize_one_batch_frame() -> bool:
    """Pytest-style entry for CI/manual run."""
    return visualize_one_batch_frame()


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Visualize one frame from a batch (image, gaze heatmap, overlay).')
    p.add_argument('--hdf5-path', type=str, default=None, help='Path to robomimic HDF5 dataset')
    p.add_argument('--output-dir', type=str, default='./vlm_gaze_vis', help='Directory to save visualizations')
    p.add_argument('--batch-size', type=int, default=4, help='Batch size')
    p.add_argument('--sample-index', type=int, default=0, help='Sample index within the batch')
    p.add_argument('--frame-index', type=int, default=-1, help='Frame index within stack/sequence (-1 = last)')
    p.add_argument('--config-name', type=str, default='train_gaze', help='Config name to load')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    ok = visualize_one_batch_frame(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        sample_index=args.sample_index,
        frame_index=args.frame_index,
        config_name=args.config_name,
    )
    import sys
    sys.exit(0 if ok else 1)


