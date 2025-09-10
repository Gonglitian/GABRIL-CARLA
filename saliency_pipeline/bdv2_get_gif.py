"""
BDV2 Trajectory GIF Generator

Generate a GIF from raw image frames (images{camera}/im_*.jpg) for a single
trajectory in the BridgeDataset v2-style dataset.

Configuration is read from a YAML file in `refactor/configs`.

Example config (bdv2_gif_config.yaml):

data:
  dataset_root: "/data3/vla-reasoning/dataset/bdv2"
  task: "open_microwave"
  timestamp: "2022-03-12_14-48-28"
  traj_group: "traj_group0"
  traj_name: "traj0"
  camera: 0            # 0/1/2 or "all" to export each camera separately
  frame_glob: "im_*.jpg"
  frame_step: 1
  frame_range: { start: null, end: null }  # optional

processing:
  resize: { width: 640, height: 360 }      # optional; remove to keep original
  grayscale: false                         # optional

output:
  out_dir: "refactor/bdv2_inspect/gifs"
  filename: "{task}_{timestamp}_{traj}_{cam}.gif"  # placeholders: task,timestamp,traj,cam
  fps: 10

Usage:
  python -m refactor.bdv2_get_gif --config refactor/configs/bdv2_gif_config.yaml
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import imageio  # type: ignore
    HAS_IMAGEIO = True
except Exception:
    imageio = None  # type: ignore
    HAS_IMAGEIO = False
from PIL import Image
import numpy as np
import yaml


def _frame_index_from_name(name: str) -> int:
    """Extract numeric frame index from filename like im_10.jpg -> 10.

    Returns -1 if parsing fails. We sort primarily by this index to avoid
    lexicographic mis-order like im_1, im_10, im_11, ..., im_2.
    """
    base = os.path.splitext(os.path.basename(name))[0]
    try:
        return int(base.split("_")[-1])
    except Exception:
        return -1


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).parent / "configs" / "bdv2_gif_config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_frames(images_dir: Path, glob_pat: str) -> List[Path]:
    frames = list(images_dir.glob(glob_pat))
    frames.sort(key=lambda p: (_frame_index_from_name(p.name), p.name))
    return frames


def maybe_resize(img: Image.Image, size: Optional[Dict[str, int]]) -> Image.Image:
    if not size:
        return img
    w = size.get("width")
    h = size.get("height")
    if w and h:
        return img.resize((int(w), int(h)), Image.BILINEAR)
    return img


def maybe_grayscale(img: Image.Image, enable: bool) -> Image.Image:
    if enable:
        return img.convert("L").convert("RGB")
    return img


def _quantize_with_global_palette(images: List[Image.Image]) -> List[Image.Image]:
    """Quantize frames to a single global palette to avoid color flicker.

    Uses the first frame to build an adaptive palette, then maps others onto it.
    """
    if not images:
        return images
    # First frame adaptive palette
    base_p = images[0].quantize(colors=256, method=Image.MEDIANCUT)
    pal = base_p.getpalette()
    out = [base_p]
    for im in images[1:]:
        # Map using the same palette to keep colors consistent across frames
        im_p = im.quantize(palette=base_p)
        # Ensure palette is applied (safeguard)
        im_p.putpalette(pal)
        out.append(im_p)
    return out


def frames_to_gif(frames: List[Path], out_path: Path, fps: int, resize: Optional[Dict[str, int]], grayscale: bool) -> bool:
    if not frames:
        print(f"   ⚠️  No frames found, skip: {out_path}")
        return False
    images = []
    for fp in frames:
        try:
            im = Image.open(fp).convert("RGB")
            im = maybe_grayscale(im, grayscale)
            im = maybe_resize(im, resize)
            images.append(im)
        except Exception as e:
            print(f"   ⚠️  Failed to read frame {fp}: {e}")
    if not images:
        print(f"   ⚠️  No readable frames, skip: {out_path}")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_IMAGEIO:
        try:
            # imageio expects numpy arrays
            imageio.mimsave(out_path, [np.array(im) for im in images], fps=fps)
            print(f"   ✅ Saved GIF: {out_path}")
            return True
        except Exception:
            pass  # fall through to PIL

    # Fallback using PIL's save
    try:
        # Reduce flicker: use a single global palette and explicit disposal
        pal_images = _quantize_with_global_palette(images)
        pal_images[0].save(
            out_path,
            save_all=True,
            append_images=pal_images[1:],
            duration=int(1000 / max(fps, 1)),
            loop=0,
            optimize=False,
            disposal=2,  # full replace each frame
        )
        print(f"   ✅ Saved GIF (PIL): {out_path}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to save GIF: {e}")
        return False


def build_traj_path(cfg: Dict[str, Any], camera: Union[int, str]) -> Tuple[Path, str]:
    data = cfg["data"]
    dataset_root = Path(data["dataset_root"]).expanduser()
    task = data["task"]
    timestamp = data["timestamp"]
    traj_group = data["traj_group"]
    traj_name = data["traj_name"]
    cam_name = f"images{camera}" if isinstance(camera, int) else str(camera)
    traj_path = dataset_root / task / timestamp / "raw" / traj_group / traj_name / cam_name
    return traj_path, cam_name


def run(cfg: Dict[str, Any]) -> None:
    data = cfg["data"]
    out_cfg = cfg.get("output", {})
    proc_cfg = cfg.get("processing", {})

    camera = data.get("camera", 0)
    frame_glob = data.get("frame_glob", "im_*.jpg")
    frame_step = int(data.get("frame_step", 1))
    frame_range = data.get("frame_range", {}) or {}
    start_idx = frame_range.get("start")
    end_idx = frame_range.get("end")

    # Resolve which cameras to process
    cameras: List[Union[int, str]]
    if isinstance(camera, str) and camera.lower() == "all":
        cameras = [0, 1, 2]
    else:
        cameras = [camera]

    out_dir = Path(out_cfg.get("out_dir", Path(__file__).parent / "bdv2_inspect" / "gifs"))
    fps = int(out_cfg.get("fps", 10))
    resize = proc_cfg.get("resize")
    grayscale = bool(proc_cfg.get("grayscale", False))

    filename_tmpl = out_cfg.get("filename", "{task}_{timestamp}_{traj}_{cam}.gif")
    for cam in cameras:
        images_dir, cam_name = build_traj_path(cfg, cam)
        task = data["task"]
        timestamp = data["timestamp"]
        traj = data["traj_name"]

        if not images_dir.exists():
            print(f"   ⚠️  Images directory not found: {images_dir}")
            continue

        all_frames = list_frames(images_dir, frame_glob)
        if start_idx is not None or end_idx is not None:
            # Filter by numeric suffix based on filename pattern im_{idx}.xxx
            all_frames = [p for p in all_frames if (start_idx is None or _frame_index_from_name(p.name) >= start_idx) and (end_idx is None or _frame_index_from_name(p.name) <= end_idx)]

        frames = all_frames[::max(1, frame_step)]
        out_name = filename_tmpl.format(task=task, timestamp=timestamp, traj=traj, cam=cam_name)
        out_path = out_dir / out_name
        print(f"Creating GIF for {images_dir} -> {out_path} (frames={len(frames)}, fps={fps})")
        frames_to_gif(frames, out_path, fps=fps, resize=resize, grayscale=grayscale)


def main():  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Generate GIF from BDV2 trajectory raw images")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "bdv2_gif_config.yaml"), help="YAML config path")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
