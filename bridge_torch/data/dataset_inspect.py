from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch

# Allow running both as a module (python -m bridge_torch.data.dataset_inspect)
# and as a standalone script (python bridge_torch/data/dataset_inspect.py)
try:
    from .bridge_numpy import BridgeNumpyDataset  # type: ignore
except Exception:
    import sys as _sys, os as _os
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    from bridge_numpy import BridgeNumpyDataset  # type: ignore


def _coerce_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            cand = os.path.join(p, "out.npy")
            if os.path.isfile(cand):
                out.append(cand)
        elif os.path.isfile(p) and p.endswith(".npy"):
            out.append(p)
    if not out:
        raise FileNotFoundError("No valid .npy paths found from --paths arguments")
    return out


def _load_action_proprio_meta(json_path: str | None) -> Dict[str, Any] | None:
    if not json_path:
        return None
    p = os.path.expanduser(json_path)
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Expected structure:
    # {"action": {"mean": [...], "std": [...]}, "proprio": {"mean": [...], "std": [...]}}
    return meta


def summarize_batch(batch: Dict[str, Any]) -> None:
    def _tinfo(name: str, t: torch.Tensor):
        print(f"- {name}: torch.Tensor, shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
        try:
            with torch.no_grad():
                mn = float(t.min().item())
                mx = float(t.max().item())
            print(f"  range: [{mn:.6g}, {mx:.6g}]")
        except Exception:
            pass

    print("Batch summary:")
    obs = batch.get("observations", {})
    goals = batch.get("goals", {})
    acts = batch.get("actions", None)

    if isinstance(obs, dict):
        img = obs.get("image")
        if isinstance(img, torch.Tensor):
            _tinfo("observations.image", img)
        prop = obs.get("proprio")
        if isinstance(prop, torch.Tensor):
            _tinfo("observations.proprio", prop)
        sal = obs.get("saliency")
        if isinstance(sal, torch.Tensor):
            _tinfo("observations.saliency", sal)

    if isinstance(goals, dict):
        gimg = goals.get("image")
        if isinstance(gimg, torch.Tensor):
            _tinfo("goals.image", gimg)

    if isinstance(acts, torch.Tensor):
        _tinfo("actions", acts)
    elif isinstance(acts, np.ndarray):
        print(f"- actions: np.ndarray, shape={acts.shape}, dtype={acts.dtype}")
        try:
            print(f"  range: [{acts.min():.6g}, {acts.max():.6g}]")
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(
        description="Initialize BridgeNumpyDataset and fetch a single batch for inspection",
    )
    p.add_argument("--paths", nargs="+", default=["/data3/vla-reasoning/test_dataset/bdv2_numpy/lift_carrot_100/train"],
                   help="One or more directories containing out.npy or direct out.npy paths")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", action="store_true", help="Enable train mode (affects augmentation)")
    p.add_argument("--obs_horizon", type=int, default=1)
    p.add_argument("--act_pred_horizon", type=int, default=1)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--augment_next_obs_goal_differently", action="store_true")
    p.add_argument("--saliency_alpha", type=float, default=1.0)
    p.add_argument("--action_proprio_meta", type=str, default="/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/bridge_torch/data/amp.json",
                   help="Optional JSON with action/proprio mean/std (proprio-only normalization used)")

    args = p.parse_args()

    npy_paths = _coerce_paths(args.paths)
    apm = _load_action_proprio_meta(args.action_proprio_meta)

    ds = BridgeNumpyDataset(
        data_paths=npy_paths,
        seed=args.seed,
        batch_size=args.batch_size,
        train=bool(args.train),
        goal_relabeling_strategy="uniform",
        goal_relabeling_kwargs=None,
        relabel_actions=True,
        shuffle_buffer_size=25000,
        augment=bool(args.augment),
        augment_next_obs_goal_differently=bool(args.augment_next_obs_goal_differently),
        act_pred_horizon=int(args.act_pred_horizon),
        obs_horizon=int(args.obs_horizon),
        augment_kwargs=None,
        load_language=False,
        load_gaze=False,
        action_proprio_metadata=apm,
        sample_weights=None,
        saliency_alpha=float(args.saliency_alpha),
    )

    it = ds.iterator()
    batch = next(it)
    summarize_batch(batch)


if __name__ == "__main__":
    main()
