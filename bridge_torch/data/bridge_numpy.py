from __future__ import annotations

import os
import sys
import random
from typing import Dict, Iterator, List, Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, IterableDataset, get_worker_info
import albumentations as A
import cv2
import ast


def _hwc_uint8_to_chw_float(x: np.ndarray) -> torch.Tensor:
    if x.ndim == 4:
        t, h, w, c = x.shape
        x = x.transpose(0, 3, 1, 2).reshape(c * t, h, w)
    else:
        x = x.transpose(2, 0, 1)
    return torch.from_numpy(x.astype(np.float32) / 255.0)


def _flatten_time_feat(x: np.ndarray) -> torch.Tensor:
    if x.ndim == 2:
        return torch.from_numpy(x.astype(np.float32))
    elif x.ndim == 3:
        b, t, p = x.shape
        return torch.from_numpy(x.reshape(b, t * p).astype(np.float32))
    else:
        b = x.shape[0]
        return torch.from_numpy(x.reshape(b, -1).astype(np.float32))


class BridgeNumpyDataset:
    def __init__(
        self,
        data_paths: List[str],
        seed: int,
        batch_size: int,
        train: bool,
        goal_relabeling_strategy: str = "uniform",
        goal_relabeling_kwargs: dict | None = None,
        relabel_actions: bool = True,
        shuffle_buffer_size: int = 25000,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        act_pred_horizon: int | None = None,
        obs_horizon: int | None = None,
        augment_kwargs: dict | None = None,
        load_language: bool = False,
        load_gaze: bool = False,
        action_proprio_metadata: dict | None = None,
        sample_weights: List[float] | None = None,
        saliency_alpha: float | None = None,
        **kwargs,
    ):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._base_seed = int(seed)
        self.batch_size = int(batch_size)
        self.train = bool(train)
        self.relabel_actions = bool(relabel_actions)
        self.goal_strategy = goal_relabeling_strategy or "uniform"
        self.goal_kwargs = goal_relabeling_kwargs or {}
        # horizons/configs
        self.act_pred_horizon = int(act_pred_horizon) if act_pred_horizon is not None else None
        self.obs_horizon = int(obs_horizon) if obs_horizon is not None else None
        self.saliency_alpha = float(saliency_alpha) if saliency_alpha is not None else 1.0
        # augmentation configs
        self.augment = bool(augment)
        self.augment_next_obs_goal_differently = bool(augment_next_obs_goal_differently)
        # 尝试将 CLI 传入的字符串形式（例如 "[0.8,1.0]"、"{...}"）转为 Python 对象
        def _canon(v):
            if isinstance(v, str):
                s = v.strip()
                if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                    try:
                        return ast.literal_eval(s)
                    except Exception:
                        return v
            return v
        def _canon_dict(d):
            if not isinstance(d, dict):
                return d
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = _canon_dict(v)
                elif isinstance(v, list):
                    out[k] = [ _canon(x) for x in v ]
                else:
                    out[k] = _canon(v)
            return out
        self.augment_kwargs = _canon_dict(augment_kwargs or {})
        self.apm = action_proprio_metadata
        # 解析标准化统计量（若提供）
        self._act_mean = None
        self._act_std = None
        self._prop_mean = None
        self._prop_std = None
        try:
            if isinstance(self.apm, dict):
                act_meta = dict(self.apm.get("action", {}) or {})
                prop_meta = dict(self.apm.get("proprio", {}) or {})
                if ("mean" in act_meta) and ("std" in act_meta):
                    self._act_mean = np.asarray(act_meta["mean"], dtype=np.float32)
                    self._act_std = np.asarray(act_meta["std"], dtype=np.float32)
                if ("mean" in prop_meta) and ("std" in prop_meta):
                    self._prop_mean = np.asarray(prop_meta["mean"], dtype=np.float32)
                    self._prop_std = np.asarray(prop_meta["std"], dtype=np.float32)
        except Exception:
            self._act_mean = None
            self._act_std = None
            self._prop_mean = None
            self._prop_std = None

        # Load trajectories from all paths. Each path is a directory like <task>/<split>/out.npy
        self.trajs: List[dict] = []
        for p in data_paths:
            npy = os.path.join(p)
            with open(npy, "rb") as f:
                trajs = np.load(f, allow_pickle=True)
            self.trajs.extend(list(trajs))

        # Precompute per-trajectory lengths
        self._lengths = [len(t["actions"]) for t in self.trajs]
        self._cum = np.cumsum([0] + self._lengths)

    def reseed(self, seed: int) -> None:
        """Reset RNGs for multi-worker settings."""
        self.rng = random.Random(int(seed))
        self.np_rng = np.random.RandomState(int(seed))

    def _valid_t_bounds(self, T: int) -> Tuple[int, int]:
        """Compute valid inclusive [lo, hi] for sampling timestep t.

        Ensures there are enough past frames for obs_horizon and enough
        future steps for act_pred_horizon if configured.
        """
        past = max(0, (self.obs_horizon or 1) - 1)
        fut = max(0, (self.act_pred_horizon or 1) - 1)
        lo = past
        hi = max(past, T - 1 - fut)
        return lo, hi

    def _sample_index(self) -> Tuple[int, int]:
        # Sample a trajectory and a timestep with horizon constraints
        idx = self.rng.randrange(len(self.trajs))
        Tlen = self._lengths[idx]
        lo, hi = self._valid_t_bounds(Tlen)
        if hi < lo:
            # degenerate (short trajectory); fallback to any timestep
            t = self.rng.randrange(Tlen)
        else:
            t = self.rng.randrange(lo, hi + 1)
        return idx, t

    def _get_transition(self, traj: dict, t: int) -> dict:
        # Observations
        obs = traj["observations"][t]
        next_obs = traj["next_observations"][t]
        act = np.array(traj["actions"][t], dtype=np.float32)
        # horizons
        obs_h = max(1, (self.obs_horizon or 1))
        act_h = max(1, (self.act_pred_horizon or 1))

        # Build goal according to strategy (uniform future by default)
        if self.goal_strategy == "uniform":
            T = len(traj["observations"]) - 1
            g_t = self.rng.randrange(t, T + 1)
            goal_img = traj["next_observations"][g_t]["images0"]
        else:
            # fallback: use next obs as goal
            goal_img = next_obs["images0"]

        # Build observation with horizon: collect last obs_h frames [t-obs_h+1, ..., t]
        if obs_h > 1:
            t0 = max(0, t - (obs_h - 1))
            obs_frames = [traj["observations"][k]["images0"] for k in range(t0, t + 1)]
            obs_img = np.stack(obs_frames, axis=0)  # (T,H,W,C)
            # proprio sequence if available
            prop_seq = None
            if obs.get("state", None) is not None:
                prop_seq = np.stack([traj["observations"][k].get("state", None) for k in range(t0, t + 1)], axis=0)
        else:
            obs_img = obs["images0"]
            prop_seq = obs.get("state", None)

        # Build action horizon (sequence for DDPM if >1)
        if act_h > 1:
            a_T = len(traj["actions"])  # same as len(observations)
            t_end = min(a_T - 1, t + (act_h - 1))
            acts = np.stack([traj["actions"][k] for k in range(t, t_end + 1)], axis=0).astype(np.float32)
        else:
            acts = act

        # # 归一化 proprio（若存在）
        # if prop_seq is not None and (self._prop_mean is not None) and (self._prop_std is not None):
        #     try:
        #         prop_seq = (prop_seq - self._prop_mean) / (self._prop_std + 1e-8)
        #     except Exception:
        #         try:
        #             prop_seq = (prop_seq - np.asarray(self._prop_mean, dtype=np.float32)) / (np.asarray(self._prop_std, dtype=np.float32) + 1e-8)
        #         except Exception:
        #             pass
        next_prop = next_obs.get("state", None)
        # if next_prop is not None and (self._prop_mean is not None) and (self._prop_std is not None):
        #     try:
        #         next_prop = (np.asarray(next_prop, dtype=np.float32) - self._prop_mean) / (self._prop_std + 1e-8)
        #     except Exception:
        #         try:
        #             next_prop = (np.asarray(next_prop, dtype=np.float32) - np.asarray(self._prop_mean, dtype=np.float32)) / (np.asarray(self._prop_std, dtype=np.float32) + 1e-8)
        #         except Exception:
        #             pass

        out = {
            "observations": {
                "image": obs_img,
                "proprio": prop_seq,
            },
            "next_observations": {
                "image": next_obs["images0"],
                "proprio": next_prop,
            },
            "goals": {
                "image": goal_img,
            },
            "actions": acts,
        }
        # Optional: attach saliency (HWC float32, C=1). When obs_horizon>1, build causal aggregation with alpha^k
        if "saliency" in obs:
            if obs_h > 1:
                t0 = max(0, t - (obs_h - 1))
                alpha = float(self.saliency_alpha)
                # accumulate s(t) + alpha^1*s(t-1) + ...
                agg = None
                for k in range(t, t0 - 1, -1):
                    d = t - k
                    w = (alpha ** d) if d > 0 else 1.0
                    s = traj["observations"][k].get("saliency", None)
                    if s is None:
                        continue
                    cur = (w * s).astype(np.float32)
                    agg = cur if agg is None else (agg + cur)
                if agg is None:
                    agg = obs["saliency"].astype(np.float32)
                # normalize to [0,1]
                mn = float(agg.min())
                mx = float(agg.max())
                if mx > mn:
                    agg = (agg - mn) / (mx - mn)
                else:
                    agg = np.zeros_like(agg, dtype=np.float32)
                out["observations"]["saliency"] = agg
            else:
                # even for single frame, ensure normalization
                s = obs["saliency"].astype(np.float32)
                mn = float(s.min())
                mx = float(s.max())
                if mx > mn:
                    s = (s - mn) / (mx - mn)
                else:
                    s = np.zeros_like(s, dtype=np.float32)
                out["observations"]["saliency"] = s
        return out

    # --------- augmentation helpers (Albumentations) ---------
    @staticmethod
    def _to_pair(v, symm=False, around1=False):
        if isinstance(v, (list, tuple)):
            if len(v) == 1:
                if symm:
                    a = -float(v[0]); b = float(v[0])
                    return (a, b)
                if around1:
                    a = 1.0 - float(v[0]); b = 1.0 + float(v[0])
                    return (a, b)
                return (0.0, float(v[0]))
            return (float(v[0]), float(v[1]))
        return (float(v), float(v))

    def _build_albu(self, H: int, W: int) -> A.ReplayCompose:
        ak = self.augment_kwargs or {}
        order = ak.get(
            "augment_order",
            [
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        )
        ops: List[A.BasicTransform] = []
        for op in order:
            if op == "random_resized_crop" and ("random_resized_crop" in ak):
                rrc = ak["random_resized_crop"]
                scale = tuple(rrc.get("scale", [0.8, 1.0]))
                ratio = tuple(rrc.get("ratio", [0.9, 1.1]))
                ops.append(A.RandomResizedCrop(height=H, width=W, scale=scale, ratio=ratio, interpolation=cv2.INTER_LINEAR, p=1.0))
            elif op == "random_brightness" and ("random_brightness" in ak):
                br = self._to_pair(ak["random_brightness"], around1=True)
                ops.append(A.ColorJitter(brightness=br, contrast=0, saturation=0, hue=0, p=1.0))
            elif op == "random_contrast" and ("random_contrast" in ak):
                ct = self._to_pair(ak["random_contrast"], around1=True)
                ops.append(A.ColorJitter(brightness=0, contrast=ct, saturation=0, hue=0, p=1.0))
            elif op == "random_saturation" and ("random_saturation" in ak):
                st = self._to_pair(ak["random_saturation"], around1=True)
                ops.append(A.ColorJitter(brightness=0, contrast=0, saturation=st, hue=0, p=1.0))
            elif op == "random_hue" and ("random_hue" in ak):
                hu = self._to_pair(ak["random_hue"], symm=True)
                ops.append(A.ColorJitter(brightness=0, contrast=0, saturation=0, hue=hu, p=1.0))
            else:
                continue
        return A.ReplayCompose(ops, p=1.0)

    def _prepare_obs_chw(self, obs_img: np.ndarray) -> Tuple[torch.Tensor, dict | None]:
        """Prepare observation image(s):
        - obs_img can be HWC uint8 or THWC uint8 when obs_horizon>1
        - returns (CHW float [0,1], params_used)
        - if augment enabled, applies the same augmentation params for all frames
        - channels are stacked across time: result C=3*T
        """
        if obs_img.ndim == 3:
            H, W = obs_img.shape[:2]
            img = obs_img
            if self.train and self.augment:
                rc = self._build_albu(H, W)
                out = rc(image=img)
                img = out["image"]
                replay = out["replay"]
            else:
                replay = None
            t = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            return t, replay
        elif obs_img.ndim == 4:
            # multiple frames (T, H, W, C)
            Tn, H, W, C = obs_img.shape
            frames = list(obs_img)
            if self.train and self.augment:
                rc = self._build_albu(H, W)
                first = rc(image=frames[0])
                replay = first["replay"]
                imgs = [first["image"]]
                for i in range(1, Tn):
                    imgs.append(A.ReplayCompose.replay(replay, image=frames[i])["image"])
            else:
                imgs = frames
                replay = None
            ts = [torch.as_tensor(im.transpose(2, 0, 1), dtype=torch.float32) / 255.0 for im in imgs]
            return torch.cat(ts, dim=0), replay
        else:
            raise ValueError(f"Unexpected observation image shape: {obs_img.shape}")

    def _prepare_saliency_chw(self, sal: np.ndarray, replay: dict | None) -> torch.Tensor:
        """Prepare saliency map as CHW float in [0,1]; apply same spatial augment via replay if provided."""
        # sal is HWC with C=1 float32 in [0,1]
        if sal.ndim != 3 or sal.shape[2] != 1:
            raise ValueError(f"Unexpected saliency shape: {sal.shape}")
        # 按用户要求：saliency 不做任何图像增强或几何变换
        t = torch.as_tensor(sal.transpose(2, 0, 1), dtype=torch.float32)
        # normalize per-map to [0,1]
        mn = t.amin(dim=(-2, -1), keepdim=True)
        mx = t.amax(dim=(-2, -1), keepdim=True)
        t = (t - mn) / (mx - mn + 1e-8)
        return t

    def _prepare_goal_chw(self, goal_img: np.ndarray, replay: dict | None) -> torch.Tensor:
        # goal_img is HWC uint8
        H, W = goal_img.shape[:2]
        img = goal_img
        if self.train and self.augment:
            if self.augment_next_obs_goal_differently or replay is None:
                rc = self._build_albu(H, W)
                img = rc(image=img)["image"]
            else:
                img = A.ReplayCompose.replay(replay, image=img)["image"]
        return torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0

    def iterator(self) -> Iterator[dict]:
        bs = self.batch_size
        while True:
            batch = [self._get_transition(self.trajs[i], t) for (i, t) in (self._sample_index() for _ in range(bs))]

            # convert/augment per-sample, handling obs_horizon stacking
            obs_tensors: List[torch.Tensor] = []
            goal_tensors: List[torch.Tensor] = []
            sal_tensors: List[torch.Tensor] = []
            for b in batch:
                oimg = b["observations"]["image"]  # HWC or THWC uint8
                gimg = b["goals"]["image"]  # HWC uint8
                ot, params = self._prepare_obs_chw(oimg)
                gt = self._prepare_goal_chw(gimg, None if self.augment_next_obs_goal_differently else params)
                # if obs has more channels (multi-frame), tile goal to match
                if gt.shape[0] != ot.shape[0]:
                    if gt.shape[0] == 3 and ot.shape[0] % 3 == 0:
                        k = ot.shape[0] // 3
                        gt = gt.repeat(k, 1, 1)
                    else:
                        # fallback: interpolate goal to match by simple zero-pad/trim
                        if gt.shape[0] < ot.shape[0]:
                            pad = ot.shape[0] - gt.shape[0]
                            gt = torch.cat([gt, gt[:pad]], dim=0)
                        elif gt.shape[0] > ot.shape[0]:
                            gt = gt[:ot.shape[0]]
                obs_tensors.append(ot)
                goal_tensors.append(gt)
                # optional saliency per-sample
                if "saliency" in b["observations"] and b["observations"]["saliency"] is not None:
                    st = self._prepare_saliency_chw(b["observations"]["saliency"], None if self.augment_next_obs_goal_differently else params)
                    # if obs has more channels (multi-frame), we keep saliency single-channel
                    sal_tensors.append(st)

            acts = batch[0]["actions"]
            if isinstance(acts, np.ndarray) and acts.ndim == 1:
                # mix of seq/single possible; standardize below by stacking
                pass
            # actions: stack with numpy to preserve 2D or 3D
            acts = np.stack([b["actions"] for b in batch], axis=0).astype(np.float32)

            # channels_last for better cuDNN kernels
            obs_img_t = torch.stack(obs_tensors, dim=0).contiguous(memory_format=torch.channels_last)
            obs_dict = {"image": obs_img_t}
            # proprio if present
            if batch[0]["observations"]["proprio"] is not None:
                prop = np.stack([b["observations"]["proprio"] for b in batch], axis=0)
                obs_dict["proprio"] = _flatten_time_feat(prop)
            # attach saliency if available for all samples
            if sal_tensors and len(sal_tensors) == len(batch):
                obs_dict["saliency"] = torch.stack(sal_tensors, dim=0).contiguous(memory_format=torch.channels_last)

            gdict = {"image": torch.stack(goal_tensors, dim=0).contiguous(memory_format=torch.channels_last)}

            out = {
                "observations": obs_dict,
                "actions": torch.from_numpy(acts),
                "goals": gdict,
            }
            # Pin memory for faster H2D copies when used with non_blocking=True
            try:
                out["observations"]["image"] = out["observations"]["image"].pin_memory()
                if "proprio" in out["observations"] and isinstance(out["observations"]["proprio"], torch.Tensor):
                    out["observations"]["proprio"] = out["observations"]["proprio"].pin_memory()
                out["goals"]["image"] = out["goals"]["image"].pin_memory()
                out["actions"] = out["actions"].pin_memory()
            except Exception:
                pass
            yield out


def build_np_bridge_dataset(
    task_globs: List[Union[str, List[str]]],
    data_root: str,
    split: str,
    seed: int,
    batch_size: int,
    bridgedata_cfg,
    dataset_kwargs: Dict,
    ddp_shard: tuple[int, int] | None = None,
):
    # Resolve directories that contain out.npy
    assert isinstance(bridgedata_cfg.include[0], list)
    task_dirs: List[List[str]] = []
    for sub in bridgedata_cfg.include:
        sub_dirs = []
        for name in sub:
            sub_dirs.append(os.path.join(data_root, name, split, "out.npy"))
        task_dirs.append(sub_dirs)

    # Flatten lists (we ignore sample_weights for simplicity in numpy path)
    flat = [p for sub in task_dirs for p in sub]
    ds = BridgeNumpyDataset(flat, seed=seed, batch_size=batch_size, train=(split == "train"), **dataset_kwargs, action_proprio_metadata=bridgedata_cfg.action_proprio_metadata)
    return ds


def iter_torch_batches(ds: BridgeNumpyDataset) -> Iterator[Dict]:
    it = ds.iterator()
    for b in it:
        yield b


class _BatchIterable(IterableDataset):
    """IterableDataset wrapper that yields pre-built batches from BridgeNumpyDataset.iterator().

    This allows using PyTorch DataLoader for parallelism, pin_memory and prefetch.
    """
    def __init__(self, ds: BridgeNumpyDataset):
        super().__init__()
        self.ds = ds

    def __iter__(self):
        wi = get_worker_info()
        if wi is not None:
            # Derive distinct seed per worker to decorrelate sampling
            worker_seed = int(self.ds._base_seed) + 997 * int(wi.id)
            self.ds.reseed(worker_seed)
        it = self.ds.iterator()
        for batch in it:
            yield batch


def iter_torch_batches_np(
    ds: BridgeNumpyDataset,
    *,
    loader_kwargs: Dict | None = None,
) -> Iterator[Dict]:
    """DataLoader-based iterator with multi-worker prefetch and pinned memory.

    - Expects ds.iterator() to yield full batches; DataLoader passes them through.
    - Set batch_size=None and collate_fn to unwrap single-element lists from workers.
    """
    defaults: Dict = {
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "batch_size": None,
    }
    if loader_kwargs:
        defaults.update(dict(loader_kwargs))

    dataset = _BatchIterable(ds)
    # Identity collate for pre-built batches
    def _collate(x):
        return x[0] if isinstance(x, list) and len(x) == 1 else x

    dl = TorchDataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=_collate,
        num_workers=int(defaults.get("num_workers", 0)),
        pin_memory=bool(defaults.get("pin_memory", True)),
        persistent_workers=bool(defaults.get("persistent_workers", False)) if int(defaults.get("num_workers", 0)) > 0 else False,
        prefetch_factor=int(defaults.get("prefetch_factor", 2)) if int(defaults.get("num_workers", 0)) > 0 else None,
    )
    for b in dl:
        yield b
