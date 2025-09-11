from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from bridge_torch.common.gaze import get_gaze_mask

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 256, 256), dropout_rate=0.0, activate_final=True):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if dropout_rate and dropout_rate > 0:
                layers += [nn.Dropout(dropout_rate)]
            last = h
        if activate_final:
            self.net = nn.Sequential(*layers)
            self.out_dim = last
        else:
            self.net = nn.Sequential(*layers[:-1])
            self.out_dim = layers[-2].out_features if layers else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        mlp: MLP,
        action_dim: int,
        tanh_squash_distribution: bool = False,
        state_dependent_std: bool = False,
        fixed_std: list[float] | None = None,
        use_proprio: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.mu = nn.Linear(self.mlp.out_dim, action_dim)
        self.tanh_squash = tanh_squash_distribution
        self.state_std = state_dependent_std
        self.use_proprio = use_proprio
        if self.state_std:
            self.log_std = nn.Linear(self.mlp.out_dim, action_dim)
        else:
            fs = torch.tensor(
                fixed_std if fixed_std is not None else [1.0] * action_dim,
                dtype=torch.float32,
            )
            self.register_buffer("fixed_std", fs)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = obs["image"]  # (N, C, H, W)
        z = self.encoder(x)
        if self.use_proprio and ("proprio" in obs):
            prop = obs["proprio"]
            if prop.dim() > 2:
                prop = prop.flatten(1)
            z = torch.cat([z, prop], dim=-1)
        h = self.mlp(z)
        mu = self.mu(h)
        if self.state_std:
            std = torch.exp(self.log_std(h)).clamp(min=1e-4, max=10.0)
        else:
            std = self.fixed_std.expand_as(mu)
        return {"mu": mu, "std": std}

    def dist(self, out: Dict[str, torch.Tensor]):
        mu, std = out["mu"], out["std"]
        return torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)


@dataclass
class BCAgentConfig:
    network_kwargs: Dict[str, Any]
    policy_kwargs: Dict[str, Any]
    use_proprio: bool = False
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    decay_steps: int = 1000000


class BCAgent:
    def __init__(self, model: GaussianPolicy, cfg: BCAgentConfig, action_dim: int, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        # Scheduler selectable via config.scheduler.type
        self.scheduler = None
        sched_type = getattr(cfg, "scheduler", {}).get("type", "warmup_cosine")
        if sched_type == "constant":
            self.scheduler = None
        else:
            scheds: list[torch.optim.lr_scheduler._LRScheduler] = []
            if cfg.warmup_steps and cfg.warmup_steps > 0:
                scheds.append(torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=cfg.warmup_steps))
            if cfg.decay_steps and cfg.decay_steps > 0:
                scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.decay_steps, eta_min=0.0))
            if scheds:
                if len(scheds) == 1:
                    self.scheduler = scheds[0]
                else:
                    self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, scheds, milestones=[cfg.warmup_steps])
        self.step = 0

        # Saliency-variant config (optional, from YAML)
        # tolerate ml_collections.ConfigDict or plain dict
        sal_cfg = {}
        try:
            pk = getattr(cfg, "policy_kwargs", {}) or {}
            if hasattr(pk, "to_dict"):
                pk = pk.to_dict()
            elif not isinstance(pk, dict):
                try:
                    pk = dict(pk)
                except Exception:
                    pk = {}
            sal_cfg = dict(pk.get("saliency", {}) or {})
            if not sal_cfg:
                direct = getattr(cfg, "saliency", {}) or {}
                if hasattr(direct, "to_dict"):
                    sal_cfg = direct.to_dict()
                elif isinstance(direct, dict):
                    sal_cfg = dict(direct)
        except Exception:
            sal_cfg = {}
        self._saliency_enabled = (str(sal_cfg.get("enabled", False)).strip().lower() in {"1", "true", "t", "yes", "y", "on"})
        self._saliency_weight = float(sal_cfg.get("weight", 1.0))
        self._saliency_beta = float(sal_cfg.get("beta", 1.0))
        self._last_spatial = None
        if self._saliency_enabled:
            self._register_spatial_hook()

    def _register_spatial_hook(self) -> None:
        """Register forward hooks on encoder to capture the last 4D activation (B,C,H,W)."""
        def _hook(_module, _inp, out):
            try:
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    self._last_spatial = out
            except Exception:
                pass
            return None
        for m in self.model.encoder.modules():
            m.register_forward_hook(_hook)

    def update(self, batch: Dict[str, torch.Tensor]):
        self.model.train()
        obs = {k: v.to(self.device, non_blocking=True) for k, v in batch["observations"].items() if isinstance(v, torch.Tensor)}
        actions = batch["actions"].to(self.device, non_blocking=True)
        out = self.model(obs)
        dist = self.model.dist(out)
        log_prob = dist.log_prob(actions)
        actor_loss = -log_prob.mean()
        reg_loss = torch.tensor(0.0, device=self.device)
        # Saliency regularization (Reg variant)
        if self._saliency_enabled and ("saliency" in obs):
            try:
                z_map = self._last_spatial  # (B,C,Hf,Wf)
                self._last_spatial = None  # reset for next step
                if z_map is not None and z_map.dim() == 4:
                    target = obs["saliency"]  # (B,1,H,W)
                    H, W = int(target.shape[-2]), int(target.shape[-1])
                    pred = get_gaze_mask(z_map, self._saliency_beta, (H, W))  # (B,1,H,W)
                    reg_loss = F.mse_loss(pred, target)
            except Exception as e:
                print(e)
        loss = actor_loss + self._saliency_weight * reg_loss
        with torch.no_grad():
            mse = F.mse_loss(out["mu"], actions, reduction="none").sum(-1).mean()
            mean_std = out["std"].mean(dim=1).mean()
            max_std = out["std"].max()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1

        return {
            "actor_loss": float(actor_loss.item()),
            "saliency_reg": float(reg_loss.item()) if reg_loss is not None else 0.0,
            "total_loss": float(loss.item()),
            "mse": mse.item(),
            "log_probs": log_prob.mean().item(),
            "mean_std": mean_std.item(),
            "max_std": max_std.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def get_debug_metrics(self, batch: Dict[str, torch.Tensor]):
        self.model.eval()
        obs = {k: v.to(self.device, non_blocking=True) for k, v in batch["observations"].items() if isinstance(v, torch.Tensor)}
        actions = batch["actions"].to(self.device, non_blocking=True)
        out = self.model(obs)
        dist = self.model.dist(out)
        log_prob = dist.log_prob(actions)
        mse = F.mse_loss(out["mu"], actions, reduction="none").sum(-1).mean()
        return {"mse": mse.item(), "log_probs": log_prob.mean().item()}

    @torch.no_grad()
    def sample_actions(self, obs: Dict[str, torch.Tensor], argmax: bool = False):
        self.model.eval()
        out = self.model({k: v.to(self.device, non_blocking=True) for k, v in obs.items()})
        dist = self.model.dist(out)
        if argmax:
            return out["mu"].cpu()
        return dist.sample().cpu()
