from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import copy as _copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from vlm_gaze.data_utils.gaze_utils import get_gaze_mask
except Exception:
    get_gaze_mask = None


class MLP(nn.Module):
    def __init__(self, hidden_dims: Tuple[int, ...], activate_final: bool = False, dropout_rate: Optional[float] = None, use_layer_norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        last = hidden_dims[0]
        for i, h in enumerate(hidden_dims[1:]):
            layers.append(nn.Linear(last, h))
            if i + 2 < len(hidden_dims) or activate_final:
                if dropout_rate:
                    layers.append(nn.Dropout(dropout_rate))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(h))
                layers.append(nn.SiLU())
            last = h
        # 若只给了一个维度，表示恒等映射
        if not layers:
            layers.append(nn.Identity())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPResNetBlock(nn.Module):
    def __init__(self, features: int, dropout_rate: Optional[float] = None, use_layer_norm: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
        self.ln = nn.LayerNorm(features) if use_layer_norm else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.dropout(self.ln(x))
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        if residual.shape != y.shape:
            residual = nn.Linear(residual.shape[-1], y.shape[-1], bias=False).to(y.device)(residual)
        return residual + y


class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, out_dim: int, dropout_rate: Optional[float] = None, use_layer_norm: bool = False, hidden_dim: int = 256):
        super().__init__()
        self.fc_in = nn.Linear(1, hidden_dim)  # will rebuild at first forward
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, dropout_rate=dropout_rate, use_layer_norm=use_layer_norm) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self._built = False

    @torch._dynamo.disable
    def _build(self, in_dim: int):
        # rebuild input projection and move to the same device as existing modules
        device = next(self.fc_out.parameters()).device if any(True for _ in self.fc_out.parameters()) else None
        self.fc_in = nn.Linear(in_dim, self.hidden_dim)
        if device is not None:
            self.fc_in = self.fc_in.to(device)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[-1])
        # guard in case device changed
        if self.fc_in.weight.device != x.device:
            self.fc_in = self.fc_in.to(x.device)
        x = self.fc_in(x)
        for b in self.blocks:
            x = b(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x


class FourierFeatures(nn.Module):
    def __init__(self, output_size: int, learnable: bool = True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        if learnable:
            self.kernel = nn.Parameter(torch.randn(output_size // 2, 1) * 0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1)
        if self.learnable:
            f = 2 * math.pi * x @ self.kernel.T
        else:
            half_dim = self.output_size // 2
            f = math.log(10000) / (half_dim - 1)
            freqs = torch.exp(torch.arange(half_dim, device=x.device) * -f)
            f = x * freqs
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class ScoreActorTorch(nn.Module):
    def __init__(self, obs_encoder: nn.Module, goal_encoder: Optional[nn.Module], early_goal_concat: bool, time_preprocess: FourierFeatures, cond_encoder: MLP, reverse_network: MLPResNet, action_seq_shape: Tuple[int, int], use_proprio: bool = False):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.goal_encoder = goal_encoder
        self.early = early_goal_concat
        self.time_preprocess = time_preprocess
        self.cond_encoder = cond_encoder
        self.reverse_network = reverse_network
        self.action_seq_shape = action_seq_shape
        self.use_proprio = use_proprio

        self._enc_dim = None

    def _encode(self, observations: Dict[str, torch.Tensor], goals: Dict[str, torch.Tensor]) -> torch.Tensor:
        zo = self.obs_encoder(observations["image"])  # (B, D)
        if self.goal_encoder is None:
            # 共享编码器场景：直接用同一个 obs_encoder 对目标图进行编码，避免模块双注册
            zg = self.obs_encoder(goals["image"]) if (goals is not None and "image" in goals) else torch.zeros_like(zo)
            z = torch.cat([zo, zg], dim=-1) if self.early else (zo + zg)
        else:
            zg = self.goal_encoder(goals["image"])  # (B, D')
            z = torch.cat([zo, zg], dim=-1) if self.early else (zo + zg)
        if self.use_proprio and ("proprio" in observations):
            prop = observations["proprio"]
            if prop.dim() > 2:
                prop = prop.flatten(1)
            z = torch.cat([z, prop], dim=-1)
        return z

    def forward(self, observations: Dict[str, torch.Tensor], goals: Dict[str, torch.Tensor], actions: torch.Tensor, time: torch.Tensor, train: bool = False) -> torch.Tensor:
        # actions: (B, T, A)
        b, t, a = actions.shape
        flat_actions = actions.reshape(b, t * a)
        t_ff = self.time_preprocess(time)  # (B, 2*time_dim)
        cond_enc = self.cond_encoder(t_ff)
        obs_enc = self._encode(observations, goals)
        rev_in = torch.cat([cond_enc, obs_enc, flat_actions], dim=-1)
        eps_pred = self.reverse_network(rev_in)
        return eps_pred.view(b, t, a)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


def vp_beta_schedule(timesteps: int) -> torch.Tensor:
    t = torch.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / (T ** 2))
    betas = 1 - alpha
    return betas


def build_ddpm_policy(
    *,
    obs_encoder: nn.Module,
    goal_encoder: Optional[nn.Module],
    early_goal_concat: bool,
    time_dim: int,
    num_blocks: int,
    dropout_rate: float,
    hidden_dim: int,
    use_layer_norm: bool,
    action_seq_shape: Tuple[int, int],
    device: torch.device,
    use_proprio: bool = False,
    proprio_feature_dim: int = 0,
) -> ScoreActorTorch:
    # 注意：FourierFeatures 最终输出维度等于其构造参数 output_size；
    # 为了让 cond_enc 接收 2*time_dim 的输入，这里设为 2*time_dim
    time_pre = FourierFeatures(2 * time_dim, learnable=True).to(device)
    cond_enc = MLP((2 * time_dim, time_dim), activate_final=False).to(device)
    rev_net = MLPResNet(num_blocks=num_blocks, out_dim=action_seq_shape[0] * action_seq_shape[1], dropout_rate=dropout_rate, use_layer_norm=use_layer_norm, hidden_dim=hidden_dim).to(device)
    policy = ScoreActorTorch(
        obs_encoder,
        goal_encoder,
        early_goal_concat,
        time_pre,
        cond_enc,
        rev_net,
        action_seq_shape,
        use_proprio=use_proprio,
    )
    # 预构建 reverse_network 的输入维度，避免首次前向在 compile 图中动态重建
    try:
        obs_dim = int(getattr(obs_encoder, "_feat_dim"))
        goal_dim = int(getattr(goal_encoder, "_feat_dim")) if goal_encoder is not None else obs_dim
        enc_dim = (obs_dim + goal_dim) if early_goal_concat else obs_dim
        # cond_enc 输出维度为 time_dim
        cond_dim = time_dim
        action_flat = int(action_seq_shape[0] * action_seq_shape[1])
        total_in = cond_dim + enc_dim + int(proprio_feature_dim) + action_flat
        rev_net._build(total_in)
    except Exception:
        # 安全降级：若无法获知特征维度，则交由首次前向构建
        pass
    return policy.to(device)


@dataclass
class GCDDPMConfig:
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    actor_decay_steps: Optional[int] = None
    beta_schedule: str = "cosine"  # cosine | linear | vp
    diffusion_steps: int = 20
    action_samples: int = 1
    repeat_last_step: int = 0
    target_update_rate: float = 0.002
    action_min: float = -2.0
    action_max: float = 2.0


class GCDDPMBCAgent:
    def __init__(self, policy: ScoreActorTorch, cfg: GCDDPMConfig, device: torch.device):
        self.model = policy.to(device)
        self.cfg = cfg
        self.device = device
        self.step = 0
        self._action_T, self._action_A = policy.action_seq_shape
        # 提前初始化 target_model，避免对已编译模型进行 deepcopy
        self.target_model: ScoreActorTorch = _copy.deepcopy(self.model).to(self.device)
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        # Scheduler selectable via config (actor path)
        self.scheduler = None
        sched_type = "warmup_cosine"  # follow torch side config; ddpm uses actor_decay_steps
        if sched_type == "constant":
            self.scheduler = None
        else:
            scheds: list[torch.optim.lr_scheduler._LRScheduler] = []
            if cfg.warmup_steps and cfg.warmup_steps > 0:
                scheds.append(torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=cfg.warmup_steps))
            if cfg.actor_decay_steps:
                scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.actor_decay_steps, eta_min=0.0))
            if scheds:
                if len(scheds) == 1:
                    self.scheduler = scheds[0]
                else:
                    self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, scheds, milestones=[cfg.warmup_steps])

        # diffusion parameters
        T = cfg.diffusion_steps
        if cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(T)
        elif cfg.beta_schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, T)
        elif cfg.beta_schedule == "vp":
            betas = vp_beta_schedule(T)
        else:
            raise ValueError(f"Unknown beta_schedule: {cfg.beta_schedule}")
        self.register_buffers(betas)

    def register_buffers(self, betas: torch.Tensor):
        self.betas = betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        alpha_hats = []
        prod = 1.0
        for i in range(len(self.betas)):
            prod = prod * self.alphas[i]
            alpha_hats.append(prod)
        self.alpha_hats = torch.tensor(alpha_hats, device=self.device, dtype=self.betas.dtype)

        # saliency variant
        sal_cfg = {}
        try:
            # ddpm config puts actor-related kwargs in agent config; allow nested dict too
            sal_cfg = dict(getattr(cfg, "saliency", {}) or {})
        except Exception:
            sal_cfg = getattr(cfg, "saliency", {}) or {}
        self._saliency_enabled = bool(sal_cfg.get("enabled", False))
        self._saliency_weight = float(sal_cfg.get("weight", 1.0))
        self._saliency_beta = float(sal_cfg.get("beta", 1.0))
        self._last_spatial = None
        if self._saliency_enabled:
            self._register_spatial_hook()

    def _register_spatial_hook(self) -> None:
        def _hook(_m, _i, out):
            try:
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    self._last_spatial = out
            except Exception:
                pass
            return None
        for m in self.model.obs_encoder.modules():
            m.register_forward_hook(_hook)

    def _polyak_update(self, tau: float):
        with torch.no_grad():
            for p, tp in zip(self.model.parameters(), self.target_model.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

    def update(self, batch: Dict[str, torch.Tensor]):
        self.model.train()
        obs = {k: v.to(self.device, non_blocking=True) for k, v in batch["observations"].items() if isinstance(v, torch.Tensor)}
        goal = {k: v.to(self.device, non_blocking=True) for k, v in batch.get("goals", {}).items() if isinstance(v, torch.Tensor)}
        actions = batch["actions"].to(self.device, non_blocking=True)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        B, T, A = actions.shape

        # sample timestep per example
        t = torch.randint(0, self.cfg.diffusion_steps, (B,), device=self.device)
        # noise
        noise = torch.randn_like(actions)
        # alpha_hat[t]
        a_hat = self.alpha_hats[t]  # (B,)
        a1 = torch.sqrt(a_hat).view(B, 1, 1)
        a2 = torch.sqrt(1 - a_hat).view(B, 1, 1)
        noisy_actions = a1 * actions + a2 * noise
        # time as float column
        t_float = t.float().view(B, 1)
        eps_pred = self.model(obs, goal, noisy_actions, t_float)
        ddpm_loss = F.mse_loss(eps_pred, noise, reduction="mean")
        reg_loss = torch.tensor(0.0, device=self.device)
        if self._saliency_enabled and (get_gaze_mask is not None) and ("saliency" in obs):
            try:
                z_map = self._last_spatial
                self._last_spatial = None
                if z_map is not None and z_map.dim() == 4:
                    target = obs["saliency"]
                    H, W = int(target.shape[-2]), int(target.shape[-1])
                    pred = get_gaze_mask(z_map, self._saliency_beta, (H, W))
                    reg_loss = F.mse_loss(pred, target)
            except Exception:
                pass
        loss = ddpm_loss + self._saliency_weight * reg_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # Update target network
        self._polyak_update(self.cfg.target_update_rate)
        self.step += 1

        info = {
            "ddpm_loss": float(ddpm_loss.item()),
            "saliency_reg": float(reg_loss.item()) if reg_loss is not None else 0.0,
            "total_loss": float(loss.item()),
            "actor_lr": self.optimizer.param_groups[0]["lr"],
        }
        return info

    @torch.no_grad()
    def sample_actions(
        self,
        observations: Dict[str, torch.Tensor],
        goals: Dict[str, torch.Tensor],
        *,
        temperature: float = 1.0,
        clip_sampler: bool = True,
    ) -> torch.Tensor:
        self.model.eval()
        obs = {k: v.to(self.device, non_blocking=True) for k, v in observations.items() if isinstance(v, torch.Tensor)}
        goal = {k: v.to(self.device, non_blocking=True) for k, v in goals.items() if isinstance(v, torch.Tensor)}
        # infer action shape
        B = list(obs.values())[0].shape[0]
        T = getattr(self, "_action_T", None)
        A = getattr(self, "_action_A", None)
        if T is None or A is None:
            # fallback: read from reverse net out_dim
            out_dim = self.model.reverse_network.fc_out.out_features
            # try to infer a square-ish (T, A) from config? We don't have; default to (1, out_dim)
            T, A = 1, out_dim
            self._action_T, self._action_A = T, A
        x = torch.randn(B, T, A, device=self.device)
        for time in reversed(range(self.cfg.diffusion_steps)):
            t_col = torch.full((B, 1), float(time), device=self.device)
            eps_pred = self.target_model(obs, goal, x, t_col)
            alpha_t = self.alphas[time]
            alpha_hat_t = self.alpha_hats[time]
            x = (1.0 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * eps_pred)
            if time > 0:
                z = torch.randn_like(x) * temperature
                x = x + torch.sqrt(self.betas[time]) * z
            if clip_sampler:
                x = torch.clamp(x, self.cfg.action_min, self.cfg.action_max)
        return x.cpu()

    @torch.no_grad()
    def get_debug_metrics(self, batch: Dict[str, torch.Tensor]):
        actions = self.sample_actions(batch["observations"], batch["goals"])
        gt = batch["actions"]
        if gt.dim() == 2:
            gt = gt.unsqueeze(1)
        if actions.shape != gt.shape:
            # try to match shapes by slicing
            min_t = min(actions.shape[1], gt.shape[1])
            min_a = min(actions.shape[2], gt.shape[2])
            actions = actions[:, :min_t, :min_a]
            gt = gt[:, :min_t, :min_a]
        mse = F.mse_loss(actions, gt, reduction="none").sum(dim=(-2, -1)).mean()
        return {"mse": float(mse.item())}
