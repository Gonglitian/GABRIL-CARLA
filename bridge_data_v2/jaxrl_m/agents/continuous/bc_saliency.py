from functools import partial
from typing import Any, Dict, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.core import FrozenDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, default_init, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper
from jaxrl_m.common.typing import Batch, PRNGKey
from jaxrl_m.networks.mlp import MLP


class PolicyWithSpatial(nn.Module):
    """
    A variant of the standard Gaussian policy that, in addition to the
    action distribution, can also return spatial encoder features to build a
    saliency map for gaze regularization.

    Notes:
    - If the underlying encoder outputs spatial features [B, H, W, C], this
      module will pool them via mean over (H, W) before the MLP head and also
      return the spatial features for saliency computation.
    - If the encoder outputs flat features [B, C], spatial features are None.
    """

    encoder: nn.Module
    network: nn.Module
    action_dim: int
    init_final: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None
    state_dependent_std: bool = True

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        temperature: float = 1.0,
        train: bool = False,
        return_features: bool = False,
    ):
        import distrax

        enc = self.encoder(observations)
        spatial_features = None
        if enc.ndim == 4:  # [B, H, W, C]
            spatial_features = enc
            enc = jnp.mean(enc, axis=(-3, -2))  # pool to [B, C]

        outputs = self.network(enc, train=train)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        if self.fixed_std is None:
            if self.state_dependent_std:
                log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(
                    outputs
                )
            else:
                log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )
        else:
            log_stds = jnp.log(jnp.array(self.fixed_std))

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max) / temperature

        if self.tanh_squash_distribution:
            distribution = distrax.Transformed(
                distribution=distrax.MultivariateNormalDiag(
                    loc=means, scale_diag=jnp.exp(log_stds)
                ),
                bijector=distrax.Block(distrax.Tanh(), 1),
            )
        else:
            distribution = distrax.MultivariateNormalDiag(
                loc=means, scale_diag=jnp.exp(log_stds)
            )

        if return_features:
            return distribution, spatial_features
        return distribution


def _normalize_minmax(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    x_min = jnp.min(x, axis=(1, 2, 3), keepdims=True)
    x_max = jnp.max(x, axis=(1, 2, 3), keepdims=True)
    return (x - x_min) / (x_max - x_min + eps)


def _to_nhwc1(x: jnp.ndarray) -> jnp.ndarray:
    """Ensure [B, H, W, 1] shape, accepting [B,H,W], [B,1,H,W], or [B,H,W,1]."""
    if x.ndim == 3:  # [B, H, W]
        x = x[..., None]
    elif x.ndim == 4 and x.shape[1] == 1:  # [B, 1, H, W] -> [B, H, W, 1]
        x = jnp.moveaxis(x, 1, -1)
    # else assume [B, H, W, 1]
    return x


def _get_gaze_mask_from_features(
    z_spatial: Optional[jnp.ndarray], beta: float, target_hw: Tuple[int, int]
) -> Optional[jnp.ndarray]:
    """
    JAX port of vlm_gaze.data_utils.gaze_utils.get_gaze_mask:
    - z_spatial: [B, H, W, C] or None
    - returns [B, H_tgt, W_tgt, 1] or None if z_spatial is None
    """
    if z_spatial is None:
        return None
    # abs-sum over channels -> [B,H,W]
    z_abs = jnp.sum(jnp.abs(z_spatial), axis=-1)
    b, h, w = z_abs.shape
    z_flat = z_abs.reshape(b, h * w)
    z_softmax = jax.nn.softmax(z_flat / beta, axis=-1).reshape(b, h, w)
    z_softmax = z_softmax[..., None]  # [B,H,W,1]
    import jax.image as jimage
    # Use linear resize for compatibility
    z_resized = jimage.resize(
        z_softmax, shape=(b, target_hw[0], target_hw[1], 1), method="linear"
    )

    return _normalize_minmax(z_resized)


class BCSaliencyAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()
    # Gaze regularization config
    reg_lambda: float = nonpytree_field()
    prob_dist_type: str = nonpytree_field()
    gaze_ratio: float = nonpytree_field()
    beta: float = nonpytree_field()

    @staticmethod
    def _select_gaze_from_batch(batch: Batch) -> Optional[jnp.ndarray]:
        """
        Try to find gaze heatmaps in the batch. Supported locations/shapes:
        - batch["gaze_heatmaps"]: [B,H,W] | [B,1,H,W] | [B,H,W,1]
        - batch["gaze"]: same shapes
        - batch["observations"]["gaze"]: same shapes
        Returns NHWC with 1 channel or None.
        """
        cand = None
        if "gaze_heatmaps" in batch:
            cand = batch["gaze_heatmaps"]
        elif "gaze" in batch:
            cand = batch["gaze"]
        elif "gaze" in batch.get("observations", {}):
            cand = batch["observations"]["gaze"]
        if cand is None:
            return None
        return _to_nhwc1(cand)

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, rng):
            rng, key, key_mask = jax.random.split(rng, 3)

            # Forward pass: get distribution and spatial features
            actor_out = self.state.apply_fn(
                {"params": params},
                batch["observations"],
                temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="actor",
                return_features=True,
            )
            if isinstance(actor_out, tuple):
                dist, spatial_feat = actor_out
            else:
                dist, spatial_feat = actor_out, None

            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])  # [B]
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            # Gaze regularization
            g_true = BCSaliencyAgent._select_gaze_from_batch(batch)
            reg_loss = 0.0
            num = 0.0
            if g_true is not None:
                images = batch["observations"]["image"]
                bsz = images.shape[0]
                target_hw = (images.shape[1], images.shape[2])  # H, W from NHWC

                # Compute model-pred saliency mask from spatial features if available
                g_pred = _get_gaze_mask_from_features(spatial_feat, self.beta, target_hw)

                # If spatial features not available, skip reg
                if g_pred is not None:
                    # Sample which samples use gaze via ratio
                    mask = (jax.random.uniform(key_mask, (bsz,)) < self.gaze_ratio).astype(
                        jnp.float32
                    )
                    mask = mask[:, None, None, None]  # [B,1,1,1]

                    g_true_nhwc = g_true

                    # Resize g_true to match image (if needed)
                    if (
                        g_true_nhwc.shape[1] != target_hw[0]
                        or g_true_nhwc.shape[2] != target_hw[1]
                    ):
                        import jax.image as jimage

                        g_true_nhwc = jimage.resize(
                            g_true_nhwc,
                            shape=(bsz, target_hw[0], target_hw[1], 1),
                            method="nearest",
                        )

                    # Align distributions if needed
                    eps = 1e-6
                    if self.prob_dist_type in ["KL", "JS", "TV"]:
                        # Normalize to probability maps
                        g1_sum = jnp.sum(g_true_nhwc, axis=(1, 2, 3), keepdims=True) + 1e-8
                        g2_sum = jnp.sum(g_pred, axis=(1, 2, 3), keepdims=True) + 1e-8
                        g1 = g_true_nhwc / g1_sum
                        g2 = g_pred / g2_sum
                    else:
                        g1 = g_true_nhwc
                        g2 = g_pred

                    # Divergences per sample
                    def _kl(a, b):
                        return jnp.sum(a * jnp.log((a + eps) / (b + eps)), axis=(1, 2, 3))

                    if self.prob_dist_type == "KL":
                        per_sample = _kl(g1, g2)
                    elif self.prob_dist_type == "TV":
                        per_sample = jnp.sum(jnp.abs(g1 - g2), axis=(1, 2, 3))
                    elif self.prob_dist_type == "JS":
                        m = 0.5 * (g1 + g2)
                        per_sample = 0.5 * (_kl(g1, m) + _kl(g2, m))
                    elif self.prob_dist_type == "MSE":
                        per_sample = jnp.mean((g1 - g2) ** 2, axis=(1, 2, 3))
                    else:
                        raise ValueError(
                            f"Invalid prob_dist_type: {self.prob_dist_type}"
                        )

                    # Apply sample mask
                    per_sample = per_sample * mask.reshape(-1)
                    denom = jnp.sum(mask) + 1e-8
                    reg_loss = jnp.sum(per_sample) / denom
                    num = denom

            total_loss = actor_loss + self.reg_lambda * reg_loss

            return (
                total_loss,
                {
                    "actor_loss": actor_loss,
                    "reg_loss": reg_loss if isinstance(reg_loss, jnp.ndarray) else jnp.array(reg_loss),
                    "log_probs": log_probs.mean(),
                    "pi_actions": pi_actions.mean(),
                    "mean_std": actor_std.mean(),
                    "max_std": actor_std.max(),
                    "gaze_used": num,
                },
            )

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax: bool = False,
    ) -> jnp.ndarray:
        out = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            temperature=temperature,
            name="actor",
        )
        if isinstance(out, tuple):
            dist, _ = out
        else:
            dist = out

        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        out = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            temperature=1.0,
            name="actor",
        )
        if isinstance(out, tuple):
            dist, _ = out
        else:
            dist = out
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])  # [B]
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
        return {"mse": mse, "log_probs": log_probs, "pi_actions": pi_actions}

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        goals: FrozenDict,  # unused but kept for API compatibility
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        use_proprio: bool = False,
        network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "state_dependent_std": False,
        },
        # Gaze regularization
        lambda_weight: float = 1.0,
        prob_dist_type: str = "KL",
        gaze_ratio: float = 1.0,
        beta: float = 1.0,
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
    ):
        # Wrap encoder with common observation logic
        encoder_def = EncodingWrapper(
            encoder=encoder_def, use_proprio=use_proprio, stop_gradient=False
        )

        network_kwargs = dict(network_kwargs)
        network_kwargs["activate_final"] = True
        networks = {
            "actor": PolicyWithSpatial(
                encoder_def,
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs,
            )
        }

        model_def = ModuleDict(networks)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, actor=[observations])["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        return cls(
            state,
            lr_schedule,
            lambda_weight,
            prob_dist_type,
            gaze_ratio,
            beta,
        )
