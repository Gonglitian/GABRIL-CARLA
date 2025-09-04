#!/usr/bin/env python3
"""
Robomimic SequenceDataset Integration for GABRIL-CARLA
Provides dataset/dataloader functionality using robomimic's efficient SequenceDataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import robomimic (external dependency or vendored under vlm_gaze.robomimic)
from robomimic.utils.dataset import SequenceDataset

# ============================================================================
# Custom Gaze Preprocessor for Robomimic
# ============================================================================

class GazePreprocessor(nn.Module):
    """
    GPU-based gaze heatmap generation compatible with robomimic
    """
    
    def __init__(
        self,
        img_height: int = 180,
        img_width: int = 320,
        gaze_sigma: float = 30.0,
        gaze_coeff: float = 0.8,
        maxpoints: int = 5,
        device: str = 'cuda',
        # Temporal accumulation hyperparameters (for multimodal Gaussian over time)
        temporal_k: int = 0,
        temporal_alpha: float = 0.7,
        temporal_beta: float = 0.8,
        temporal_gamma: float = 1.0,
        temporal_use_future: bool = True,
        # Multiscale causal aggregation parameters (optional)
        temporal_mode: str = 'alpha_decay',  # 'alpha_decay' or 'multiscale'
        temporal_sigmas: list | None = None,  # e.g., [30, 24, 18, 12]
        temporal_coeffs: list | None = None,  # e.g., [1.0, 0.8, 0.6, 0.4]
        temporal_offset_start: int = 0,
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.maxpoints = maxpoints
        self.gaze_sigma = gaze_sigma
        self.gaze_coeff = gaze_coeff
        self.device = device
        # Temporal hyperparameters
        self.temporal_k = int(max(0, temporal_k))
        self.temporal_alpha = float(temporal_alpha)
        self.temporal_beta = float(temporal_beta)
        self.temporal_gamma = float(temporal_gamma)
        self.temporal_use_future = bool(temporal_use_future)
        # Multiscale temporal aggregation configs
        self.temporal_mode = str(temporal_mode)
        # Convert possibly OmegaConf ListConfig to plain python lists
        if temporal_sigmas is not None:
            self.temporal_sigmas = [float(x) for x in temporal_sigmas]
        else:
            self.temporal_sigmas = None
        if temporal_coeffs is not None:
            self.temporal_coeffs = [float(x) for x in temporal_coeffs]
        else:
            self.temporal_coeffs = None
        self.temporal_offset_start = int(max(0, temporal_offset_start))
        
        # Pre-compute Gaussian kernel for base forward()
        kernel_size = int(4 * gaze_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * gaze_sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        self.register_buffer('kernel_1d', kernel_1d.to(device))
        # Cache for variable-sigma kernels used in multiscale aggregation
        self._sigma_kernel_cache: dict[float, torch.Tensor] = {}
    
    def forward(self, gaze_coords: torch.Tensor) -> torch.Tensor:
        """
        Generate gaze heatmaps from coordinates
        
        Args:
            gaze_coords: [B, T, maxpoints*2] or [B, T, maxpoints, 2] tensor
        
        Returns:
            heatmaps: [B, T, 1, H, W] tensor
        """
        if gaze_coords.dim() == 3 and gaze_coords.shape[-1] == self.maxpoints * 2:
            # [B, T, maxpoints*2] -> [B, T, maxpoints, 2]
            gaze_coords = rearrange(gaze_coords, 'b t (p c) -> b t p c', p=self.maxpoints, c=2)
        elif gaze_coords.dim() == 2:
            # [T, maxpoints*2] -> [1, T, maxpoints, 2]
            gaze_coords = rearrange(gaze_coords, 't (p c) -> 1 t p c', p=self.maxpoints, c=2)

        B, T, P, _ = gaze_coords.shape
        H, W = self.img_height, self.img_width
        device = gaze_coords.device

        # Valid points mask: (x>=0 and y>=0)
        valid_mask = (gaze_coords[..., 0] >= 0) & (gaze_coords[..., 1] >= 0)  # [B, T, P]

        # Scale coords to pixel indices
        x_coords = (gaze_coords[..., 0].clamp(0, 1) * (W - 1)).long().clamp(0, W - 1)
        y_coords = (gaze_coords[..., 1].clamp(0, 1) * (H - 1)).long().clamp(0, H - 1)

        # Uniform weights per valid gaze point within each (B, T), independent of gaze_coeff
        weights = valid_mask.float()

        # Build delta maps for all (B,T) at once via scatter_add
        num_samples = B * T
        delta = torch.zeros(num_samples, H * W, device=device, dtype=torch.float32)
        # Compute linear indices per pixel
        lin_idx = rearrange(y_coords * W + x_coords, 'b t p -> (b t) p')  # [B*T, P]
        w_flat = rearrange(weights, 'b t p -> (b t) p')
        delta.scatter_add_(dim=1, index=lin_idx, src=w_flat)
        delta = rearrange(delta, '(b t) (h w) -> b t 1 h w', b=B, t=T, h=H, w=W)

        # Apply separable Gaussian blur in a vectorized manner
        padding = self.kernel_size // 2
        kernel = rearrange(self.kernel_1d, 'l -> 1 1 l 1')
        # Flatten batch and time for convolution
        delta_bt = rearrange(delta, 'b t c h w -> (b t) c h w')
        blurred = F.conv2d(delta_bt, kernel, padding=(0, padding))
        kernel_t = rearrange(kernel, 'a b h w -> a b w h')
        blurred = F.conv2d(blurred, kernel_t, padding=(padding, 0))
        # Normalize each heatmap to [0, 1] via min-max
        min_vals = blurred.amin(dim=(2, 3), keepdim=True)
        max_vals = blurred.amax(dim=(2, 3), keepdim=True)
        normalized = (blurred - min_vals) / (max_vals - min_vals + 1e-8)
        heatmaps = rearrange(normalized, '(b t) c h w -> b t c h w', b=B, t=T)

        return heatmaps

    # ---------------------------------------------------------------------
    # Stack-aware helpers and unified APIs for training scripts
    # ---------------------------------------------------------------------
    @staticmethod
    def _gather_last_s_frames(seq: torch.Tensor, center_idx: int, stack_len: int) -> torch.Tensor:
        """
        Generic utility: from [B, L, ...] gather a window [B, S, ...] that ends at center_idx.
        Pads by clamping at boundaries to preserve length S.
        """
        assert seq.dim() >= 2, f"Expected [B, L, ...], got {seq.shape}"
        B, L = seq.shape[0], seq.shape[1]
        start = center_idx - (stack_len - 1)
        idxs = [min(max(i, 0), L - 1) for i in range(start, center_idx + 1)]
        while len(idxs) < stack_len:
            idxs.insert(0, idxs[0])
        index_tensor = torch.tensor(idxs, device=seq.device, dtype=torch.long)
        return seq.index_select(dim=1, index=index_tensor)

    @staticmethod
    def extract_image_stack_around_center(images_seq: torch.Tensor, center_idx: int, frame_stack: int) -> torch.Tensor:
        """
        From [B, L, H, W, C], build [B, S, H, W, C] ending at center_idx.
        If input is already [B, H, W, C], return as-is.
        """
        if images_seq.dim() != 5:
            return images_seq
        return GazePreprocessor._gather_last_s_frames(images_seq, center_idx=center_idx, stack_len=frame_stack)

    @staticmethod
    def extract_gaze_stack_around_center(gaze_seq: torch.Tensor, center_idx: int, frame_stack: int) -> torch.Tensor:
        """
        From [B, L, P*2] or [B, L, P, 2], build [B, S, P*2] or [B, S, P, 2].
        If input has no time dimension, return as-is.
        """
        if gaze_seq.dim() < 3:
            return gaze_seq
        return GazePreprocessor._gather_last_s_frames(gaze_seq, center_idx=center_idx, stack_len=frame_stack)

    @staticmethod
    def _format_obs_image(images: torch.Tensor, frame_stack: int, grayscale: bool) -> torch.Tensor:
        """
        Format images for encoder input. Accepts [B, S, H, W, C] or [B, H, W, C].
        Returns channels-first tensor [B, C_img, H, W] where C_img = S * (1 or 3).
        """
        from einops import rearrange as _rearr
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        # [B, S, H, W, C]
        if images.dim() == 5 and images.shape[1] == frame_stack:
            B, S, H, W, C = images.shape
            x = _rearr(images, 'b s h w c -> b s c h w')
            if grayscale and C == 3:
                x = 0.299 * x[:, :, 0:1] + 0.587 * x[:, :, 1:2] + 0.114 * x[:, :, 2:3]
            x = _rearr(x, 'b s c h w -> b (s c) h w')
            return x
        # [B, H, W, C]
        if images.dim() == 4 and images.shape[-1] in [1, 3]:
            x = _rearr(images, 'b h w c -> b c h w')
            if grayscale and x.shape[1] == 3:
                x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            return x
        return images

    def build_stack_heatmaps(self, gaze_seq: torch.Tensor, frame_stack: int, center_idx: int) -> torch.Tensor:
        """
        Build per-stack heatmaps with causal aggregation along stack S.

        Args:
            gaze_seq: [B, L, P*2] or [B, L, P, 2]
            frame_stack: S
            center_idx: center time to end the stack window on

        Returns:
            heatmaps_stack: [B, S, H, W] in [0,1]
        """
        # 1) Extract [B, S, ...]
        gaze_stack = self.extract_gaze_stack_around_center(gaze_seq, center_idx=center_idx, frame_stack=frame_stack)

        # If multiscale mode with provided sigma list, use per-step gaussian blur and coeffs
        if self.temporal_mode == 'multiscale' and self.temporal_sigmas is not None and len(self.temporal_sigmas) > 0:
            device = gaze_stack.device
            B = gaze_stack.shape[0]
            S = frame_stack
            H, W = self.img_height, self.img_width

            # Build delta maps per step: [B, S, 1, H, W]
            delta_stack = self._build_delta_from_gaze_stack(gaze_stack, H, W)

            # Prepare per-step coeffs
            coeffs_list = [1.0] * S
            if self.temporal_coeffs is not None and len(self.temporal_coeffs) > 0:
                for s in range(S):
                    idx = min(self.temporal_offset_start + s, len(self.temporal_coeffs) - 1)
                    coeffs_list[s] = float(self.temporal_coeffs[idx])

            # Apply per-step gaussian blur with sigma list, then causal sum
            blurred_steps = torch.zeros_like(delta_stack)
            for s in range(S):
                sigma = float(self.temporal_sigmas[min(self.temporal_offset_start + s, len(self.temporal_sigmas) - 1)])
                kernel_1d = self._get_or_make_kernel_1d(sigma, device=device, dtype=delta_stack.dtype)
                k = kernel_1d.view(1, 1, -1, 1)
                # blur current step
                cur = delta_stack[:, s]  # [B,1,H,W]
                cur = F.conv2d(cur, k, padding=(0, k.shape[2] // 2))
                kt = k.permute(0, 1, 3, 2)
                cur = F.conv2d(cur, kt, padding=(k.shape[2] // 2, 0))
                # scale by coeff
                blurred_steps[:, s] = cur * coeffs_list[s]

            # Causal accumulation
            agg_stack = torch.zeros_like(blurred_steps)
            for s in range(S):
                agg_stack[:, s] = blurred_steps[:, :s+1].sum(dim=1)

            # Normalize per-step to [0,1]
            amin = agg_stack.amin(dim=(-2, -1), keepdim=True)
            amax = agg_stack.amax(dim=(-2, -1), keepdim=True)
            agg_stack = (agg_stack - amin) / (amax - amin + 1e-8)
            return agg_stack.squeeze(2)  # [B, S, H, W]

        # Fallback: base per-step heatmaps + alpha-decay causal aggregation
        base_stack_heat = self.forward(gaze_stack)  # [B,S,1,H,W]
        if base_stack_heat.dim() != 5:
            base_stack_heat = base_stack_heat.unsqueeze(1)
        alpha = float(self.temporal_alpha)
        B, S = base_stack_heat.shape[0], base_stack_heat.shape[1]
        agg_stack = torch.zeros_like(base_stack_heat)
        for s in range(S):
            if s == 0:
                agg_stack[:, s] = base_stack_heat[:, s]
            else:
                coeffs = torch.tensor([alpha ** (s - j) for j in range(s + 1)], device=base_stack_heat.device, dtype=base_stack_heat.dtype)
                coeffs = coeffs.view(1, s + 1, 1, 1, 1)
                agg_stack[:, s] = (base_stack_heat[:, :s+1] * coeffs).sum(dim=1)
        amin = agg_stack.amin(dim=(-2, -1), keepdim=True)
        amax = agg_stack.amax(dim=(-2, -1), keepdim=True)
        agg_stack = (agg_stack - amin) / (amax - amin + 1e-8)
        return agg_stack.squeeze(2)

    def _get_or_make_kernel_1d(self, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return cached 1D Gaussian kernel for a given sigma (on correct device / dtype)."""
        key = float(sigma)
        k = self._sigma_kernel_cache.get(key, None)
        if k is None:
            size = int(4 * sigma + 1)
            if size % 2 == 0:
                size += 1
            x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
            k = torch.exp(-x ** 2 / (2 * sigma ** 2))
            k = (k / (k.sum() + 1e-8)).to(dtype)
            self._sigma_kernel_cache[key] = k
        else:
            # ensure device / dtype match
            if k.device != device or k.dtype != dtype:
                k = k.to(device=device, dtype=dtype)
                self._sigma_kernel_cache[key] = k
        return k

    def _build_delta_from_gaze_stack(self, gaze_stack: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Vectorized delta map construction for [B,S,P*2] or [B,S,P,2] -> [B,S,1,H,W]."""
        from einops import rearrange as _rearr
        if gaze_stack.dim() == 3 and gaze_stack.shape[-1] == self.maxpoints * 2:
            gaze_stack = _rearr(gaze_stack, 'b s (p c) -> b s p c', p=self.maxpoints, c=2)
        B, S, P, _ = gaze_stack.shape
        device = gaze_stack.device
        valid_mask = (gaze_stack[..., 0] >= 0) & (gaze_stack[..., 1] >= 0)
        x_coords = (gaze_stack[..., 0].clamp(0, 1) * (W - 1)).long().clamp(0, W - 1)
        y_coords = (gaze_stack[..., 1].clamp(0, 1) * (H - 1)).long().clamp(0, H - 1)
        weights = valid_mask.float()
        num_samples = B * S
        delta = torch.zeros(num_samples, H * W, device=device, dtype=torch.float32)
        lin_idx = (y_coords * W + x_coords).view(num_samples, P)
        w_flat = weights.view(num_samples, P)
        delta.scatter_add_(dim=1, index=lin_idx, src=w_flat)
        delta = _rearr(delta, '(b s) (h w) -> b s 1 h w', b=B, s=S, h=H, w=W)
        return delta

    def prepare_for_bc(self,
                       obs_image_seq: torch.Tensor,
                       gaze_seq: torch.Tensor,
                       frame_stack: int,
                       grayscale: bool = False,
                       aggregate_stack: bool = True):
        """
        One-call API for BC training to get encoder-ready images and stack-aggregated gaze heatmaps.

        Args:
            obs_image_seq: [B, L, H, W, C]
            gaze_seq: [B, L, P*2] or [B, L, P, 2]
            frame_stack: S
            seq_length: configured sequence length L (used to pick center)
            grayscale: whether to convert RGB to 1-channel

        Returns:
            obs_image: [B, S*C', H, W]
            gaze_heatmaps: [B, S, H, W]
            center_idx: int
        """
        # Determine center index: use last available step (ignore external seq_length)
        center_idx = (obs_image_seq.shape[1] - 1) if obs_image_seq.dim() > 1 else 0

        # Build image stack window and format to channels-first
        imgs_stack = self.extract_image_stack_around_center(obs_image_seq, center_idx=center_idx, frame_stack=frame_stack)
        obs_image = self._format_obs_image(imgs_stack, frame_stack=frame_stack, grayscale=grayscale)

        # Build gaze heatmaps along stack
        if aggregate_stack:
            gaze_heatmaps = self.build_stack_heatmaps(gaze_seq, frame_stack=frame_stack, center_idx=center_idx)
        else:
            # Base per-stack heatmaps without causal aggregation
            gaze_stack = self.extract_gaze_stack_around_center(gaze_seq, center_idx=center_idx, frame_stack=frame_stack)
            base_stack = self.forward(gaze_stack)  # [B, S, 1, H, W]
            if base_stack.dim() == 5:
                gaze_heatmaps = base_stack.squeeze(2)  # [B, S, H, W]
            else:
                # Robustness: if [B,1,H,W], tile to S then squeeze
                gaze_heatmaps = base_stack
                if gaze_heatmaps.dim() == 4 and gaze_heatmaps.shape[1] == 1 and frame_stack > 1:
                    gaze_heatmaps = gaze_heatmaps.repeat(1, frame_stack, 1, 1)
        return obs_image, gaze_heatmaps, center_idx

    def prepare_for_gaze_predictor(self,
                                   obs_image_seq: torch.Tensor,
                                   gaze_seq: torch.Tensor,
                                   frame_stack: int,
                                   grayscale: bool = False):
        """
        One-call API for gaze predictor training.
        Builds an image stack [B, S, H, W, C] and aggregates gaze along stack to [B, 1, H, W]
        using forward_temporal centered at the last stack frame.
        """
        center_idx = (obs_image_seq.shape[1] - 1) if obs_image_seq.dim() > 1 else 0
        imgs_stack = self.extract_image_stack_around_center(obs_image_seq, center_idx=center_idx, frame_stack=frame_stack)
        obs_image = self._format_obs_image(imgs_stack, frame_stack=frame_stack, grayscale=grayscale)

        # Extract gaze stack and apply causal aggregation along stack to the last step only
        gaze_stack_agg = self.build_stack_heatmaps(gaze_seq, frame_stack=frame_stack, center_idx=center_idx)  # [B,S,H,W]
        last = gaze_stack_agg[:, -1]  # [B,H,W]
        return obs_image, last.unsqueeze(1), center_idx  # [B,1,H,W]


# ============================================================================
# GABRIL Dataset Wrapper for Robomimic
# ============================================================================

class GABRILSequenceDataset(SequenceDataset):
    """
    Thin wrapper over SequenceDataset for potential future extensions.
    Currently, it behaves identically to the parent class and does not
    perform any gaze heatmap generation. That should be handled in the
    training loop after fetching batches.
    """
    def __init__(self, hdf5_path, obs_keys, dataset_keys, action_keys=None, action_config=None, **kwargs):
        # Set default action_keys and action_config if not provided
        if action_keys is None:
            action_keys = ['actions']
        if action_config is None:
            action_config = {
                'actions': {
                    'normalization': None
                }
            }
        super().__init__(
            hdf5_path=hdf5_path, 
            obs_keys=obs_keys, 
            dataset_keys=dataset_keys,
            action_keys=action_keys,
            action_config=action_config,
            **kwargs
        )


# Testing utilities and CLI main removed (unused)
