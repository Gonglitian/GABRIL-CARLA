#!/usr/bin/env python3
"""
Shared preprocessing utilities for image and gaze tensors
"""

import torch
from einops import rearrange


def format_obs_image(obs_image: torch.Tensor, frame_stack: int, grayscale: bool) -> torch.Tensor:
    # Normalize if uint8
    if obs_image.dtype == torch.uint8:
        obs_image = obs_image.float() / 255.0

    if obs_image.dim() == 5 and obs_image.shape[1] == frame_stack:
        # [B, stack, H, W, C] -> [B, stack*C, H, W] (with optional grayscale)
        B, stack, H, W, C = obs_image.shape
        obs_image = rearrange(obs_image, 'b s h w c -> b s c h w')
        if grayscale and C == 3:
            obs_image = 0.299 * obs_image[:, :, 0:1] + 0.587 * obs_image[:, :, 1:2] + 0.114 * obs_image[:, :, 2:3]
        obs_image = rearrange(obs_image, 'b s c h w -> b (s c) h w')
        return obs_image

    if obs_image.dim() == 4:
        if obs_image.shape[1] == frame_stack and obs_image.shape[-1] in [1, 3]:
            B, stack, H, W, C = obs_image.shape
            obs_image = rearrange(obs_image, 'b s h w c -> b s c h w')
            if grayscale and C == 3:
                obs_image = 0.299 * obs_image[:, :, 0:1] + 0.587 * obs_image[:, :, 1:2] + 0.114 * obs_image[:, :, 2:3]
            obs_image = rearrange(obs_image, 'b s c h w -> b (s c) h w')
            return obs_image
        if obs_image.shape[-1] in [1, 3]:
            obs_image = rearrange(obs_image, 'b h w c -> b c h w')
            if grayscale and obs_image.shape[1] == 3:
                obs_image = 0.299 * obs_image[:, 0:1] + 0.587 * obs_image[:, 1:2] + 0.114 * obs_image[:, 2:3]
            return obs_image

    return obs_image


def format_gaze_coords(gaze_coords: torch.Tensor) -> torch.Tensor:
    # Select last frame if stacked
    if gaze_coords.dim() == 3:
        gaze_coords = gaze_coords[:, -1]
    return gaze_coords


