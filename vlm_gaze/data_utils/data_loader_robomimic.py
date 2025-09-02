#!/usr/bin/env python3
"""
Robomimic SequenceDataset Integration for GABRIL-CARLA
Provides dataset/dataloader functionality using robomimic's efficient SequenceDataset
"""

import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Optional
import argparse
from tqdm import tqdm
from einops import rearrange

# Import robomimic (external dependency or vendored under vlm_gaze.robomimic)
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.obs_utils as ObsUtils

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
        device: str = 'cuda'
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.maxpoints = maxpoints
        self.gaze_sigma = gaze_sigma
        self.gaze_coeff = gaze_coeff
        self.device = device
        
        # Pre-compute Gaussian kernel
        kernel_size = int(4 * gaze_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * gaze_sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        self.register_buffer('kernel_1d', kernel_1d.to(device))
    
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

        # Exponential decay weights per valid order within each (B,T)
        # rank = cumsum(valid_mask) - 1 over P, only for valid entries
        rank = torch.cumsum(valid_mask.to(torch.int64), dim=2) - 1  # [B, T, P]
        rank = rank.clamp(min=0).to(torch.float32)
        weights = torch.pow(torch.tensor(self.gaze_coeff, device=device, dtype=torch.float32), rank) * valid_mask.float()

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
        # Normalize each heatmap
        max_vals = blurred.amax(dim=(2, 3), keepdim=True)
        normalized = blurred / (max_vals + 1e-8)
        heatmaps = rearrange(normalized, '(b t) c h w -> b t c h w', b=B, t=T)

        return heatmaps


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


# ============================================================================
# Testing and Benchmarking
# ============================================================================

def test_robomimic_dataloader(
    hdf5_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda',
    num_batches: int = 10,
    seq_length: int = 1,
    frame_stack: int = 2,
    cache_mode: str = 'low_dim'
):
    """Test robomimic SequenceDataset performance"""
    print("\n" + "="*60)
    print("Testing Robomimic SequenceDataset Integration")
    print("="*60)
    
    # Initialize ObsUtils with observation modalities
    # This ensures gaze_coords is properly registered as low_dim for caching
    obs_modality_specs = {
        'obs': {
            'rgb': ['image'],
            'low_dim': ['gaze_coords']
        },
        'goal': {
            'rgb': [],
            'low_dim': []
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)
    
    # Define observation keys
    obs_keys = ['image', 'gaze_coords']
    action_keys = ['actions']
    dataset_keys = ['actions', 'rewards', 'dones']
    
    # Action configuration (no normalization for testing)
    action_config = {
        'actions': {
            'normalization': None
        }
    }
    
    # Create dataset
    dataset = GABRILSequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=obs_keys,
        action_keys=action_keys,
        dataset_keys=dataset_keys,
        action_config=action_config,
        frame_stack=frame_stack,
        seq_length=seq_length,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=cache_mode,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        load_next_obs=True
    )
    
    # Set identity action normalization to skip full dataset scan
    # This avoids the expensive initial scan even when normalization is None.
    # Note: when cache_mode == "all", internal hdf5_cache is removed, so infer
    # action dim from a cached sample instead of touching the file handle.
    sample0 = dataset[0]
    a_dim = sample0["actions"].shape[-1]
    identity_stats = OrderedDict(actions={
        "scale": np.ones((1, a_dim), dtype=np.float32),
        "offset": np.zeros((1, a_dim), dtype=np.float32),
    })
    dataset.set_action_normalization_stats(identity_stats)
    print(f"Set identity action normalization stats (dim={a_dim}) to skip dataset scan")
    
    print(f"\nDataset info:")
    print(f"  Total sequences: {len(dataset)}")
    print(f"  Number of demos: {dataset.n_demos}")
    print(f"  Cache mode: {cache_mode}")
    print(f"  Frame stack: {frame_stack}")
    print(f"  Sequence length: {seq_length}")
    
    # Create dataloader (default collate_fn). Gaze heatmap should be generated
    # later in the training loop after batches are fetched on the main process / GPU
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    print(f"\nBenchmarking {num_batches} batches...")
    print(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    # Timing
    times = {
        'fetch': [],
        'transfer': [],
        'total': []
    }
    
    # Create iterator
    data_iter = iter(dataloader)
    
    # Warmup
    print("Running warmup...")
    for _ in range(2):
        try:
            _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            _ = next(data_iter)
    
    # Benchmark
    torch.cuda.synchronize()
    pbar = tqdm(total=num_batches, desc="Processing batches")
    
    for i in range(num_batches):
        # Fetch batch
        t0 = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        t1 = time.time()
        times['fetch'].append(t1 - t0)
        
        # Transfer to GPU
        if device.startswith('cuda'):
            for key in batch:
                if isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if torch.is_tensor(batch[key][sub_key]):
                            batch[key][sub_key] = batch[key][sub_key].to(device, non_blocking=True)
                elif torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device, non_blocking=True)
            torch.cuda.synchronize()
        t2 = time.time()
        times['transfer'].append(t2 - t1)
        times['total'].append(t2 - t0)
        
        # Update progress
        avg_time = np.mean(times['total']) * 1000
        throughput = batch_size / np.mean(times['total'])
        pbar.set_postfix({
            'avg_ms': f'{avg_time:.1f}',
            'throughput': f'{throughput:.0f} samples/s'
        })
        pbar.update(1)
        
        if i == 0:
            print(f"\nSample batch info:")
            print(f"  Image shape: {batch['obs']['image'].shape}")
            print(f"  Gaze coords shape: {batch['obs']['gaze_coords'].shape}")
            if 'gaze_heatmap' in batch['obs']:
                print(f"  Gaze heatmap shape: {batch['obs']['gaze_heatmap'].shape}")
            print(f"  Actions shape: {batch['actions'].shape}")
    
    pbar.close()
    
    # Statistics
    print("\n" + "-"*40)
    print("Performance Statistics (ms):")
    print("-"*40)
    
    for key in ['fetch', 'transfer', 'total']:
        values = np.array(times[key]) * 1000
        print(f"{key.capitalize():12s}: "
              f"mean={np.mean(values):6.2f}, "
              f"std={np.std(values):6.2f}, "
              f"min={np.min(values):6.2f}, "
              f"max={np.max(values):6.2f}")
    
    total_samples = num_batches * batch_size
    total_time = sum(times['total'])
    throughput = total_samples / total_time
    print(f"\nThroughput: {throughput:.1f} samples/sec")
    print(f"Time per batch: {total_time/num_batches*1000:.1f} ms")
    
    return times


# ============================================================================
# Main (Testing Only)
# ============================================================================

def main():
    """Main function for testing dataloader"""
    parser = argparse.ArgumentParser(description='Test Robomimic SequenceDataset Integration')
    parser.add_argument('--hdf5-path', type=str, default='/data3/vla-reasoning/dataset/bench2drive220_robomimic.hdf5',
                        help='Path to robomimic HDF5 file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--num-batches', type=int, default=20, help='Number of batches to test')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seq-length', type=int, default=1, help='Sequence length')
    parser.add_argument('--frame-stack', type=int, default=2, help='Frame stack')
    parser.add_argument('--cache-mode', type=str, default='all', 
                        choices=['all', 'low_dim', None], help='HDF5 cache mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.hdf5_path):
        print(f"Error: HDF5 file not found: {args.hdf5_path}")
        print("Please run data_convert.py first to create the robomimic format HDF5 file")
        return
    
    test_robomimic_dataloader(
        hdf5_path=args.hdf5_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        num_batches=args.num_batches,
        seq_length=args.seq_length,
        frame_stack=args.frame_stack,
        cache_mode=args.cache_mode
    )


if __name__ == "__main__":
    main()