#!/usr/bin/env python3
"""
Common data builders: obs specs, dataset and dataloader
"""

from typing import Optional

import robomimic.utils.obs_utils as ObsUtils
from torch.utils.data import DataLoader

from vlm_gaze.data_utils import GABRILSequenceDataset


def build_obs_specs(cfg_data):
    """Initialize ObsUtils with the chosen gaze key."""
    gaze_key = getattr(cfg_data, 'gaze_key', 'gaze_coords')
    obs_modality_specs = {
        'obs': {
            'rgb': ['image'],
            'low_dim': [gaze_key],
        },
        'goal': {'rgb': [], 'low_dim': []},
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)


def build_dataset(cfg_data):
    gaze_key = getattr(cfg_data, 'gaze_key', 'gaze_coords')
    obs_keys = ['image', gaze_key]
    dataset_keys = ['actions', 'rewards', 'dones']
    action_keys = ['actions']
    action_config = {'actions': {'normalization': None}}

    dataset = GABRILSequenceDataset(
        hdf5_path=cfg_data.hdf5_path,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        action_keys=action_keys,
        action_config=action_config,
        frame_stack=cfg_data.frame_stack,
        # 仅依赖 stack 维：将 seq_length 固定为 1，
        # 使 robomimic 返回长度恰为 S 的时间轴（含前向 pad），后续在训练侧当作 stack 维使用
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=cfg_data.cache_mode,
        hdf5_cache_getitem=getattr(cfg_data, 'cache_getitem', False),
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        load_next_obs=True,
        filter_by_attribute=None,
        # Limit number of demos loaded according to config
        demo_limit=getattr(cfg_data, 'num_episodes', None),
    )
    return dataset


def build_dataloader(dataset, cfg_data, sampler: Optional[object], grad_accum_steps: int = 1):
    loader = DataLoader(
        dataset,
        batch_size=cfg_data.batch_size // max(1, grad_accum_steps),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg_data.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg_data.num_workers > 0 else False,
        prefetch_factor=cfg_data.prefetch_factor if cfg_data.num_workers > 0 else None,
    )
    return loader
