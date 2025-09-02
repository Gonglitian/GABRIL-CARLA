#!/usr/bin/env python3
"""
Training script for Gaze Predictor using Hydra configuration.
This subclass now only implements the 5 required hooks for BaseTrainer:
 - build_models, get_optim_params, set_train_mode, compute_loss, save_for_epoch
All training loop, logging, checkpoint directory management are handled by BaseTrainer.
"""


import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from einops import rearrange

# Import using absolute package paths
from vlm_gaze.data_utils import GazePreprocessor
from vlm_gaze.models import AutoEncoder, Encoder, Decoder
from vlm_gaze.train.common.preprocess import format_obs_image, format_gaze_coords
from vlm_gaze.train.common.base_trainer import BaseTrainer

class GazePredictorTrainer(BaseTrainer):
    """Trainer class for gaze predictor model"""
    
    def __init__(self, cfg: DictConfig):
        self.gaze_preprocessor = None
        super().__init__(cfg)
        # preprocessor after device ready
        self.gaze_preprocessor = GazePreprocessor(
            img_height=self.cfg.data.img_height,
            img_width=self.cfg.data.img_width,
            gaze_sigma=self.cfg.gaze.sigma,
            gaze_coeff=self.cfg.gaze.coeff,
            maxpoints=self.cfg.gaze.max_points,
            device=str(self.device)
        )
    
    def _print_rank0(self, message: str):
        if self.rank == 0:
            print(message)
        
    # Base hooks
    def build_models(self):
        cfg = self.cfg.model
        encoder = Encoder(
            input_channels=cfg.frame_stack * (1 if cfg.grayscale else 3),
            embedding_dim=cfg.embedding_dim,
            num_hiddens=cfg.num_hiddens,
            num_residual_layers=cfg.num_residual_layers,
            num_residual_hiddens=cfg.num_residual_hiddens,
        ).to(self.device)
        decoder = Decoder(
            out_channels=1,
            embedding_dim=cfg.embedding_dim,
            num_hiddens=cfg.num_hiddens,
            num_residual_layers=cfg.num_residual_layers,
            num_residual_hiddens=cfg.num_residual_hiddens,
        ).to(self.device)
        self.model = AutoEncoder(encoder, decoder).to(self.device)
        if self.cfg.training.use_compile and hasattr(torch, 'compile'):
            backend = getattr(self.cfg.training, 'compile_backend', 'inductor')
            mode = getattr(self.cfg.training, 'compile_mode', 'default')
            self._print_rank0(f"Compiling model... backend={backend}, mode={mode}")
            try:
                self.model = torch.compile(self.model, backend=backend, mode=mode)
            except TypeError:
                self.model = torch.compile(self.model)
        if self.is_distributed:
            from vlm_gaze.train.common.distributed import wrap_ddp
            self.model = wrap_ddp(self.model, self.cfg.training, self.local_rank)

    def get_optim_params(self):
        return self.model.parameters()

    def set_train_mode(self):
        self.model.train()

    def compute_loss(self, batch):
        criterion = nn.MSELoss()
        obs_image = batch['obs']['image'].to(self.device, non_blocking=True)
        gaze_key = getattr(self.cfg.data, 'gaze_key', 'gaze_coords')
        gaze_coords = batch['obs'][gaze_key].to(self.device, non_blocking=True)
        obs_image = format_obs_image(obs_image, self.cfg.data.frame_stack, self.cfg.model.grayscale)
        gaze_coords = format_gaze_coords(gaze_coords)
        with torch.no_grad():
            if gaze_coords.dim() == 2:
                # 输入为 [B, P]（每样本单步），在 dim=1 处增加时间维 -> [B, 1, P]
                gaze_coords_for_pre = gaze_coords.unsqueeze(1)
            else:
                gaze_coords_for_pre = gaze_coords
            gaze_heatmaps = self.gaze_preprocessor(gaze_coords_for_pre)
            if gaze_heatmaps.dim() == 5:
                gaze_heatmaps = gaze_heatmaps[:, 0]
        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp):
            output = self.model(obs_image)
            loss = criterion(output, gaze_heatmaps)
        batch_size = obs_image.size(0)
        return loss, batch_size, {"loss": float(loss.item())}

    def save_for_epoch(self, epoch: int):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        self.experiment.save_checkpoint(model_to_save, epoch, final=(epoch == self.cfg.training.epochs))
        if self.cfg.logging.save_params:
            params = {
                'model_type': 'gaze_predictor',
                'grayscale': self.cfg.model.grayscale,
                'stack': self.cfg.data.frame_stack,
                'embedding_dim': self.cfg.model.embedding_dim,
                'num_hiddens': self.cfg.model.num_hiddens,
                'num_residual_layers': self.cfg.model.num_residual_layers,
                'num_residual_hiddens': self.cfg.model.num_residual_hiddens,
                'gaze_mask_sigma': self.cfg.gaze.sigma,
                'gaze_mask_coeff': self.cfg.gaze.coeff,
                'models_path': str(self.checkpoint_dir),
                'epochs': epoch,
            }
            self.experiment.save_params_json(params)


@hydra.main(version_base=None, config_path="../configs", config_name="train_gaze")
def main(cfg: DictConfig):
    """Main entry point"""
    import os
    if 'RANK' not in os.environ or os.environ.get('RANK', '0') == '0':
        print(OmegaConf.to_yaml(cfg))
    
    trainer = GazePredictorTrainer(cfg)
    trainer.train()
    
    import os
    if 'RANK' not in os.environ or os.environ.get('RANK', '0') == '0':
        print("Training completed!")


if __name__ == "__main__":
    main()
