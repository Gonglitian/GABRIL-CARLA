#!/usr/bin/env python3

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import hydra
from omegaconf import DictConfig, OmegaConf

# Import using absolute package paths
from vlm_gaze.data_utils import GazePreprocessor
from vlm_gaze.data_utils.gaze_utils import get_gaze_mask, apply_gmd_dropout
from vlm_gaze.models import Encoder, VectorQuantizer, weight_init
from vlm_gaze.train.common.preprocess import format_obs_image, format_gaze_coords
from vlm_gaze.train.common.base_trainer import BaseTrainer
from vlm_gaze.train.common.distributed import wrap_ddp, destroy_distributed_if_initialized


class BCTrainer(BaseTrainer):
    """Trainer class for behavior cloning with gaze regularization"""
    
    def __init__(self, cfg: DictConfig):
        # defers to BaseTrainer for setup; after init, create preprocessor/criterion
        self.gaze_preprocessor = None
        super().__init__(cfg)
        self.gaze_preprocessor = GazePreprocessor(
            img_height=self.cfg.data.img_height,
            img_width=self.cfg.data.img_width,
            gaze_sigma=self.cfg.gaze.mask_sigma,
            gaze_coeff=self.cfg.gaze.mask_coeff,
            maxpoints=self.cfg.gaze.max_points,
            device=str(self.device)
        )
        self.criterion = nn.MSELoss()

    def _print_rank0(self, message: str):
        """
        Print message only on rank 0
        Args:
            message (str): Message to print
        """
        if self.rank == 0:
            print(message)
        
    # Base hooks
    def build_models(self):
        cfg = self.cfg.model
        # channels 
        coeff = 2 if self.cfg.gaze.method == 'ViSaRL' else 1
        input_channels = coeff * cfg.frame_stack * (1 if cfg.grayscale else 3)
        self.encoder = Encoder(
            input_channels=input_channels,
            embedding_dim=cfg.embedding_dim,
            num_hiddens=cfg.num_hiddens,
            num_residual_layers=cfg.num_residual_layers,
            num_residual_hiddens=cfg.num_residual_hiddens,
        ).to(self.device)
        self.encoder_agil = None
        if self.cfg.gaze.method == 'AGIL':
            self.encoder_agil = Encoder(
                input_channels=cfg.frame_stack * (1 if cfg.grayscale else 3),
                embedding_dim=cfg.embedding_dim,
                num_hiddens=cfg.num_hiddens,
                num_residual_layers=cfg.num_residual_layers,
                num_residual_hiddens=cfg.num_residual_hiddens,
            ).to(self.device)
        encoder_output_dim = 20 * 38 * cfg.embedding_dim
        self.pre_actor = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(encoder_output_dim, cfg.z_dim)).to(self.device)
        self.pre_actor.apply(weight_init)
        self.actor = nn.Sequential(nn.Linear(cfg.z_dim, cfg.z_dim), nn.ReLU(), nn.Linear(cfg.z_dim, self.cfg.data.action_dim)).to(self.device)
        self.actor.apply(weight_init)
        self.gril_gaze_coord_predictor = None
        if self.cfg.gaze.method == 'GRIL':
            self.gril_gaze_coord_predictor = nn.Sequential(nn.Linear(cfg.z_dim, cfg.z_dim), nn.ReLU(), nn.Linear(cfg.z_dim, self.cfg.gaze.max_points * 2)).to(self.device)
            self.gril_gaze_coord_predictor.apply(weight_init)
        self.quantizer = None
        if self.cfg.dropout.method == 'Oreo':
            self.quantizer = VectorQuantizer(cfg.embedding_dim, self.cfg.dropout.num_embeddings, 0.25).to(self.device)
            vqvae_path = Path(self.cfg.dropout.vqvae_path)
            if vqvae_path.exists():
                for p in self.quantizer.parameters():
                    p.requires_grad = False
                vqvae_dict = torch.load(vqvae_path, map_location="cpu", weights_only=True)
                self.encoder.load_state_dict({k[9:]: v for k, v in vqvae_dict.items() if "_encoder" in k})
                self.quantizer.load_state_dict({k[11:]: v for k, v in vqvae_dict.items() if "_quantizer" in k})
                self._print_rank0(f"Loaded VQ-VAE from {vqvae_path}")
            else:
                self._print_rank0(f"Warning: VQ-VAE model not found at {vqvae_path}")
        if self.cfg.training.use_compile and hasattr(torch, 'compile'):
            backend = getattr(self.cfg.training, 'compile_backend', 'inductor')
            mode = getattr(self.cfg.training, 'compile_mode', 'default')
            self._print_rank0(f"Compiling models... backend={backend}, mode={mode}")
            def _compile_safe(module):
                try:
                    return torch.compile(module, backend=backend, mode=mode)
                except TypeError:
                    return torch.compile(module)
            self.encoder = _compile_safe(self.encoder)
            self.pre_actor = _compile_safe(self.pre_actor)
            self.actor = _compile_safe(self.actor)
            if self.encoder_agil is not None:
                self.encoder_agil = _compile_safe(self.encoder_agil)
            if self.gril_gaze_coord_predictor is not None:
                self.gril_gaze_coord_predictor = _compile_safe(self.gril_gaze_coord_predictor)
        if self.is_distributed:
            self.encoder = wrap_ddp(self.encoder, self.cfg.training, self.local_rank)
            self.pre_actor = wrap_ddp(self.pre_actor, self.cfg.training, self.local_rank)
            self.actor = wrap_ddp(self.actor, self.cfg.training, self.local_rank)
            if self.encoder_agil is not None:
                self.encoder_agil = wrap_ddp(self.encoder_agil, self.cfg.training, self.local_rank)
            if self.gril_gaze_coord_predictor is not None:
                self.gril_gaze_coord_predictor = wrap_ddp(self.gril_gaze_coord_predictor, self.cfg.training, self.local_rank)

    def get_optim_params(self):
        params = list(self.encoder.parameters()) + list(self.pre_actor.parameters()) + list(self.actor.parameters())
        if self.encoder_agil is not None:
            params += list(self.encoder_agil.parameters())
        if self.gril_gaze_coord_predictor is not None:
            params += list(self.gril_gaze_coord_predictor.parameters())
        return params
    
    def compute_gaze_regularization_loss(
        self, 
        z: torch.Tensor, 
        gg: torch.Tensor, 
        gc: torch.Tensor,
        xx: torch.Tensor,
        ivg: torch.Tensor
    ) -> torch.Tensor:
        """Compute gaze regularization loss based on the configured method"""
        
        reg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        if self.cfg.gaze.method in ['Teacher', 'Reg']:
            # Gaze regularization loss
            with torch.no_grad():
                g1 = gg[:, -1:, :, :][ivg > 0].float()
            
            g2 = get_gaze_mask(z, self.cfg.gaze.beta, (xx.shape[-2], xx.shape[-1]))[ivg > 0]
            
            if self.cfg.gaze.prob_dist_type in ['TV', 'JS', 'KL']:
                # Normalize to probability distributions
                g1_sum = g1.sum(dim=(-1, -2, -3), keepdim=True) + 1e-8
                g2_sum = g2.sum(dim=(-1, -2, -3), keepdim=True) + 1e-8
                g1 = g1 / g1_sum.detach()
                g2 = g2 / g2_sum.detach()
            
            def KL(a, b):
                return (a * torch.log((a + 1e-6) / (b + 1e-6))).sum(dim=(1,2,3)).mean(0)
            
            if self.cfg.gaze.prob_dist_type == 'KL':
                reg_loss = KL(g1, g2)
            elif self.cfg.gaze.prob_dist_type == 'TV':
                reg_loss = (g1 - g2).abs().sum(dim=(1,2,3)).mean(0)
            elif self.cfg.gaze.prob_dist_type == 'JS':
                reg_loss = 0.5 * (KL(g1, (g1+g2)/2) + KL(g2, (g1+g2)/2))
            elif self.cfg.gaze.prob_dist_type == 'MSE':
                reg_loss = F.mse_loss(g1, g2)
            else:
                raise ValueError(f'Invalid prob_dist_type: {self.cfg.gaze.prob_dist_type}')
        
        elif self.cfg.gaze.method == 'Contrastive':
            positive_images = gg[ivg > 0][:, :self.cfg.data.frame_stack] / 255.0
            negative_images = gg[ivg > 0][:, self.cfg.data.frame_stack:] / 255.0
            z_plus = self.encoder(positive_images)
            z_minus = self.encoder(negative_images)
            t1 = torch.linalg.vector_norm(z[ivg > 0] - z_plus, dim=(1, 2, 3)) ** 2
            t2 = torch.linalg.vector_norm(z[ivg > 0] - z_minus, dim=(1, 2, 3)) ** 2
            reg_loss = torch.max(torch.zeros_like(t1), t1 - t2 + self.cfg.gaze.contrastive_threshold).mean()
        
        elif self.cfg.gaze.method == 'GRIL' and self.gril_gaze_coord_predictor is not None:
            if ivg.sum() > 0:
                # z should already be flattened from pre_actor
                gaze_coord_pred = self.gril_gaze_coord_predictor(z[ivg > 0])
                gc_sel = gc[ivg > 0].float()
                if gc_sel.dim() == 3:
                    gaze_target = rearrange(gc_sel, 'b p c -> b (p c)')
                else:
                    gaze_target = gc_sel
                gaze_coord_loss = F.mse_loss(gaze_coord_pred, gaze_target) + 1e-8
                reg_loss = torch.clamp(gaze_coord_loss, min=0.0, max=100.0)
        
        return reg_loss
    
    def set_train_mode(self):
        self.encoder.train(); self.pre_actor.train(); self.actor.train()
        if self.encoder_agil is not None:
            self.encoder_agil.train()
        if self.gril_gaze_coord_predictor is not None:
            self.gril_gaze_coord_predictor.train()

    def compute_loss(self, batch):
        obs_image = batch['obs']['image'].to(self.device, non_blocking=True)
        gaze_key = getattr(self.cfg.data, 'gaze_key', 'gaze_coords')
        gaze_coords = batch['obs'][gaze_key].to(self.device, non_blocking=True)
        actions = batch['actions'].to(self.device, non_blocking=True)
        obs_image = format_obs_image(obs_image, self.cfg.data.frame_stack, self.cfg.model.grayscale)
        gaze_coords = format_gaze_coords(gaze_coords)
        # 如果 actions 带有时间或堆叠维度（[B, T, A] 或 [B, S, A]），取最后一步对齐网络输出形状 [B, A]
        if actions.dim() == 3:
            actions = actions[:, -1, :]
        batch_size = obs_image.shape[0]
        with torch.no_grad():
            if gaze_coords.dim() == 2:
                # 输入为 [B, P]（每样本单步），在 dim=1 处增加时间维 -> [B, 1, P]
                gaze_coords_for_preprocessor = gaze_coords.unsqueeze(1)
            else:
                gaze_coords_for_preprocessor = gaze_coords
            gaze_heatmaps = self.gaze_preprocessor(gaze_coords_for_preprocessor)
            if gaze_heatmaps.dim() == 5:
                gaze_heatmaps = gaze_heatmaps[:, 0]
            if self.cfg.data.frame_stack > 1:
                gaze_heatmaps = repeat(gaze_heatmaps, 'b 1 h w -> b s h w', s=self.cfg.data.frame_stack)
        ivg = torch.ones(batch_size, dtype=torch.float32, device=self.device)
        with torch.amp.autocast('cuda', enabled=self.cfg.training.use_amp):
            xx = obs_image; gg = gaze_heatmaps; gc = gaze_coords
            gaze_dropout_mask = gg if self.cfg.dropout.method == 'IGMD' else None
            z = self.encoder(xx if self.cfg.gaze.method != 'Mask' else xx * gg, dropout_mask=gaze_dropout_mask)
            if self.cfg.gaze.method == 'ViSaRL':
                z = self.encoder(torch.cat([obs_image, gg], dim=1), dropout_mask=gaze_dropout_mask)
            if self.cfg.gaze.method == 'AGIL' and self.encoder_agil is not None:
                z = (z + self.encoder_agil(obs_image * gg)) / 2
            if self.cfg.dropout.method == 'GMD':
                z = apply_gmd_dropout(z, gg, test_mode=False)
            elif self.cfg.dropout.method == 'Oreo' and self.quantizer is not None:
                with torch.no_grad():
                    _, _, _, _, encoding_indices, _ = self.quantizer(z)
                    prob = torch.ones(batch_size * self.cfg.dropout.oreo_num_mask, self.cfg.dropout.num_embeddings) * (1 - self.cfg.dropout.oreo_prob)
                    code_mask = torch.bernoulli(prob).to(self.device)
                    encoding_indices_flatten = rearrange(encoding_indices, 'b h w -> (b h w)')
                    encoding_indices_onehot = torch.zeros((len(encoding_indices_flatten), self.cfg.dropout.num_embeddings), device=encoding_indices_flatten.device)
                    encoding_indices_onehot.scatter_(1, encoding_indices_flatten.unsqueeze(1), 1)
                    encoding_indices_onehot = rearrange(encoding_indices_onehot, '(b hw) e -> b hw e', b=batch_size)
                    mask = (code_mask.unsqueeze(1) * repeat(encoding_indices_onehot, 'b hw e -> (m b) hw e', m=self.cfg.dropout.oreo_num_mask)).sum(2)
                    mask = rearrange(mask, 'b (h w) -> b h w', h=20, w=38)
                mask = rearrange(mask, 'b h w -> b 1 h w')
                z = repeat(z, 'b c h w -> (m b) c h w', m=self.cfg.dropout.oreo_num_mask) * mask
                z = z / (1.0 - self.cfg.dropout.oreo_prob)
                actions = repeat(actions, 'b a -> (m b) a', m=self.cfg.dropout.oreo_num_mask)
            z_flat = self.pre_actor(z)
            logits = self.actor(z_flat)
            actor_loss = self.criterion(logits, actions)
            if self.cfg.gaze.method == 'GRIL' and self.gril_gaze_coord_predictor is not None:
                reg_loss = self.compute_gaze_regularization_loss(z_flat, gg, gc, obs_image, ivg)
            else:
                reg_loss = self.compute_gaze_regularization_loss(z, gg, gc, obs_image, ivg)
            total = self.cfg.gaze.lambda_weight * reg_loss + actor_loss
        metrics = {"Loss/actor": float(actor_loss.item()), "Loss/reg": float(reg_loss.item())}
        return total, batch_size, metrics

    def save_for_epoch(self, epoch: int):
        # 分模块分别保存，命名与旧版保持一致
        import torch
        enc_to_save = self.encoder.module if hasattr(self.encoder, 'module') else self.encoder
        pre_to_save = self.pre_actor.module if hasattr(self.pre_actor, 'module') else self.pre_actor
        act_to_save = self.actor.module if hasattr(self.actor, 'module') else self.actor
        torch.save(enc_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_encoder.pth")
        torch.save(act_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_actor.pth")
        torch.save(pre_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_pre_actor.pth")
        if self.gril_gaze_coord_predictor is not None:
            gril_to_save = self.gril_gaze_coord_predictor.module if hasattr(self.gril_gaze_coord_predictor, 'module') else self.gril_gaze_coord_predictor
            torch.save(gril_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_gril_gaze_coord_predictor.pth")
        if self.encoder_agil is not None:
            agil_to_save = self.encoder_agil.module if hasattr(self.encoder_agil, 'module') else self.encoder_agil
            torch.save(agil_to_save.state_dict(), self.checkpoint_dir / f"ep{epoch}_encoder_agil.pth")
        if self.cfg.logging.save_params:
            # Save params expected by eval/my_agents/bc_agent.py
            params = {
                'gaze_method': self.cfg.gaze.method,
                'dp_method': self.cfg.dropout.method,
                'grayscale': self.cfg.model.grayscale,
                'stack': self.cfg.model.frame_stack,
                'embedding_dim': self.cfg.model.embedding_dim,
                'num_embeddings': self.cfg.dropout.num_embeddings,
                'num_hiddens': self.cfg.model.num_hiddens,
                'num_residual_layers': self.cfg.model.num_residual_layers,
                'num_residual_hiddens': self.cfg.model.num_residual_hiddens,
                'z_dim': self.cfg.model.z_dim,
                # Path for optional gaze predictor used by some eval agents; left empty here
                'gaze_predictor_path': '/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/trained_models/gaze_predictor_models/Mixed_/2025_08_03_15_35_15_s1_n200_stack2_grayTrue_bs3000_lr0.001_step50_pseudo_amp_compile/model_ep150.torch',
                'models_path': str(self.checkpoint_dir),
                'epochs': epoch,
                'action_dim': self.cfg.data.action_dim,
            }
            self.experiment.save_params_json(params)
    
@hydra.main(version_base=None, config_path="../configs", config_name="train_bc")
def main(cfg: DictConfig):
    """Main entry point"""
    import os
    if 'RANK' not in os.environ or os.environ.get('RANK', '0') == '0':
        print(OmegaConf.to_yaml(cfg))
    
    trainer = BCTrainer(cfg)
    trainer.train()
    
    import os
    if 'RANK' not in os.environ or os.environ.get('RANK', '0') == '0':
        print("Training completed!")
    # Ensure DDP shuts down cleanly to avoid NCCL resource leak warnings
    destroy_distributed_if_initialized()


if __name__ == "__main__":
    # Print config only on rank 0
    import os
    should_print = 'RANK' not in os.environ or os.environ.get('RANK', '0') == '0'
    if should_print:
        pass
    main()
