#!/usr/bin/env python3
"""
Base training skeleton with DDP, logging, optimizer/scheduler, and epoch loop.
Subclasses must implement model setup, loss computation, and checkpoint saving.
"""

from typing import Dict, Tuple, Any, Iterable, Optional

import torch
from tqdm import tqdm

from .distributed import init_distributed_if_enabled
from .logging import ExperimentLogger
from .data import build_obs_specs, build_dataset, build_dataloader
from .optim import build_optimizer, build_scheduler


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # DDP and device
        self.is_distributed, self.rank, self.world_size, self.local_rank, self.device = \
            init_distributed_if_enabled(cfg.training)
        # Seed
        self._set_seed(cfg.training.seed)

        # Data
        self._setup_data()
        # Model
        self._setup_model()
        # Optimizer / Scheduler
        self._setup_optim_scheduler()
        # Logger
        self.experiment = ExperimentLogger(self.cfg, self.cfg.data.task, self.rank)
        self.save_dir = self.experiment.save_dir
        self.writer = self.experiment.writer
        self.checkpoint_dir = self.experiment.ckpt_dir
        self._print_rank0(f"Logging to: {self.experiment.log_dir}")
        self._print_rank0(f"Checkpoints: {self.experiment.ckpt_dir}")

        # Scaler
        self.scaler = torch.amp.GradScaler('cuda') if cfg.training.use_amp else None

    # ---------- Hooks to override ----------
    def build_models(self):
        raise NotImplementedError

    def get_optim_params(self) -> Iterable:
        raise NotImplementedError

    def set_train_mode(self):
        """Set all trainable modules to train mode (default: no-op)."""
        pass

    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        """Return (loss, batch_size, metrics_for_logging)."""
        raise NotImplementedError

    def save_for_epoch(self, epoch: int):
        """Perform checkpoint saving for this epoch."""
        raise NotImplementedError

    # ---------- Internal setup ----------
    def _set_seed(self, seed: int):
        import random, numpy as np
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _print_rank0(self, message: str):
        if self.rank == 0:
            print(message)

    def _setup_data(self):
        # Initialize obs specs with configured gaze key
        build_obs_specs(self.cfg.data)
        dataset = build_dataset(self.cfg.data)
        self._print_rank0("Dataset info:")
        self._print_rank0(f"  Total sequences: {len(dataset)}")
        self._print_rank0(f"  Number of demos: {dataset.n_demos}")
        self._print_rank0(f"  Cache mode: {self.cfg.data.cache_mode}")
        self._print_rank0(f"  Frame stack: {self.cfg.data.frame_stack}")
        # Sequence length is fixed to 1 in our pipeline; stack is handled explicitly
        self._print_rank0(f"  Gaze key: {getattr(self.cfg.data, 'gaze_key', 'gaze_coords')}")

        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=False)
        self.train_sampler = sampler
        self.train_dataset = dataset
        self.train_loader = build_dataloader(
            dataset,
            self.cfg.data,
            sampler,
            grad_accum_steps=self.cfg.training.gradient_accumulation_steps,
        )
        self._print_rank0(f"Train dataloader created with {len(self.train_loader)} batches")

    def _setup_model(self):
        self.build_models()

    def _setup_optim_scheduler(self):
        self.optimizer = build_optimizer(self.get_optim_params(), self.cfg.optimizer)
        self.scheduler, self.batch_scheduler_update = build_scheduler(
            self.optimizer,
            len(self.train_loader),
            self.cfg.training,
            self.cfg.scheduler,
            self.cfg.training.gradient_accumulation_steps,
        )

    # ---------- Training loop ----------
    def train(self):
        grad_accum_steps = self.cfg.training.gradient_accumulation_steps
        for epoch in range(self.cfg.training.epochs):
            if self.is_distributed and getattr(self, 'train_sampler', None) is not None and hasattr(self.train_sampler, 'set_epoch'):
                self.train_sampler.set_epoch(epoch)

            self.set_train_mode()
            epoch_total = 0.0
            epoch_count = 0
            metrics_sums = {}
            pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.cfg.training.epochs}") if self.rank == 0 else None
            self.optimizer.zero_grad(set_to_none=True)

            for i, batch in enumerate(self.train_loader):
                loss, batch_size, metrics = self.compute_loss(batch)
                # divide for grad accumulation
                loss_to_back = loss / grad_accum_steps
                if self.cfg.training.use_amp:
                    self.scaler.scale(loss_to_back).backward()
                else:
                    loss_to_back.backward()

                if (i + 1) % grad_accum_steps == 0:
                    if self.cfg.training.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        # torch.nn.utils.clip_grad_norm_(self.get_optim_params(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # torch.nn.utils.clip_grad_norm_(self.get_optim_params(), max_norm=1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.batch_scheduler_update and self.scheduler:
                        self.scheduler.step()

                epoch_total += float(loss.item()) * batch_size
                epoch_count += batch_size
                # accumulate metrics
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        try:
                            metrics_sums[k] = metrics_sums.get(k, 0.0) + float(v) * batch_size
                        except Exception:
                            pass
                if self.rank == 0 and pbar is not None:
                    pbar.set_postfix({'loss': f'{(epoch_total/epoch_count):.4f}'})
                    pbar.update(1)

            if epoch_count % grad_accum_steps != 0:
                if self.cfg.training.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.rank == 0 and pbar is not None:
                pbar.close()

            avg_loss = epoch_total / max(1, epoch_count)
            if not self.batch_scheduler_update and self.scheduler:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self._print_rank0(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={current_lr:.6f}")
            if self.writer is not None:
                self.writer.add_scalar("Loss/epoch", avg_loss, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)
                # log averaged metrics
                if metrics_sums:
                    for k, total_v in metrics_sums.items():
                        self.writer.add_scalar(k, total_v / max(1, epoch_count), epoch)
                self.writer.flush()

            if (epoch + 1) % self.cfg.training.save_interval == 0 or (epoch + 1) == self.cfg.training.epochs:
                if self.rank == 0:
                    self.save_for_epoch(epoch + 1)
