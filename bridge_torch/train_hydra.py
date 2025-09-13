import os
import json
import signal
import sys
from functools import partial
import logging

import numpy as np
import torch
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from torch.amp import autocast, GradScaler

from models.encoders import build_encoder
from agents.bc import BCAgent, BCAgentConfig, MLP as MLP_BC, GaussianPolicy
from data.bridge_numpy import build_bridge_dataset, make_dataloader
from torch.utils.data.distributed import DistributedSampler
from common.wandb import WandBLogger
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def _prep_device(device_flag: str) -> torch.device:
    if device_flag == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def save_checkpoint(save_dir: str, agent, step: int):
    _ensure_dir(save_dir)
    path = os.path.join(save_dir, f"ckpt_{step}.pt")
    model_state = agent.model.module.state_dict() if isinstance(agent.model, DDP) else agent.model.state_dict()
    payload = {"model": model_state, "step": agent.step}
    if hasattr(agent, "target_model") and agent.target_model is not None:
        tm_state = agent.target_model.module.state_dict() if isinstance(agent.target_model, DDP) else agent.target_model.state_dict()
        payload["target_model"] = tm_state
    torch.save(payload, path)
    return path


def _ddp_init_if_needed(device: torch.device):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # 优先使用 IPv4 回环，避免某些系统上 localhost 解析到 IPv6 导致 AF_INET6 问题
        ma = os.environ.get("MASTER_ADDR", "")
        if (not ma) or (ma.lower() == "localhost") or (ma == "::1"):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("MASTER_PORT", "29500")
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def basic_env_setup(cfg: DictConfig):
    pass

@hydra_main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = _prep_device(str(getattr(cfg, "device", "cuda")))

    # Enable backend accelerations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Suppress prints on non-main processes
    if not is_main_process():
        os.environ['WANDB_SILENT'] = 'true'
        _orig_print = print
        def _quiet(*args, **kwargs):
            if any(keyword in str(args) for keyword in ['rank', 'W0909', 'I0909', 'wandb']):
                return
            return _orig_print(*args, **kwargs)
        import builtins
        builtins.print = _quiet

    # Graceful shutdown flag and signal handlers
    stop_requested = {"val": False}
    def _signal_handler(signum, frame):
        stop_requested["val"] = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # init (optional) DDP
    ddp_cfg = getattr(cfg, "ddp", {}) or {}
    ddp_enabled = bool(ddp_cfg.get("enabled", False))
    ddp_active, rank, world_size, local_rank = _ddp_init_if_needed(device)
    if ddp_enabled and not ddp_active and torch.cuda.device_count() > 1:
        logging.warning("DDP enabled in config but process not launched with torchrun. Proceeding single-process.")

    # set up wandb (only on main process)
    wandb_logger = None
    save_root: str | None = None
    if is_main_process():
        wb_cfg = getattr(cfg, "wandb", {}) or {}
        wb_enabled = bool(wb_cfg.get("enabled", True)) if isinstance(wb_cfg, dict) else True
        project_name = wb_cfg.get("project", "bridgedata_torch") if isinstance(wb_cfg, dict) else "bridgedata_torch"
        exp_descriptor = str(getattr(cfg, "name", ""))

        wandb_config = WandBLogger.get_default_config()
        cfg_plain = OmegaConf.to_container(cfg, resolve=True)
        wandb_config.update({"project": project_name, "exp_descriptor": exp_descriptor})
        if wb_enabled:
            wandb_logger = WandBLogger(wandb_config=wandb_config, variant=cfg_plain, debug=bool(getattr(cfg, "debug", False)))
        else:
            wandb_logger = None

        unique_id = wandb_logger.config.unique_identifier if wandb_logger else None
        if not unique_id:
            from datetime import datetime as _dt
            unique_id = _dt.now().strftime("%Y%m%d_%H%M%S")

        save_root = os.path.join(str(cfg.save_dir), project_name, f"{exp_descriptor}_{unique_id}")
        _ensure_dir(save_root)
        snap = OmegaConf.to_container(cfg, resolve=True)
        with open(os.path.join(save_root, "config.json"), "w") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False)
        logging.info("Config saved to %s", save_root)
        if wandb_logger and getattr(wandb_logger, "run", None):
            logging.info("WandB run initialized: %s", wandb_logger.run.url)

    # Build numpy datasets (required)
    train_seed = int(cfg.seed) + (rank if ddp_active else 0)
    val_seed = int(cfg.seed)

    per_rank_bs = int(cfg.batch_size // (world_size if ddp_active else 1))
    per_rank_val_bs = int(getattr(cfg, "val_batch_size", cfg.batch_size) // (world_size if ddp_active else 1))
    per_rank_bs = max(per_rank_bs, 1)
    per_rank_val_bs = max(per_rank_val_bs, 1)

    bdcfg = cfg.bridgedata
    # Build torch-style datasets
    data_cfg = getattr(cfg, "data", {}) or {}
    data_kwargs = OmegaConf.to_container(data_cfg, resolve=True) if isinstance(data_cfg, DictConfig) else dict(data_cfg)
    # inject saliency.alpha
    if "saliency" in cfg and hasattr(cfg.saliency, "alpha"):
        data_kwargs.setdefault("saliency_alpha", float(cfg.saliency.alpha))

    train_ds = build_bridge_dataset(
        task_globs=bdcfg.include,
        data_root=str(cfg.data_path),
        split="train",
        seed=train_seed,
        bridgedata_cfg=bdcfg,
        dataset_kwargs=data_kwargs,
    )
    val_ds = build_bridge_dataset(
        task_globs=bdcfg.include,
        data_root=str(cfg.data_path),
        split="val",
        seed=val_seed,
        bridgedata_cfg=bdcfg,
        dataset_kwargs=data_kwargs,
    )

    # DataLoaders
    dl_cfg = getattr(cfg, "dataloader", {})
    dl_kwargs = OmegaConf.to_container(dl_cfg, resolve=True) if isinstance(dl_cfg, DictConfig) else (dl_cfg.to_dict() if hasattr(dl_cfg, "to_dict") else dict(dl_cfg))
    train_sampler = None
    val_sampler = None
    if ddp_enabled and ddp_active:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = make_dataloader(train_ds, batch_size=per_rank_bs, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=False, **dl_kwargs)
    val_loader = make_dataloader(val_ds, batch_size=per_rank_val_bs, shuffle=False, sampler=val_sampler, drop_last=False, **dl_kwargs)

    # Peek one batch to infer shapes
    warmup_batch = next(iter(train_loader))
    obs_img = warmup_batch["observations"]["image"]
    obs_shape = obs_img.shape
    action_dim = int(warmup_batch["actions"].shape[-1])
    goal_img = warmup_batch.get("goals", {}).get("image") if isinstance(warmup_batch.get("goals"), dict) else None
    goal_shape = goal_img.shape if goal_img is not None else None
    if is_main_process():
        logging.info("Obs batch shape: %s, action_dim: %s, goal shape: %s", obs_shape, action_dim, goal_shape)

    # Build encoder — 容错读取（若未从 algo 预设合成成功则回退到默认）
    enc_name = cfg.get("encoder", "resnetv1-34-bridge") if isinstance(cfg, DictConfig) else getattr(cfg, "encoder", "resnetv1-34-bridge")
    enc_kwargs_obj = cfg.get("encoder_kwargs", {}) if isinstance(cfg, DictConfig) else getattr(cfg, "encoder_kwargs", {})
    enc_kwargs = OmegaConf.to_container(enc_kwargs_obj, resolve=True) if isinstance(enc_kwargs_obj, DictConfig) else (dict(enc_kwargs_obj) if isinstance(enc_kwargs_obj, dict) else {})
    if "arch" not in enc_kwargs:
        enc_kwargs.setdefault("pooling_method", "avg")
        enc_kwargs.setdefault("add_spatial_coordinates", True)
        enc_kwargs.setdefault("act", "swish")
        enc_kwargs["arch"] = "resnet34"
    enc = build_encoder(enc_name, **enc_kwargs).to(device)

    # torch.compile on encoder if enabled (and not DDP-active)
    compile_enabled = getattr(cfg, "compile", {}).get("enabled", False)
    compile_kwargs = getattr(cfg, "compile", {}).get("kwargs", {})
    if compile_enabled and hasattr(torch, 'compile') and not (ddp_enabled and ddp_active):
        enc = torch.compile(enc, **compile_kwargs)
        if is_main_process():
            logging.info("Encoder compiled with torch.compile")
    elif compile_enabled and (ddp_enabled and ddp_active) and is_main_process():
        logging.info("DDP active: skip compiling encoder; will compile full model instead")

    # Warm-up forward to initialize lazy shapes (use channels_last for conv perf)
    _warm = warmup_batch["observations"]["image"].to(device)
    if _warm.dim() == 4:
        _warm = _warm.contiguous(memory_format=torch.channels_last)
    _ = enc(_warm)
    enc_feat = getattr(enc, "_feat_dim")
    # Model/policy configuration (flattened Hydra schema)
    model_cfg_obj = cfg.get("model", {}) if isinstance(cfg, DictConfig) else getattr(cfg, "model", {})
    model_cfg = OmegaConf.to_container(model_cfg_obj, resolve=True) if isinstance(model_cfg_obj, DictConfig) else (dict(model_cfg_obj) if isinstance(model_cfg_obj, dict) else {})

    prop_dim = 0
    if bool(model_cfg.get("use_proprio", False)) and ("proprio" in warmup_batch["observations"]):
        prop_dim = int(warmup_batch["observations"]["proprio"].shape[1])

    # 统一解析 saliency 配置（顶层优先）
    def _resolve_saliency(dc: DictConfig) -> dict:
        s = {}
        if hasattr(dc, "saliency") and dc.saliency is not None:
            s = OmegaConf.to_container(dc.saliency, resolve=True)  # type: ignore
        else:
            # 兼容旧路径：bc/gc_bc 在 policy_kwargs.saliency，ddpm 在 agent_kwargs.saliency
            try:
                pk = dc.agent_kwargs.get("policy_kwargs", {})
                if isinstance(pk, DictConfig):
                    pk = OmegaConf.to_container(pk, resolve=True)
                s = dict(pk.get("saliency", {}))
            except Exception:
                pass
            if not s:
                try:
                    s = dict(dc.agent_kwargs.get("saliency", {}))
                except Exception:
                    s = {}
        # 归一化键与类型
        if s:
            s = {
                "enabled": bool(str(s.get("enabled", False)).lower() in {"1","true","t","yes","y","on"}),
                "weight": float(s.get("weight", 0.0)),
                "beta": float(s.get("beta", 1.0)),
            }
        return s

    sal_cfg = _resolve_saliency(cfg)

    # Build BC agent/model
    mlp = MLP_BC(input_dim=enc_feat + prop_dim, hidden_dims=tuple(model_cfg.get("hidden_dims", [256,256,256])), dropout_rate=float(model_cfg.get("dropout_rate", 0.0)))
    policy_kwargs = {
        "tanh_squash_distribution": bool(model_cfg.get("tanh_squash_distribution", False)),
        "state_dependent_std": bool(model_cfg.get("state_dependent_std", False)),
        "fixed_std": list(model_cfg.get("fixed_std", [])) or None,
        "use_proprio": bool(model_cfg.get("use_proprio", False)),
    }
    agent_cfg = BCAgentConfig(
        learning_rate=float(getattr(cfg, "optimizer", {}).get("lr", 3e-4) if isinstance(getattr(cfg, "optimizer", {}), dict) else 3e-4),
        weight_decay=float(getattr(cfg, "optimizer", {}).get("weight_decay", 0.0) if isinstance(getattr(cfg, "optimizer", {}), dict) else 0.0),
        warmup_steps=int(getattr(cfg, "scheduler", {}).get("warmup_steps", 1000) if isinstance(getattr(cfg, "scheduler", {}), dict) else 1000),
        decay_steps=int(getattr(cfg, "scheduler", {}).get("decay_steps", getattr(cfg, "num_steps", 1000000)) if isinstance(getattr(cfg, "scheduler", {}), dict) else int(getattr(cfg, "num_steps", 1000000))),
        scheduler=OmegaConf.to_container(getattr(cfg, "scheduler", {}), resolve=True) if isinstance(getattr(cfg, "scheduler", {}), DictConfig) else (getattr(cfg, "scheduler", {}) if isinstance(getattr(cfg, "scheduler", {}), dict) else {}),
        saliency=sal_cfg or {},
    )
    agent = BCAgent(
        model=GaussianPolicy(
            enc, mlp, action_dim=action_dim, **policy_kwargs
        ),
        cfg=agent_cfg,
        action_dim=action_dim,
        device=device,
    )

    # DDP wrap for full model if enabled
    if ddp_enabled and ddp_active:
        agent.model = DDP(
            agent.model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(getattr(cfg, "ddp", {}).get("find_unused_parameters", False)),
        )

    # AMP
    amp_cfg = getattr(cfg, "amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False))
    amp_dtype_name = str(amp_cfg.get("dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    scaler = None
    if amp_enabled and device.type == "cuda":
        if amp_dtype is torch.float16:
            scaler = GradScaler(
                enabled=True,
                growth_factor=float(amp_cfg.get("growth_factor", 2.0)),
                backoff_factor=float(amp_cfg.get("backoff_factor", 0.5)),
                growth_interval=int(amp_cfg.get("growth_interval", 2000)),
            )
            if is_main_process():
                logging.info("AMP FP16 enabled with GradScaler")
        else:
            scaler = None
            if is_main_process():
                logging.info("AMP BF16 enabled (no GradScaler)")
    elif amp_enabled and device.type != "cuda" and is_main_process():
        logging.warning("AMP requested but CUDA not available; running in FP32")

    # Optionally resume
    if getattr(cfg, "resume_path", None):
        ckpt = torch.load(str(cfg.resume_path), map_location=device)
        model_to_load = agent.model.module if isinstance(agent.model, DDP) else agent.model
        model_to_load.load_state_dict(ckpt["model"])  # type: ignore

    train_iter = iter(train_loader)

    progress = tqdm(range(int(cfg.num_steps)), disable=not is_main_process(), desc="Train", dynamic_ncols=True)
    try:
        for step in progress:
            if stop_requested["val"]:
                break
            try:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
            except KeyboardInterrupt:
                stop_requested["val"] = True
                break

            if amp_enabled:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    info = agent.update(batch)
            else:
                info = agent.update(batch)

            if is_main_process() and (step + 1) % int(cfg.log_interval) == 0:
                show = {k: (float(v) if isinstance(v, (int, float)) else float(v)) for k, v in info.items() if k in ("ddpm_loss", "mse") and v is not None}
                progress.set_postfix(show, refresh=True)
                logging.info("[step %d] %s", step + 1, info)
                if wandb_logger:
                    wandb_logger.log({"training": info}, step=step + 1)

            if (step + 1) % int(cfg.eval_interval) == 0 and is_main_process():
                eval_batches_config = getattr(cfg, "eval_batches", None)
                if eval_batches_config is None:
                    logging.info("[eval %d] Skipped - eval_batches set to null", step + 1)
                else:
                    metrics = []
                    val_iter = iter(val_loader)
                    max_eval_batches = int(eval_batches_config) if int(eval_batches_config) > 0 else 50
                    seen = 0
                    val_pb = tqdm(total=int(max_eval_batches), desc="Eval", leave=False, disable=not is_main_process(), dynamic_ncols=True)
                    for vb in val_iter:
                        metrics.append(agent.get_debug_metrics(vb))
                        seen += 1
                        if seen >= max_eval_batches:
                            break
                        val_pb.update(1)
                    val_pb.close()
                    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}
                    logging.info("[eval %d] %s", step + 1, avg)
                    if wandb_logger:
                        wandb_logger.log({"validation": avg}, step=step + 1)

            if is_main_process() and (step + 1) % int(cfg.save_interval) == 0:
                ckpt_path = save_checkpoint(save_root or str(cfg.save_dir), agent, step + 1)
                logging.info("Saved checkpoint to %s", ckpt_path)
    except KeyboardInterrupt:
        stop_requested["val"] = True
    finally:
        if ddp_enabled and ddp_active and dist.is_initialized():
            dist.destroy_process_group()
        if is_main_process():
            if wandb_logger and getattr(wandb_logger, "run", None):
                wandb_logger.run.finish()  # type: ignore[attr-defined]
        if stop_requested["val"]:
            sys.exit(0)


if __name__ == "__main__":
    main()
