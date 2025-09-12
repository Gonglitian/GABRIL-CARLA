import os
import json
import warnings
import signal
import sys
from functools import partial

# Early warnings setup
warnings.filterwarnings(
    "ignore",
    message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary",
    category=UserWarning,
)
# 屏蔽 torch._sympy 的 pow_by_natural 相关告警（Dynamo 符号推理产生，非致命）
warnings.filterwarnings(
    "ignore",
    message=r".*pow_by_natural.*",
    category=UserWarning,
)

import numpy as np
import torch
from absl import app, flags, logging
import logging as std_logging  # 标准 logging，用于控制 PyTorch 内部 logger 级别
from ml_collections import config_flags
from tqdm import tqdm

# Add AMP support
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler

from bridge_torch.models.encoders import build_encoder
from bridge_torch.agents.bc import BCAgent, BCAgentConfig, MLP as MLP_BC
from bridge_torch.agents.gc_bc import GCBCAgent, GCBCConfig, MLP as MLP_GC
from bridge_torch.agents.gc_ddpm_bc import GCDDPMBCAgent, GCDDPMConfig, build_ddpm_policy
from bridge_torch.data.bridge_numpy import build_np_bridge_dataset, iter_torch_batches_np
from bridge_torch.common.wandb import WandBLogger
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)

def _prep_device(device_flag: str) -> torch.device:
    if device_flag == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def save_checkpoint(save_dir: str, agent, step: int, cfg=None):
    _ensure_dir(save_dir)
    path = os.path.join(save_dir, f"ckpt_{step}.pt")
    # unwrap DDP if present
    model_state = agent.model.module.state_dict() if isinstance(agent.model, DDP) else agent.model.state_dict()
    payload = {"model": model_state, "step": agent.step}
    if hasattr(agent, "target_model") and agent.target_model is not None:
        tm_state = agent.target_model.module.state_dict() if isinstance(agent.target_model, DDP) else agent.target_model.state_dict()
        payload["target_model"] = tm_state
    torch.save(payload, path)
    return path


def _ddp_init_if_needed(device: torch.device):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def image_batch_shape_of(ds):
    # Peek one batch to infer action dims and image size for encoder warmup
    it = iter_torch_batches_np(ds)
    first = next(it)
    obs_img = first["observations"]["image"]
    act = first["actions"]
    goal_img = first.get("goals", {}).get("image") if isinstance(first.get("goals"), dict) else None
    return obs_img.shape, act.shape[-1], (goal_img.shape if goal_img is not None else None), first


def main(_):
    cfg = FLAGS.config
    bdcfg = FLAGS.bridgedata_config

    device = _prep_device(getattr(cfg, "device", "cuda"))

    # Enable backend accelerations
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # For PyTorch >= 2.0
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Suppress non-main process outputs and warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pow_by_natural.*")
    # 进一步降低 PyTorch 内部 logger 的噪声
    try:
        std_logging.getLogger("torch.utils._sympy").setLevel(std_logging.ERROR)
        std_logging.getLogger("torch._dynamo").setLevel(std_logging.ERROR)
    except Exception:
        pass

    if not is_main_process():
        # Disable wandb logging for non-main processes
        os.environ['WANDB_SILENT'] = 'true'
        # Suppress other unnecessary prints
        original_print = print
        def suppressed_print(*args, **kwargs):
            if any(keyword in str(args) for keyword in ['rank', 'W0909', 'I0909', 'wandb']):
                return  # Suppress rank-specific prints
            return original_print(*args, **kwargs)
        import builtins
        builtins.print = suppressed_print

    # Graceful shutdown flag and signal handlers
    stop_requested = {"val": False}

    def _signal_handler(signum, frame):
        stop_requested["val"] = True
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

    # init (optional) DDP
    ddp_enabled = bool(getattr(cfg, "ddp", {}).get("enabled", False))
    ddp_active, rank, world_size, local_rank = _ddp_init_if_needed(device)
    if ddp_enabled and not ddp_active and torch.cuda.device_count() > 1:
        logging.warning("DDP enabled in config but process not launched with torchrun. Proceeding single-process.")

    # set up wandb (only on main process), allow disabling via config.wandb.enabled
    wandb_logger = None
    save_root: str | None = None
    if is_main_process():
        wb_cfg = getattr(cfg, "wandb", {}) or {}
        try:
            wb_enabled = bool(wb_cfg.get("enabled", True)) if isinstance(wb_cfg, dict) else True
        except Exception:
            wb_enabled = True

        project_name = wb_cfg.get("project", "bridgedata_torch") if isinstance(wb_cfg, dict) else "bridgedata_torch"

        unique_id = None
        exp_descriptor = FLAGS.name

        if wb_enabled:
            wandb_config = WandBLogger.get_default_config()
            wandb_config.update({"project": project_name, "exp_descriptor": exp_descriptor})
            try:
                wandb_logger = WandBLogger(wandb_config=wandb_config, variant=cfg.to_dict(), debug=FLAGS.debug)
                unique_id = wandb_logger.config.unique_identifier
            except Exception:
                wandb_logger = None

        # Prepare dirs and config snapshot following original layout
        if unique_id is None:
            from datetime import datetime as _dt
            unique_id = _dt.now().strftime("%Y%m%d_%H%M%S")

        save_root = os.path.join(
            cfg.save_dir,
            project_name,
            f"{exp_descriptor}_{unique_id}",
        )
        _ensure_dir(save_root)
        try:
            snap = cfg.to_dict()
            snap["bridgedata_config"] = bdcfg.to_dict()
            with open(os.path.join(save_root, "config.json"), "w") as f:
                json.dump(snap, f)
            logging.info("Config saved to %s", save_root)
            if wandb_logger and getattr(wandb_logger, "run", None):
                logging.info("WandB run initialized: %s", wandb_logger.run.url)
            else:
                if wb_enabled:
                    logging.warning("WandB enabled but run not initialized (offline or import issue)")
                else:
                    logging.info("WandB disabled via config; logging to console only")
        except Exception as e:
            logging.warning("Failed to write config.json: %s", e)

    # Build numpy datasets (required)
    # Use distinct seeds per rank to decorrelate sampling under DDP
    train_seed = int(cfg.seed) + (rank if ddp_active else 0)
    val_seed = int(cfg.seed)  # evaluation only on main process; keep fixed

    # Per-rank batch size under DDP
    per_rank_bs = int(cfg.batch_size // (world_size if ddp_active else 1))
    per_rank_val_bs = int(getattr(cfg, "val_batch_size", cfg.batch_size) // (world_size if ddp_active else 1))
    if per_rank_bs <= 0:
        per_rank_bs = 1
    if per_rank_val_bs <= 0:
        per_rank_val_bs = 1

    train_ds = build_np_bridge_dataset(
        task_globs=bdcfg.include,
        data_root=cfg.data_path,
        split="train",
        seed=train_seed,
        batch_size=per_rank_bs,
        bridgedata_cfg=bdcfg,
        dataset_kwargs=cfg.dataset_kwargs,
        ddp_shard=None,
    )
    val_ds = build_np_bridge_dataset(
        task_globs=bdcfg.include,
        data_root=cfg.data_path,
        split="val",
        seed=val_seed,
        batch_size=per_rank_val_bs,
        bridgedata_cfg=bdcfg,
        dataset_kwargs=cfg.dataset_kwargs,
        ddp_shard=None,
    )
    # DataLoader kwargs from config if provided
    dl_kwargs = {}
    try:
        dl_cfg = getattr(cfg, "dataloader", {})
        if hasattr(dl_cfg, "to_dict"):
            dl_kwargs = dl_cfg.to_dict()
        elif isinstance(dl_cfg, dict):
            dl_kwargs = dict(dl_cfg)
    except Exception:
        dl_kwargs = {}
    iter_batches = partial(iter_torch_batches_np, loader_kwargs=dl_kwargs)

    # Peek one batch
    warmup_batch = next(iter_batches(train_ds))
    obs_img = warmup_batch["observations"]["image"]
    obs_shape = obs_img.shape
    action_dim = int(warmup_batch["actions"].shape[-1])
    goal_img = warmup_batch.get("goals", {}).get("image") if isinstance(warmup_batch.get("goals"), dict) else None
    goal_shape = goal_img.shape if goal_img is not None else None
    if is_main_process():
        logging.info("Obs batch shape: %s, action_dim: %s, goal shape: %s", obs_shape, action_dim, goal_shape)

    # Build encoder(s)
    enc = build_encoder(cfg.encoder, **cfg.encoder_kwargs).to(device)

    # Apply torch.compile if enabled (before DDP wrapping)
    compile_enabled = getattr(cfg, "compile", {}).get("enabled", False)
    compile_kwargs = getattr(cfg, "compile", {}).get("kwargs", {})
    # 在 DDP 激活时避免对子模块 encoder 进行独立编译，防止子模块共享导致的 Dynamo 冲突
    if compile_enabled and hasattr(torch, 'compile') and not (ddp_enabled and ddp_active):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            enc = torch.compile(enc, **compile_kwargs)
        if is_main_process():
            logging.info("Encoder compiled with torch.compile")
    elif compile_enabled and (ddp_enabled and ddp_active) and is_main_process():
        logging.info("DDP active: skip compiling encoder; will compile full model instead")
    elif compile_enabled and not hasattr(torch, 'compile'):
        if is_main_process():
            logging.warning("torch.compile requested but not available (requires PyTorch >= 2.0)")

    # Warm-up forward to initialize lazy shapes
    _ = enc(warmup_batch["observations"]["image"].to(device))
    enc_feat = getattr(enc, "_feat_dim")
    prop_dim = 0
    if cfg.agent_kwargs.get("use_proprio", False) and ("proprio" in warmup_batch["observations"]):
        prop_dim = int(warmup_batch["observations"]["proprio"].shape[1])

    # Build agent/model
    if cfg.agent == "bc":
        mlp = MLP_BC(input_dim=enc_feat + prop_dim)
        policy = torch.nn.Sequential()  # placeholder not used directly
        model = None  # set below via BCAgent
        # Filter out saliency from policy kwargs (handled by agent config)
        _policy_kwargs_bc = dict(cfg.agent_kwargs.get("policy_kwargs", {}))
        _policy_kwargs_bc.pop("saliency", None)
        # Create BCAgentConfig with saliency configuration if present
        bc_config = BCAgentConfig(
            network_kwargs=cfg.agent_kwargs.get("network_kwargs", {}),
            policy_kwargs=_policy_kwargs_bc,  # Use the filtered policy_kwargs without saliency
            use_proprio=cfg.agent_kwargs.get("use_proprio", False),
            learning_rate=cfg.agent_kwargs.get("learning_rate", 3e-4),
            weight_decay=cfg.agent_kwargs.get("weight_decay", 0.0),
            warmup_steps=cfg.agent_kwargs.get("warmup_steps", 1000),
            decay_steps=cfg.num_steps,
        )
        # Add saliency config to BCAgentConfig if present in original policy_kwargs
        original_policy_kwargs = cfg.agent_kwargs.get("policy_kwargs", {})
        if "saliency" in original_policy_kwargs:
            bc_config.saliency = original_policy_kwargs["saliency"]
        
        agent = BCAgent(
            model=
            __import__("bridge_torch.agents.bc", fromlist=["GaussianPolicy"])  # late import to reuse encoder
            .GaussianPolicy(enc, mlp, action_dim=action_dim, use_proprio=cfg.agent_kwargs.get("use_proprio", False), **_policy_kwargs_bc),
            cfg=bc_config,
            action_dim=action_dim,
            device=device,
        )

        # Apply torch.compile to agent model if enabled
        if compile_enabled and hasattr(torch, 'compile'):
            if not ddp_enabled:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    agent.model = torch.compile(agent.model, **compile_kwargs)
                if is_main_process():
                    logging.info("BC Agent model compiled with torch.compile")
    elif cfg.agent == "gc_bc":
        # 当 shared_goal_encoder=True 时，传入 None，在策略内部复用 obs_encoder，避免 torch.compile 对相同模块的双注册导致 Dynamo 冲突
        enc_goal = None if cfg.agent_kwargs.get("shared_goal_encoder", True) else build_encoder(cfg.encoder, **cfg.encoder_kwargs).to(device)
        # warm-up goal
        if goal_shape is not None:
            _ = (enc if enc_goal is None else enc_goal)(warmup_batch["goals"]["image"].to(device))
        inp_dim = getattr(enc, "_feat_dim") + (getattr(enc, "_feat_dim") if enc_goal is None else getattr(enc_goal, "_feat_dim")) + prop_dim
        mlp = MLP_GC(input_dim=inp_dim, **cfg.agent_kwargs.get("network_kwargs", {}))
        # Filter out saliency from policy kwargs (handled by agent config)
        _policy_kwargs_gc = dict(cfg.agent_kwargs.get("policy_kwargs", {}))
        _policy_kwargs_gc.pop("saliency", None)
        model = __import__("bridge_torch.agents.gc_bc", fromlist=["GCBCPolicy"]).GCBCPolicy(
            obs_encoder=enc,
            goal_encoder=enc_goal,
            mlp=mlp,
            action_dim=action_dim,
            early_goal_concat=cfg.agent_kwargs.get("early_goal_concat", True),
            use_proprio=cfg.agent_kwargs.get("use_proprio", False),
            **_policy_kwargs_gc,
        )
        agent = GCBCAgent(
            model=model,
            cfg=GCBCConfig(
                network_kwargs=cfg.agent_kwargs.get("network_kwargs", {}),
                policy_kwargs=cfg.agent_kwargs.get("policy_kwargs", {}),
                early_goal_concat=cfg.agent_kwargs.get("early_goal_concat", True),
                shared_goal_encoder=cfg.agent_kwargs.get("shared_goal_encoder", True),
                use_proprio=cfg.agent_kwargs.get("use_proprio", False),
                learning_rate=cfg.agent_kwargs.get("learning_rate", 3e-4),
                weight_decay=cfg.agent_kwargs.get("weight_decay", 0.0),
                warmup_steps=cfg.agent_kwargs.get("warmup_steps", 1000),
                decay_steps=cfg.num_steps,
            ),
            action_dim=action_dim,
            device=device,
        )

        # Apply torch.compile to agent model if enabled
        if compile_enabled and hasattr(torch, 'compile'):
            if not ddp_enabled:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    agent.model = torch.compile(agent.model, **compile_kwargs)
                if is_main_process():
                    logging.info("GC-BC Agent model compiled with torch.compile")
    elif cfg.agent == "gc_ddpm_bc":
        # Expect action chunks (B, T, A); if absent, treat T=1
        act_shape = warmup_batch["actions"].shape
        action_seq_shape = act_shape[1:] if len(act_shape) == 3 else (1, act_shape[-1])
        # 共享目标编码器时，传入 None，在策略内部使用 obs_encoder 复用，避免模块双注册
        enc_goal = None if cfg.agent_kwargs.get("shared_goal_encoder", True) else build_encoder(cfg.encoder, **cfg.encoder_kwargs).to(device)
        if goal_shape is not None:
            _ = (enc if enc_goal is None else enc_goal)(warmup_batch["goals"]["image"].to(device))
        # Build DDPM policy
        policy = build_ddpm_policy(
            obs_encoder=enc,
            goal_encoder=enc_goal,
            early_goal_concat=cfg.agent_kwargs.get("early_goal_concat", False),
            time_dim=cfg.agent_kwargs.get("score_network_kwargs", {}).get("time_dim", 32),
            num_blocks=cfg.agent_kwargs.get("score_network_kwargs", {}).get("num_blocks", 3),
            dropout_rate=cfg.agent_kwargs.get("score_network_kwargs", {}).get("dropout_rate", 0.1),
            hidden_dim=cfg.agent_kwargs.get("score_network_kwargs", {}).get("hidden_dim", 256),
            use_layer_norm=cfg.agent_kwargs.get("score_network_kwargs", {}).get("use_layer_norm", True),
            action_seq_shape=tuple(action_seq_shape),
            device=device,
            use_proprio=cfg.agent_kwargs.get("use_proprio", False),
            proprio_feature_dim=prop_dim,
        )
        agent = GCDDPMBCAgent(
            policy=policy,
            cfg=GCDDPMConfig(
                learning_rate=cfg.agent_kwargs.get("learning_rate", 3e-4),
                warmup_steps=cfg.agent_kwargs.get("warmup_steps", 2000),
                actor_decay_steps=cfg.num_steps,
                beta_schedule=cfg.agent_kwargs.get("beta_schedule", "cosine"),
                diffusion_steps=cfg.agent_kwargs.get("diffusion_steps", 20),
                action_samples=cfg.agent_kwargs.get("action_samples", 1),
                repeat_last_step=cfg.agent_kwargs.get("repeat_last_step", 0),
                target_update_rate=cfg.agent_kwargs.get("target_update_rate", 0.002),
                action_min=-2.0,
                action_max=2.0,
            ),
            device=device,
        )

        # Apply torch.compile to agent model if enabled（在 DDP 包装前执行）
        if compile_enabled and hasattr(torch, 'compile'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                agent.model = torch.compile(agent.model, **compile_kwargs)
            if is_main_process():
                logging.info("GC-DDPM Agent model compiled with torch.compile")
    else:
        raise ValueError(f"Unsupported torch agent: {cfg.agent}")

    # Wrap with DDP if requested and available
    if ddp_enabled and ddp_active:
        # Note: DDP requires model on correct device and gradients enabled
        agent.model = DDP(agent.model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=getattr(cfg, "ddp", {}).get("find_unused_parameters", False))

    # Initialize AMP components and dtype
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
            if ddp_enabled and is_main_process():
                logging.warning("AMP FP16 enabled with DDP - ensure model compatibility")
            if is_main_process():
                logging.info("AMP FP16 enabled with GradScaler")
        else:
            scaler = None
            if is_main_process():
                logging.info("AMP BF16 enabled (no GradScaler)")
    elif amp_enabled and device.type != "cuda" and is_main_process():
        logging.warning("AMP requested but CUDA not available, disabling AMP")
        amp_enabled = False

    # Optionally resume
    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path, map_location=device)
        model_to_load = agent.model.module if isinstance(agent.model, DDP) else agent.model
        model_to_load.load_state_dict(ckpt["model"])  # type: ignore

    train_iter = iter_batches(train_ds)


    progress = tqdm(range(int(cfg.num_steps)), disable=not is_main_process(), desc="Train", dynamic_ncols=True)
    try:
        for step in progress:
            if stop_requested["val"]:
                break
            try:
                batch = next(train_iter)
            except KeyboardInterrupt:
                stop_requested["val"] = True
                break

            # Use AMP if enabled
            if amp_enabled:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    info = agent.update(batch)
            else:
                info = agent.update(batch)


            # Update progress bar with loss info
            if is_main_process() and (step + 1) % cfg.log_interval == 0:
                # brief postfix for fast glance
                try:
                    show = {k: (float(v) if isinstance(v, (int, float)) else float(v)) for k, v in info.items() if k in ("ddpm_loss", "mse") and v is not None}
                except Exception:
                    show = info
                progress.set_postfix(show, refresh=True)
                logging.info("[step %d] %s", step + 1, info)
                if wandb_logger:
                    try:
                        wandb_logger.log({"training": info}, step=step + 1)
                    except Exception as e:
                        logging.warning("Failed to log to WandB: %s", e)

            if (step + 1) % cfg.eval_interval == 0 and is_main_process():
                # Check if eval_batches is explicitly set to null to skip evaluation
                eval_batches_config = getattr(cfg, "eval_batches", None)
                if eval_batches_config is None:
                    logging.info("[eval %d] Skipped - eval_batches set to null", step + 1)
                else:
                    # evaluate
                    metrics = []
                    val_iter = iter_batches(val_ds)
                    # Cap evaluation to a finite number of batches to avoid infinite iterator
                    max_eval_batches = eval_batches_config
                    try:
                        if max_eval_batches is None or int(max_eval_batches) <= 0:
                            max_eval_batches = 50
                        else:
                            max_eval_batches = int(max_eval_batches)
                    except Exception:
                        max_eval_batches = 50
                    seen = 0
                    # show a short progress bar for eval if batch cap is provided
                    val_pb = tqdm(total=int(max_eval_batches), desc="Eval", leave=False, disable=not is_main_process(), dynamic_ncols=True)
                    for vb in val_iter:
                        metrics.append(agent.get_debug_metrics(vb))
                        seen += 1
                        if seen >= max_eval_batches:
                            break
                        val_pb.update(1)
                    val_pb.close()
                    if metrics:
                        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
                    logging.info("[eval %d] %s", step + 1, avg)
                    if wandb_logger:
                        try:
                            wandb_logger.log({"validation": avg}, step=step + 1)
                        except Exception as e:
                            logging.warning("Failed to log validation to WandB: %s", e)
                    else:
                        logging.warning("No validation batches available; skipping eval.")

            # Save checkpoint independently of eval, strictly by save_interval
            if is_main_process() and (step + 1) % cfg.save_interval == 0:
                ckpt_path = save_checkpoint(save_root or cfg.save_dir, agent, step + 1, cfg)
                logging.info("Saved checkpoint to %s", ckpt_path)
    except KeyboardInterrupt:
        stop_requested["val"] = True
    finally:
        # cleanly shutdown DDP
        try:
            if ddp_enabled and ddp_active and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        # ensure wandb run is finished on main process
        if is_main_process():
            try:
                if wandb_logger and getattr(wandb_logger, "run", None):
                    wandb_logger.run.finish()  # type: ignore[attr-defined]
            except Exception:
                pass
        # exit code 0 for graceful stop via SIGINT
        if stop_requested["val"]:
            sys.exit(0)


if __name__ == "__main__":
    app.run(main)
