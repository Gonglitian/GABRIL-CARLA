#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import shlex
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    print("[ERROR] PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise


def to_flag_list(mapping: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in mapping.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            args.extend([flag, str(value)])
        else:
            args.extend([flag, str(value)])
    return args


def build_run_args(run: Dict[str, Any], g: Dict[str, Any]) -> List[str]:
    algo = run["algo"]
    task = run["task"]

    name = run.get("name") or g.get("name_format", "{algo}_{task}").format(
        algo=algo, task=task
    )

    base_args = [
        "--name",
        name,
        "--config",
        f"bridge_torch/experiments/configs/train_config.py:{algo}",
        "--bridgedata_config",
        f"bridge_torch/experiments/configs/data_config.py:{task}",
    ]

    cfg_overrides: Dict[str, Any] = {
        "config.data_path": g["data_root"],
        "config.save_dir": g["save_dir"],
        "config.batch_size": g["train_batch_size"],
        "config.val_batch_size": g.get("val_batch_size", g["train_batch_size"]),
        "config.num_steps": g["num_steps"],
        "config.eval_interval": g["eval_interval"],
        "config.save_interval": g["save_interval"],
        "config.log_interval": g["log_interval"],
        "config.agent_kwargs.warmup_steps": g["warmup_steps"],
    }

    if g.get("eval_batches") is not None:
        cfg_overrides["config.eval_batches"] = g["eval_batches"]

    run_overrides: Dict[str, Any] = run.get("config_overrides", {})
    if algo == "bc" and "config.agent_kwargs.decay_steps" not in run_overrides:
        cfg_overrides["config.agent_kwargs.decay_steps"] = g.get("decay_steps_bc", 1000000)
    if algo == "gc_bc" and "config.agent_kwargs.decay_steps" not in run_overrides:
        cfg_overrides["config.agent_kwargs.decay_steps"] = g.get("decay_steps_gc_bc", 1000000)
    if algo == "gc_ddpm_bc" and "config.agent_kwargs.actor_decay_steps" not in run_overrides:
        cfg_overrides["config.agent_kwargs.actor_decay_steps"] = g.get("actor_decay_steps_gc_ddpm_bc", 1000000)

    cfg_overrides.update(run_overrides)

    # Dataset knobs: allow YAML to set obs/act horizons and augmentation (including nested augment_kwargs)
    def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a or {})
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out

    ds_global = g.get("dataset", {})
    ds_run = run.get("dataset", {})
    ds_cfg: Dict[str, Any] = _merge(ds_global, ds_run)

    # Inject global saliency knobs: dataset alpha and agent saliency switches
    sal = g.get("saliency", {}) or {}
    if sal:
        # dataset-side alpha for causal aggregation of saliency
        if "alpha" in sal and sal["alpha"] is not None:
            ds_cfg["saliency_alpha"] = sal["alpha"]

    # Ensure horizons compatible across algorithms
    if algo in {"bc", "gc_bc"}:
        if "act_pred_horizon" in ds_cfg and int(ds_cfg.get("act_pred_horizon", 1)) != 1:
            # clamp to 1 for single-step actors
            ds_cfg["act_pred_horizon"] = 1

    # Flatten dataset config to --config.dataset_kwargs.* flags
    def _as_cli_value(v: Any) -> str:
        # ml_collections 不支持直接传 list 覆盖 tuple，需使用字符串形式的元组
        if isinstance(v, list):
            return "(" + ", ".join(str(x) for x in v) + ")"
        return str(v)

    def _flatten(prefix: str, d: Dict[str, Any], out: Dict[str, Any]):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(key, v, out)
            else:
                out[key] = _as_cli_value(v)

    if ds_cfg:
        ds_overrides: Dict[str, Any] = {}
        _flatten("config.dataset_kwargs", ds_cfg, ds_overrides)
        cfg_overrides.update(ds_overrides)

    # Agent-side saliency (enabled/weight/beta) – map to algo-specific paths
    if sal:
        if algo in {"bc", "gc_bc"}:
            if "enabled" in sal:
                cfg_overrides["config.agent_kwargs.policy_kwargs.saliency.enabled"] = str(bool(sal["enabled"]))
            if "weight" in sal and sal["weight"] is not None:
                cfg_overrides["config.agent_kwargs.policy_kwargs.saliency.weight"] = str(sal["weight"])
            if "beta" in sal and sal["beta"] is not None:
                cfg_overrides["config.agent_kwargs.policy_kwargs.saliency.beta"] = str(sal["beta"])
        elif algo == "gc_ddpm_bc":
            if "enabled" in sal:
                cfg_overrides["config.agent_kwargs.saliency.enabled"] = str(bool(sal["enabled"]))
            if "weight" in sal and sal["weight"] is not None:
                cfg_overrides["config.agent_kwargs.saliency.weight"] = str(sal["weight"])
            if "beta" in sal and sal["beta"] is not None:
                cfg_overrides["config.agent_kwargs.saliency.beta"] = str(sal["beta"])

    # Pass through global knobs: amp / compile / dataloader / profiler / scheduler / wandb
    for knob_key in ("amp", "compile", "dataloader", "profiler", "scheduler", "wandb"):
        knob = g.get(knob_key)
        if knob:
            flat: Dict[str, Any] = {}
            _flatten(f"config.{knob_key}", knob, flat)
            cfg_overrides.update(flat)

    args = base_args + to_flag_list(cfg_overrides)

    bd_overrides = run.get("bridgedata_overrides", {})
    if bd_overrides:
        args += to_flag_list({f"bridgedata_config.{k}": v for k, v in bd_overrides.items()})

    extra_args = g.get("extra_args", []) + run.get("extra_args", [])
    for token in extra_args:
        args.extend(shlex.split(str(token)))

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch multi-run trainer (YAML-driven)")
    parser.add_argument(
        "--config",
        type=str,
        default="bridge_torch/experiments/configs/multi_train.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    g = cfg.get("global", {})
    if not g:
        print("[ERROR] 'global' section is required in YAML", file=sys.stderr)
        sys.exit(2)

    runs = cfg.get("runs")
    if not runs:
        matrix = cfg.get("matrix", {})
        algos = matrix.get("algos", [])
        tasks = matrix.get("tasks", [])
        runs = [{"algo": a, "task": t} for a in algos for t in tasks]

    if not runs:
        print("[WARN] No runs defined in YAML; exiting.", file=sys.stderr)
        return

    base_env = os.environ.copy()
    if g.get("cuda_visible_devices") is not None:
        base_env["CUDA_VISIBLE_DEVICES"] = str(g["cuda_visible_devices"])

    # Optional DDP launcher settings
    ddp_cfg = g.get("ddp", {}) or {}
    ddp_enabled = bool(ddp_cfg.get("enabled", False))
    nproc = int(ddp_cfg.get("nproc_per_node", 0))

    for i, run in enumerate(runs, 1):
        run_args = build_run_args(run, g)
        if ddp_enabled:
            # Launch with torch.distributed.run (torchrun)
            launcher = [sys.executable, "-m", "torch.distributed.run"]
            if nproc and nproc > 0:
                launcher += ["--nproc_per_node", str(nproc)]
            # Train as a module to ensure package imports work
            argv = launcher + ["-m", "bridge_torch.experiments.train"] + run_args
        else:
            argv = [sys.executable, "-m", "bridge_torch.experiments.train"] + run_args

        print(f"[RUN {i:02d}/{len(runs)}] {' '.join(shlex.quote(a) for a in argv)}")
        if args.dry_run:
            continue
        try:
            proc = subprocess.Popen(argv, env=base_env)
            rc = proc.wait()
        except KeyboardInterrupt:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
            finally:
                raise
        if rc != 0:
            print(f"[ERROR] Run {i} failed with exit code {rc}", file=sys.stderr)
            sys.exit(rc)
    print("[ALL DONE] Torch experiments finished")


if __name__ == "__main__":
    main()
