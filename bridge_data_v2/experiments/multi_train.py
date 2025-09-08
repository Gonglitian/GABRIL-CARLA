#!/usr/bin/env python3
"""
Multi-run launcher driven by a YAML configuration.

Reads a YAML file describing global defaults and a set of runs, then
invokes `experiments/train.py` for each run with the appropriate flags.

This avoids maintaining multiple bash scripts and makes multi-experiment
configurations reproducible and explicit.
"""

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
    """Converts a mapping of flag->value into a flat argv segment.

    Example: {"config.batch_size": 256} -> ["--config.batch_size", "256"]
    """
    args: List[str] = []
    for key, value in mapping.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            # absl flags expect explicit true/false values
            args.extend([flag, str(value)])
        else:
            args.extend([flag, str(value)])
    return args


def build_run_args(
    run: Dict[str, Any],
    g: Dict[str, Any],
) -> List[str]:
    algo = run["algo"]
    task = run["task"]

    # Name
    name = run.get("name") or g.get("name_format", "{algo}_{task}").format(
        algo=algo, task=task
    )

    base_args = [
        "--name",
        name,
        "--config",
        f"experiments/configs/train_config.py:{algo}",
        "--bridgedata_config",
        f"experiments/configs/data_config.py:{task}",
    ]

    # Required training config overrides (from global defaults)
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

    # Optional eval batches cap
    if g.get("eval_batches") is not None:
        cfg_overrides["config.eval_batches"] = g["eval_batches"]

    # Per‑algo default decay flags unless explicitly overridden in run
    run_overrides: Dict[str, Any] = run.get("config_overrides", {})
    if algo == "bc" and "config.agent_kwargs.decay_steps" not in run_overrides:
        cfg_overrides["config.agent_kwargs.decay_steps"] = g.get(
            "decay_steps_bc", 1000000
        )
    if algo == "gc_bc" and "config.agent_kwargs.decay_steps" not in run_overrides:
        cfg_overrides["config.agent_kwargs.decay_steps"] = g.get(
            "decay_steps_gc_bc", 1000000
        )
    if (
        algo == "gc_ddpm_bc"
        and "config.agent_kwargs.actor_decay_steps" not in run_overrides
    ):
        cfg_overrides["config.agent_kwargs.actor_decay_steps"] = g.get(
            "actor_decay_steps_gc_ddpm_bc", 1000000
        )

    # Merge per-run overrides (take precedence)
    cfg_overrides.update(run_overrides)

    args = base_args + to_flag_list(cfg_overrides)

    # Bridgedata overrides if any (rarely used)
    bd_overrides = run.get("bridgedata_overrides", {})
    if bd_overrides:
        args += to_flag_list({f"bridgedata_config.{k}": v for k, v in bd_overrides.items()})

    # Extra raw args (list of strings, each split with shlex)
    extra_args = g.get("extra_args", []) + run.get("extra_args", [])
    for token in extra_args:
        args.extend(shlex.split(str(token)))

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-run trainer (YAML-driven)")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/multi_train.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    g = cfg.get("global", {})
    if not g:
        print("[ERROR] 'global' section is required in YAML", file=sys.stderr)
        sys.exit(2)

    # Expand run matrix if explicit runs not provided
    runs = cfg.get("runs")
    if not runs:
        matrix = cfg.get("matrix", {})
        algos = matrix.get("algos", [])
        tasks = matrix.get("tasks", [])
        runs = [{"algo": a, "task": t} for a in algos for t in tasks]

    if not runs:
        print("[WARN] No runs defined in YAML; exiting.", file=sys.stderr)
        return

    # Base environment for spawned processes
    base_env = os.environ.copy()
    if g.get("cuda_visible_devices"):
        base_env["CUDA_VISIBLE_DEVICES"] = str(g["cuda_visible_devices"])

    for i, run in enumerate(runs, 1):
        argv = [sys.executable, "experiments/train.py"] + build_run_args(run, g)
        print(f"[RUN {i:02d}/{len(runs)}] {' '.join(shlex.quote(a) for a in argv)}")
        if args.dry_run:
            continue

        proc = subprocess.Popen(argv, env=base_env)
        rc = proc.wait()
        if rc != 0:
            print(f"[ERROR] Run {i} failed with exit code {rc}", file=sys.stderr)
            sys.exit(rc)

    print("[ALL DONE] All experiments finished")


if __name__ == "__main__":
    main()

