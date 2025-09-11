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

    # Determine base run name
    sal = g.get("saliency", {}) or {}
    name = run.get("name") or g.get("name_format", "{algo}_{task}").format(
        algo=algo, task=task
    )

    # Build informative suffix from matrix/run config
    suffix_parts: List[str] = []

    # Helper: parse bool flag from extra_args (supports tokens like "--config.agent_kwargs.use_proprio True")
    def _parse_bool_from_extra(extra_list: List[str] | None, key: str) -> bool | None:
        if not extra_list:
            return None
        val: bool | None = None
        for s in extra_list:
            if key in s:
                if " True" in s or s.endswith("True"):
                    val = True
                elif " False" in s or s.endswith("False"):
                    val = False
        return val

    # use_proprio: prefer run.config_overrides -> run.extra_args -> global.extra_args
    use_proprio: bool | None = None
    co = run.get("config_overrides", {}) or {}
    if "config.agent_kwargs.use_proprio" in co:
        try:
            use_proprio = bool(co["config.agent_kwargs.use_proprio"]) if isinstance(co["config.agent_kwargs.use_proprio"], bool) else str(co["config.agent_kwargs.use_proprio"]).lower() == "true"
        except Exception:
            use_proprio = None
    if use_proprio is None:
        use_proprio = _parse_bool_from_extra(run.get("extra_args"), "--config.agent_kwargs.use_proprio")
    if use_proprio is None:
        use_proprio = _parse_bool_from_extra(g.get("extra_args", []), "--config.agent_kwargs.use_proprio")
    if use_proprio:
        suffix_parts.append("proprio")

    # saliency weight (run override or global)
    def _sanitize_float_for_name(x: float | int | str) -> str:
        s = str(x)
        s = s.replace(".", "p")
        return s

    eff_weight = None
    if "saliency_weight" in run and run["saliency_weight"] is not None:
        eff_weight = run["saliency_weight"]
    elif isinstance(run.get("saliency"), dict) and run["saliency"].get("weight") is not None:
        eff_weight = run["saliency"]["weight"]
    elif sal.get("weight") is not None:
        eff_weight = sal["weight"]
    if eff_weight is not None:
        try:
            # Attach only when > 0 or saliency enabled
            if float(eff_weight) > 0 or bool(sal.get("enabled", False)):
                suffix_parts.append(f"sw{_sanitize_float_for_name(eff_weight)}")
        except Exception:
            suffix_parts.append(f"sw{_sanitize_float_for_name(eff_weight)}")

    # obs_horizon from per-run dataset override or global dataset
    ds_run = run.get("dataset", {}) or {}
    ds_global = g.get("dataset", {}) or {}
    eff_obs = ds_run.get("obs_horizon", ds_global.get("obs_horizon"))
    if eff_obs is not None:
        try:
            suffix_parts.append(f"obs{int(eff_obs)}")
        except Exception:
            pass

    # backbone/encoder arch
    arch: str | None = None
    # Priority: run.backbone -> run.encoder_kwargs.arch -> run.encoder alias -> global.encoder_kwargs.arch -> global.encoder alias
    if run.get("backbone"):
        arch = str(run["backbone"]).lower()
    elif isinstance(run.get("encoder_kwargs"), dict) and run["encoder_kwargs"].get("arch"):
        arch = str(run["encoder_kwargs"]["arch"]).lower()
    elif run.get("encoder"):
        enc = str(run["encoder"]).lower()
        if "-18-" in enc:
            arch = "resnet18"
        elif "-34-" in enc:
            arch = "resnet34"
        elif "-50-" in enc:
            arch = "resnet50"
    if arch is None:
        ekg = g.get("encoder_kwargs", {}) or {}
        if ekg.get("arch"):
            arch = str(ekg["arch"]).lower()
        else:
            eg = str(g.get("encoder", "")).lower()
            if "-18-" in eg:
                arch = "resnet18"
            elif "-34-" in eg:
                arch = "resnet34"
            elif "-50-" in eg:
                arch = "resnet50"
    if arch:
        if arch.endswith("18"):
            suffix_parts.append("res18")
        elif arch.endswith("34"):
            suffix_parts.append("res34")
        elif arch.endswith("50"):
            suffix_parts.append("res50")

    if suffix_parts:
        name = f"{name}_" + "_".join(suffix_parts)

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

    # Per-run saliency overrides: allow a single scalar weight sweep or full dict
    if "saliency_weight" in run and run["saliency_weight"] is not None:
        if algo in {"bc", "gc_bc"}:
            cfg_overrides["config.agent_kwargs.policy_kwargs.saliency.weight"] = str(run["saliency_weight"])
        elif algo == "gc_ddpm_bc":
            cfg_overrides["config.agent_kwargs.saliency.weight"] = str(run["saliency_weight"])
    if "saliency" in run and isinstance(run["saliency"], dict):
        rsal = run["saliency"]
        if algo in {"bc", "gc_bc"}:
            if "enabled" in rsal:
                cfg_overrides["config.agent_kwargs.policy_kwargs.saliency.enabled"] = str(bool(rsal["enabled"]))
            if "weight" in rsal and rsal["weight"] is not None:
                cfg_overrides["config.agent_kwargs.policy_kwargs.saliency.weight"] = str(rsal["weight"])
            if "beta" in rsal and rsal["beta"] is not None:
                cfg_overrides["config.agent_kwargs.policy_kwargs.saliency.beta"] = str(rsal["beta"])
        elif algo == "gc_ddpm_bc":
            if "enabled" in rsal:
                cfg_overrides["config.agent_kwargs.saliency.enabled"] = str(bool(rsal["enabled"]))
            if "weight" in rsal and rsal["weight"] is not None:
                cfg_overrides["config.agent_kwargs.saliency.weight"] = str(rsal["weight"])
            if "beta" in rsal and rsal["beta"] is not None:
                cfg_overrides["config.agent_kwargs.saliency.beta"] = str(rsal["beta"])

    # Pass through global knobs: amp / compile / dataloader / profiler / scheduler / wandb
    for knob_key in ("amp", "compile", "dataloader", "profiler", "scheduler", "wandb"):
        knob = g.get(knob_key)
        if knob:
            flat: Dict[str, Any] = {}
            _flatten(f"config.{knob_key}", knob, flat)
            cfg_overrides.update(flat)

    # Encoder selection (global and per-run overrides)
    # Allow YAML to specify either explicit encoder name or backbone via encoder_kwargs.arch
    enc_global = g.get("encoder")
    enc_kwargs_global = g.get("encoder_kwargs", {}) or {}
    if enc_global:
        cfg_overrides["config.encoder"] = enc_global
    if enc_kwargs_global:
        enc_kw_flat: Dict[str, Any] = {}
        _flatten("config.encoder_kwargs", enc_kwargs_global, enc_kw_flat)
        cfg_overrides.update(enc_kw_flat)

    # Per-run encoder override
    if "encoder" in run:
        cfg_overrides["config.encoder"] = run["encoder"]
    if "encoder_kwargs" in run and run["encoder_kwargs"]:
        enc_kw_flat_r: Dict[str, Any] = {}
        _flatten("config.encoder_kwargs", run["encoder_kwargs"], enc_kw_flat_r)
        cfg_overrides.update(enc_kw_flat_r)

    # Support shorthand per-run backbone field (e.g., resnet18/34/50)
    if "backbone" in run and run["backbone"]:
        # Use generic builder name and pass arch via encoder_kwargs.arch if not explicitly set
        cfg_overrides.setdefault("config.encoder", "resnetv1-bridge")
        cfg_overrides["config.encoder_kwargs.arch"] = run["backbone"]

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
        encoders = matrix.get("encoders")  # explicit encoder names
        backbones = matrix.get("backbones")  # shorthand backbone arch names
        saliency_weights = matrix.get("saliency_weights")  # list of scalars to sweep
        obs_horizons = matrix.get("obs_horizons")  # list of ints to sweep dataset obs_horizon

        if encoders:
            runs = [{"algo": a, "task": t, "encoder": e} for a in algos for t in tasks for e in encoders]
        elif backbones:
            runs = [{"algo": a, "task": t, "backbone": b} for a in algos for t in tasks for b in backbones]
        else:
            runs = [{"algo": a, "task": t} for a in algos for t in tasks]

        if saliency_weights:
            expanded = []
            for r in runs:
                for w in saliency_weights:
                    r2 = dict(r)
                    r2["saliency_weight"] = w
                    expanded.append(r2)
            runs = expanded

        if obs_horizons:
            expanded = []
            for r in runs:
                for oh in obs_horizons:
                    r2 = dict(r)
                    ds = dict(r2.get("dataset", {}))
                    ds["obs_horizon"] = int(oh)
                    r2["dataset"] = ds
                    expanded.append(r2)
            runs = expanded

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
