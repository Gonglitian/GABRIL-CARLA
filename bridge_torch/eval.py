#!/usr/bin/env python3

import os
import sys
import time
import json
import argparse
from datetime import datetime
from collections import deque
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import cv2
from PIL import Image
import imageio


from models.encoders import build_encoder
from agents.bc import BCAgent, BCAgentConfig, MLP as MLP_BC, GaussianPolicy

from experiments.widowx_envs.widowx_env_service import WidowXClient, WidowXStatus, WidowXConfigs
from experiments.utils import state_to_eep, stack_obs


STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
FIXED_STD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}
ACTION_DIM = 7

def _prep_device(arg: str) -> torch.device:
    if arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_config_and_ckpt(save_dir: str) -> Tuple[Dict, str]:
    cfg_path = os.path.join(save_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found in {save_dir}")
    with open(cfg_path, "r") as f:
        config = json.load(f)
    # find latest ckpt_*.pt
    ckpts = [p for p in os.listdir(save_dir) if p.startswith("ckpt_") and p.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {save_dir}")
    # sort by step number
    def _step(p: str) -> int:
        try:
            return int(p.split("_")[-1].split(".")[0])
        except Exception:
            return -1
    ckpts.sort(key=_step)
    ckpt_path = os.path.join(save_dir, ckpts[-1])
    return config, ckpt_path


def _list_ckpts(save_dir: str) -> list:
    ckpts = [p for p in os.listdir(save_dir) if p.startswith("ckpt_") and p.endswith(".pt")]
    def _step(p: str) -> int:
        try:
            return int(p.split("_")[-1].split(".")[0])
        except Exception:
            return -1
    ckpts.sort(key=_step)
    return [os.path.join(save_dir, p) for p in ckpts]


def _load_policy(save_dir: str, device: torch.device, im_size: int, ckpt_path: Optional[str] = None):
    """Load one policy from save_dir and return a registry dict."""
    if ckpt_path is None:
        cfg, ckpt_path = _load_config_and_ckpt(save_dir)
    else:
        cfg_path = os.path.join(save_dir, "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    # build agent/model
    agent, kind, obs_horizon, act_pred_horizon = _build_agent_from_config(cfg, device, im_size)
    # hydrate agent weights, stripping compiled prefixes like "_orig_mod."
    payload = torch.load(ckpt_path, map_location=device)
    model_to_load = agent.model.module if hasattr(agent.model, "module") else agent.model
    state = payload["model"]
    try:
        # remove any occurrence(s) of "_orig_mod." in keys
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    except Exception:
        pass
    model_to_load.load_state_dict(state, strict=False)  # type: ignore
    # make display name
    name = os.path.basename(save_dir.rstrip("/"))
    return {
        "name": name,
        "agent": agent,
        "kind": kind,
        "obs_horizon": obs_horizon,
        "act_pred_horizon": act_pred_horizon,
        "ckpt_path": ckpt_path,
    }


def _discover_runs(root_dir: str) -> list:
    """Discover valid run directories under root_dir (one level).
    A valid run dir contains config.json and at least one ckpt_*.pt
    """
    if not os.path.isdir(root_dir):
        return []
    runs = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        cfg_ok = os.path.isfile(os.path.join(p, "config.json"))
        if not cfg_ok:
            continue
        has_ckpt = any(f.startswith("ckpt_") and f.endswith(".pt") for f in os.listdir(p))
        if has_ckpt:
            runs.append(p)
    return runs


def _build_agent_from_config(cfg: Dict, device: torch.device, im_size: int):
    algo = "bc"
    # Robust encoder resolution with sensible defaults
    enc_name = cfg.get("encoder", "resnetv1-34-bridge")
    enc_kwargs = dict(cfg.get("encoder_kwargs", {})) if isinstance(cfg.get("encoder_kwargs", {}), dict) else {}
    if "arch" not in enc_kwargs:
        enc_kwargs.setdefault("pooling_method", "avg")
        enc_kwargs.setdefault("add_spatial_coordinates", True)
        enc_kwargs.setdefault("act", "swish")
        enc_kwargs["arch"] = "resnet34"
    enc = build_encoder(enc_name, **enc_kwargs).to(device)

    # dummy shapes from dataset config
    data_kwargs = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    obs_horizon = int(data_kwargs.get("obs_horizon", 1))
    act_pred_horizon = int(data_kwargs.get("act_pred_horizon", 1))

    # we will infer action_dim from bridgedata_config metadata

    # Warmup to set encoder feature dim (use im_size from args)
    dummy_image = torch.zeros(1, 3 * obs_horizon, im_size, im_size, device=device)
    _ = enc(dummy_image)
    enc_feat = int(getattr(enc, "_feat_dim"))
    # Use new schema under model
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    use_proprio = bool(model_cfg.get("use_proprio", False))
    prop_dim = 0
    if use_proprio:
        prop_dim = 7  # conservative default; not heavily used at eval time

    # Only BC supported in refactor
    mlp = MLP_BC(
        input_dim=enc_feat + prop_dim,
        hidden_dims=tuple(model_cfg.get("hidden_dims", [256, 256, 256])),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.0)),
    )
    policy_kwargs = {
        "tanh_squash_distribution": bool(model_cfg.get("tanh_squash_distribution", False)),
        "state_dependent_std": bool(model_cfg.get("state_dependent_std", False)),
        "fixed_std": list(model_cfg.get("fixed_std", [])) or None,
        "use_proprio": use_proprio,
    }
    model = GaussianPolicy(
        enc,
        mlp,
        action_dim=ACTION_DIM,
        **policy_kwargs,
    )
    # Optimizer/scheduler/saliency with defaults if missing
    opt_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer", {}), dict) else {}
    sch_cfg = cfg.get("scheduler", {}) if isinstance(cfg.get("scheduler", {}), dict) else {}
    sal_cfg = cfg.get("saliency", {}) if isinstance(cfg.get("saliency", {}), dict) else {}

    agent = BCAgent(
        model=model,
        cfg=BCAgentConfig(
            learning_rate=float(opt_cfg.get("lr", 3e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            warmup_steps=int(sch_cfg.get("warmup_steps", 1000)),
            decay_steps=int(sch_cfg.get("decay_steps", 1000000)),
            scheduler=sch_cfg,
            saliency=sal_cfg,
        ),
        action_dim=ACTION_DIM,
        device=device,
    )
    get_action_kind = "bc"

    return agent, get_action_kind, obs_horizon, act_pred_horizon


def _build_get_action(agent, kind: str, deterministic: bool):
    @torch.no_grad()
    def get_action(obs: Dict[str, np.ndarray], goal_obs: Dict[str, np.ndarray]):
        # convert to torch batch
        def to_tensor_image(x: np.ndarray) -> torch.Tensor:
            if x.ndim == 3 and x.shape[-1] == 3:
                # HWC uint8 [0,255] -> CHW float [0,1]
                t = torch.from_numpy(x).float().permute(2, 0, 1) / 255.0
            elif x.ndim == 4 and x.shape[-1] == 3:
                # THWC -> TCHW then stack as channels
                t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
                t = t.reshape(-1, t.shape[-2], t.shape[-1])  # (T*C,H,W) 不改变 H,W
                # 需要在外部再添加 batch 维度
                t = t
            else:
                t = torch.from_numpy(x).float()
            return t

        device = next(agent.model.parameters()).device
        batch_obs: Dict[str, torch.Tensor] = {}

        image = obs.get("image")
        if image is not None:
            img_t = to_tensor_image(image)
            if img_t.dim() == 3:
                img_t = img_t.unsqueeze(0)
            batch_obs["image"] = img_t.to(device)
        if "proprio" in obs and obs["proprio"] is not None:
            # Use raw proprio (no normalization)
            prop_np = obs["proprio"]
            prop = torch.from_numpy(prop_np).float().unsqueeze(0).to(device)
            batch_obs["proprio"] = prop

        batch_goal: Dict[str, torch.Tensor] = {}
        if goal_obs is not None:
            if "image" in goal_obs and goal_obs["image"] is not None:
                gimg = to_tensor_image(goal_obs["image"])
                if gimg.dim() == 3:
                    gimg = gimg.unsqueeze(0)
                batch_goal["image"] = gimg.to(device)
            if "language" in goal_obs and goal_obs["language"] is not None:
                # torch 版本未实现 language encoder，这里直接传占位（不支持 LC）
                batch_goal["language"] = torch.from_numpy(goal_obs["language"]).float().unsqueeze(0).to(device)

        if kind == "bc":
            act = agent.sample_actions(batch_obs, argmax=deterministic)
        elif kind == "gc_bc":
            act = agent.sample_actions(batch_obs, batch_goal, argmax=deterministic)
        else:
            act_seq = agent.sample_actions(batch_obs, batch_goal)
            act = act_seq[:, 0, :]  # 仅执行第一个时间步

        action = act.squeeze(0).cpu().numpy()
        # Use model output as-is (no de-normalization)
        return action

    return get_action


def request_goal_image(image_goal: Optional[np.ndarray], widowx_client: WidowXClient, im_size: int) -> np.ndarray:
    if image_goal is None:
        ch = "y"
    else:
        ch = input("Taking a new goal? [y/n]")
    if ch == "y":
        widowx_client.move_gripper(1.0)
        input("Press [Enter] when ready for taking the goal image. ")
        obs = widowx_client.get_observation()
        while obs is None:
            print("WARNING retrying to get observation...")
            obs = widowx_client.get_observation()
            time.sleep(1)
        image_goal = (
            obs["image"].reshape(3, im_size, im_size).transpose(1, 2, 0) * 255
        ).astype(np.uint8)
    return image_goal


def main():
    parser = argparse.ArgumentParser(description="Torch real-robot eval for WidowX")
    parser.add_argument("--save_dir", type=str, nargs='+', default=None, help="One or more training run dirs that contain config.json and ckpt_*.pt")
    parser.add_argument("--runs_root", type=str, default=None, help="Root dir containing multiple run subdirs; each subdir should contain config.json and ckpt_*.pt")
    parser.add_argument("--goal_type", type=str, choices=["gc", "bc"], required=True, help="Goal type: gc (goal image) or bc (no goal)")
    parser.add_argument("--im_size", type=int, required=True)
    parser.add_argument("--video_save_path", type=str, default=None)
    parser.add_argument("--goal_image_path", type=str, default=None)
    parser.add_argument("--num_timesteps", type=int, default=120)
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--goal_eep", type=float, nargs=3, default=[0.3, 0.0, 0.15])
    parser.add_argument("--initial_eep", type=float, nargs=3, default=[0.3, 0.0, 0.15])
    parser.add_argument("--act_exec_horizon", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--show_image", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--policy_index", type=int, default=None, help="Non-interactive: select which policy index to use when multiple are provided")
    args = parser.parse_args()

    device = _prep_device(args.device)

    # build run dir list: from --save_dir and/or --runs_root
    run_dirs = []
    if args.save_dir:
        run_dirs.extend(list(args.save_dir))
    if args.runs_root:
        discovered = _discover_runs(args.runs_root)
        if not discovered:
            print(f"[WARN] No valid runs discovered under {args.runs_root}")
        run_dirs.extend(discovered)
    if not run_dirs:
        raise RuntimeError("Please provide --runs_root or --save_dir")

    # First list candidates, then load only the selected policy to avoid heavy init
    names = [os.path.basename(p.rstrip("/")) for p in run_dirs]
    if len(names) == 1:
        selected = 0
    else:
        if args.policy_index is not None:
            selected = int(args.policy_index)
            if selected < 0 or selected >= len(names):
                raise ValueError(f"policy_index out of range: got {selected}, available [0..{len(names)-1}]")
        else:
            print("policies:")
            for i, n in enumerate(names):
                print(f"{i}) {n}")
            selected = int(input("select policy: "))

    # After selecting run, optionally select ckpt from that run
    chosen_run = run_dirs[selected]
    ckpt_files = _list_ckpts(chosen_run)
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {chosen_run}")
    if len(ckpt_files) == 1:
        chosen_ckpt = ckpt_files[0]
    else:
        # show list and let user pick
        print("checkpoints:")
        for i, p in enumerate(ckpt_files):
            print(f"{i}) {os.path.basename(p)}")
        try:
            ckpt_idx = int(input("select ckpt (default latest): ") or str(len(ckpt_files) - 1))
        except Exception:
            ckpt_idx = len(ckpt_files) - 1
        if ckpt_idx < 0 or ckpt_idx >= len(ckpt_files):
            ckpt_idx = len(ckpt_files) - 1
        chosen_ckpt = ckpt_files[ckpt_idx]

    # Load the selected policy with the chosen ckpt
    picked = _load_policy(chosen_run, device, args.im_size, ckpt_path=chosen_ckpt)

    # build get_action wrapper for picked policy
    get_action = _build_get_action(
        picked["agent"],
        picked["kind"],
        args.deterministic,
    )
    obs_horizon = picked["obs_horizon"]
    act_pred_horizon = picked["act_pred_horizon"]

    # set up environment
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["state_state"] = list(np.concatenate([args.initial_eep, [0, 0, 0, 1]]))
    widowx_client = WidowXClient(host=args.ip, port=args.port)
    widowx_client.init(env_params, image_size=args.im_size)

    # optional preview before requesting goal, for monitoring camera stream
    if args.show_image:
        obs = widowx_client.get_observation()
        while obs is None:
            print("Waiting for observations...")
            obs = widowx_client.get_observation()
            time.sleep(1)
        bgr_img = cv2.cvtColor(obs["full_image"], cv2.COLOR_RGB2BGR)
        cv2.imshow("img_view", bgr_img)
        cv2.waitKey(100)

    # load goals
    if args.goal_type == "gc":
        image_goal = None
        if args.goal_image_path is not None:
            image_goal = np.array(Image.open(args.goal_image_path))
    else:  # bc
        image_goal = None

    # reset env
    widowx_client.reset()
    time.sleep(2.5)

    # move to initial position
    if args.initial_eep is not None:
        initial_eep = [float(e) for e in args.initial_eep]
        eep = state_to_eep(initial_eep, 0)
        widowx_client.move_gripper(1.0)
        move_status = None
        while move_status != WidowXStatus.SUCCESS:
            move_status = widowx_client.move(eep, duration=1.5)

    last_tstep = time.time()
    images = []
    image_goals = []
    t = 0
    obs_hist = deque(maxlen=obs_horizon) if (obs_horizon and obs_horizon > 1) else None
    is_gripper_closed = False
    num_consecutive_gripper_change_actions = 0

    try:
        # optionally request goal
        if args.goal_type == "gc":
            image_goal = request_goal_image(image_goal, widowx_client, args.im_size)
            goal_obs = {"image": image_goal}
            input("Press [Enter] to start.")
        else:
            goal_obs = {"image": np.zeros((args.im_size, args.im_size, 3), dtype=np.uint8)}
            input("Press [Enter] to start BC evaluation.")

        while t < args.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or args.blocking:
                obs = widowx_client.get_observation()
                if obs is None:
                    print("WARNING retrying to get observation...")
                    continue

                if args.show_image:
                    bgr_img = cv2.cvtColor(obs["full_image"], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(10)

                image_obs = (
                    obs["image"].reshape(3, args.im_size, args.im_size).transpose(1, 2, 0) * 255
                ).astype(np.uint8)
                curr = {"image": image_obs, "proprio": obs["state"]}
                if obs_hist is not None:
                    if len(obs_hist) == 0:
                        obs_hist.extend([curr] * obs_hist.maxlen)
                    else:
                        obs_hist.append(curr)
                    model_obs = stack_obs(list(obs_hist))
                else:
                    model_obs = curr

                last_tstep = time.time()
                actions = get_action(model_obs, goal_obs)
                if actions.ndim == 1:
                    actions = actions[None]
                for i in range(args.act_exec_horizon):
                    action = actions[i]
                    action += np.random.normal(0, FIXED_STD)

                    # sticky gripper logic
                    if (action[-1] < 0.5) != is_gripper_closed:
                        num_consecutive_gripper_change_actions += 1
                    else:
                        num_consecutive_gripper_change_actions = 0

                    if num_consecutive_gripper_change_actions >= STICKY_GRIPPER_NUM_STEPS:
                        is_gripper_closed = not is_gripper_closed
                        num_consecutive_gripper_change_actions = 0

                    action[-1] = 0.0 if is_gripper_closed else 1.0

                    # remove degrees of freedom
                    if NO_PITCH_ROLL:
                        action[3] = 0
                        action[4] = 0
                    if NO_YAW:
                        action[5] = 0

                    widowx_client.step_action(action, blocking=args.blocking)

                    images.append(image_obs)
                    if args.goal_type == "gc":
                        image_goals.append(image_goal)
                    t += 1
    except Exception as e:
        print(str(e), file=sys.stderr)

    # save video
    if args.video_save_path is not None:
        os.makedirs(args.video_save_path, exist_ok=True)
        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(
            args.video_save_path,
            f"{curr_time}_torch_eval_sticky_{STICKY_GRIPPER_NUM_STEPS}.mp4",
        )
        if args.goal_type == "gc" and image_goals:
            video = np.concatenate([np.stack(image_goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)
        else:
            imageio.mimsave(save_path, images, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    main()
