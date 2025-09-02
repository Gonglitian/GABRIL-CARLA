#!/usr/bin/env python3
"""
DDP 单轮训练测试：在 CUDA 设备 4/5/6/7 上分别对 Gaze 与 BC 训练脚本跑 1 个 epoch
要求：torchrun 可用，数据路径可访问。
"""

import os
import subprocess
from pathlib import Path


ROOT = "/home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA"


def run_cmd(cmd, env=None):
    print("\n==> Run:", cmd)
    proc = subprocess.Popen(cmd, shell=True, env=env)
    ret = proc.wait()
    print("<== Return:", ret)
    return ret == 0


def test_ddp_gaze_one_epoch():
    env = os.environ.copy()
    # 绑定到 GPU 4,5,6,7
    env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    # 开启分布式
    overrides = [
        "training.distributed.enabled=true",
        "training.epochs=5",
        "training.save_interval=1",
        "training.vis_interval=1000",
        "data.num_episodes=8",
        f"logging.log_dir={ROOT}/logs/gaze",
        f"logging.checkpoint_dir={ROOT}/trained_models/gaze",
        "data.batch_size=4000",
        "data.num_workers=8",
        "data.cache_mode=all",
    ]
    override_str = " ".join(overrides)
    cmd = f"cd {ROOT} && torchrun --nproc_per_node=4 --standalone vlm_gaze/train/train_gaze_predictor.py {override_str}"
    return run_cmd(cmd, env)


def test_ddp_bc_one_epoch():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    overrides = [
        "training.distributed.enabled=true",
        "training.epochs=1",
        "training.save_interval=1",
        "data.num_episodes=8",
        "data.hdf5_path=/data3/vla-reasoning/dataset/bench2drive220_robomimic_large_chunk.hdf5",
        f"logging.log_dir={ROOT}/logs/bc",
        f"logging.checkpoint_dir={ROOT}/trained_models/bc",
        "data.batch_size=4",
        "data.num_workers=2",
        "data.cache_mode=low_dim",
        "gaze.method=Reg",
    ]
    override_str = " ".join(overrides)
    cmd = f"cd {ROOT} && torchrun --nproc_per_node=4 --standalone vlm_gaze/train/train_bc.py {override_str}"
    return run_cmd(cmd, env)


def main():
    print("\n==============================")
    print("DDP One-epoch Training Tests")
    print("==============================\n")

    ok1 = test_ddp_gaze_one_epoch()
    ok2 = test_ddp_bc_one_epoch()

    print("\nSummary:")
    print("Gaze DDP:", "OK" if ok1 else "FAIL")
    print("BC   DDP:", "OK" if ok2 else "FAIL")

    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())


