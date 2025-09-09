import os
import sys
import json
import pickle
from typing import Any, Dict


def _print_dict(d: Dict[str, Any], prefix: str = "", max_items: int = 5):
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}: <dict> ({len(v)})")
        elif hasattr(v, "shape"):
            try:
                print(f"{prefix}{k}: array shape={v.shape}, dtype={getattr(v, 'dtype', None)}")
            except Exception:
                print(f"{prefix}{k}: array-like")
        elif isinstance(v, (list, tuple)):
            n = len(v)
            head = v[:max_items]
            preview = head
            # for arrays in list, show shape
            def _fmt(x):
                if hasattr(x, "shape"):
                    return f"<ndarray shape={x.shape} dtype={getattr(x, 'dtype', None)}>"
                return repr(x)
            preview_str = ", ".join(_fmt(x) for x in preview)
            tail = "" if n <= max_items else f", ... (total {n})"
            print(f"{prefix}{k}: list[{n}] = [{preview_str}{tail}]")
        else:
            print(f"{prefix}{k}: {type(v).__name__} = {repr(v)[:200]}")


def inspect_traj(traj_dir: str) -> None:
    obs_pkl = os.path.join(traj_dir, "obs_dict.pkl")
    pol_pkl = os.path.join(traj_dir, "policy_out.pkl")
    agt_pkl = os.path.join(traj_dir, "agent_data.pkl")

    print(f"[traj] {traj_dir}")

    if os.path.isfile(obs_pkl):
        with open(obs_pkl, "rb") as f:
            obs = pickle.load(f)
        print("\n== obs_dict.pkl ==")
        _print_dict(obs, prefix="  ")
    else:
        print("obs_dict.pkl not found")

    if os.path.isfile(pol_pkl):
        with open(pol_pkl, "rb") as f:
            pol = pickle.load(f)
        print("\n== policy_out.pkl ==")
        print(f"  len={len(pol)}")
        print(f"  policy_out type={type(pol)}")
        print(f"  first element type={type(pol[0])}")
        print(f"  first element keys={pol[0].keys()}")
        _print_dict(pol[0], prefix="  ")
    else:
        print("policy_out.pkl not found")

    if os.path.isfile(agt_pkl):
        print("\n== agent_data.pkl ==")
        try:
            with open(agt_pkl, "rb") as f:
                agt = pickle.load(f)
            if isinstance(agt, dict):
                _print_dict(agt, prefix="  ")
            else:
                print(f"  type={type(agt).__name__}")
        except Exception as e:
            print(f"  <skipped: {e.__class__.__name__}: {e}>")
    else:
        print("agent_data.pkl not found")


def auto_pick_one(bdv2_root: str) -> str:
    for task in sorted(os.listdir(bdv2_root)):
        tdir = os.path.join(bdv2_root, task)
        if not os.path.isdir(tdir):
            continue
        for col in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, col)
            raw = os.path.join(cdir, "raw")
            if not os.path.isdir(raw):
                continue
            for grp in sorted(os.listdir(raw)):
                gdir = os.path.join(raw, grp)
                if not os.path.isdir(gdir):
                    continue
                for traj in sorted(os.listdir(gdir)):
                    tdir2 = os.path.join(gdir, traj)
                    if os.path.isdir(tdir2):
                        return tdir2
    raise FileNotFoundError("No trajectory found under bdv2 root")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Inspect one BDV2 trajectory pkl contents")
    p.add_argument("--traj_dir", type=str, default="")
    p.add_argument("--bdv2_root", type=str, default="/scr/litian/dataset/bdv2")
    args = p.parse_args()

    traj = args.traj_dir if args.traj_dir else auto_pick_one(args.bdv2_root)
    inspect_traj(traj)


if __name__ == "__main__":
    main()


