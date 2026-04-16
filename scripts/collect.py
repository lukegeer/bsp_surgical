"""Collect demonstrations from a SurRoL oracle policy.

Example:
    python scripts/collect.py --task NeedleReach --num-episodes 500 \\
        --resolution 128 --max-steps 100 --seed 0 --out-dir data/needle_reach
"""
import argparse
import time
from pathlib import Path

import numpy as np

from bsp_surgical.data.collector import collect_episode
from bsp_surgical.data.io import save_trajectory


TASK_REGISTRY = {
    "NeedleReach": ("surrol.tasks.needle_reach_RL", "NeedleReach"),
    "NeedlePick": ("surrol.tasks.needle_pick_RL_2", "NeedlePickRL"),
}


def _make_env(task: str):
    import importlib

    module_name, cls_name = TASK_REGISTRY[task]
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(render_mode=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), default="NeedleReach")
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--keep-failures", action="store_true",
                        help="By default, episodes where the oracle failed are discarded.")
    parser.add_argument("--crop", type=int, nargs=4, default=None, metavar=("Y1", "Y2", "X1", "X2"),
                        help="Crop rendered frame to [Y1:Y2, X1:X2] before resizing. "
                             "Simulates a zoomed surgical camera focused on the workspace.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(args.task)

    saved, failed, total_t = 0, 0, 0
    t_start = time.time()
    for ep in range(args.num_episodes):
        crop_box = tuple(args.crop) if args.crop else None
        traj = collect_episode(
            env,
            lambda obs: env.get_oracle_action(obs),
            max_steps=args.max_steps,
            resolution=args.resolution,
            task_name=args.task,
            episode_id=ep,
            crop_box=crop_box,
        )
        if not traj.success and not args.keep_failures:
            failed += 1
            continue

        path = args.out_dir / f"ep_{ep:05d}.npz"
        save_trajectory(traj, path)
        saved += 1
        total_t += traj.num_transitions

        if (ep + 1) % 25 == 0 or ep == args.num_episodes - 1:
            dt = time.time() - t_start
            print(f"[{ep+1:>5}/{args.num_episodes}] "
                  f"saved={saved} failed={failed} transitions={total_t} "
                  f"({(ep+1)/dt:.2f} ep/s, {total_t/dt:.1f} t/s)")

    print(f"\nDone. {saved} episodes, {total_t} transitions, "
          f"{failed} failures, {time.time()-t_start:.1f}s total.")


if __name__ == "__main__":
    main()
