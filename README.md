# bsp_surgical

Backward latent subgoal planning for goal-conditioned surgical robot control.
See `../BRAINSTORM.md` for the design rationale and `../LITERATURE_REVIEW.md`
for related work.

## Layout

```
bsp_surgical/
  src/bsp_surgical/     # library code (importable package)
  tests/                # pytest tests (TDD-first)
  configs/              # experiment configs (yaml)
  scripts/              # entry points: collect_data, train, evaluate
  envs/                 # conda env specs
  data/                 # collected demos (gitignored)
  checkpoints/          # model weights (gitignored)
```

## Environment

Single conda env, Python 3.10, arm64 macOS + MPS. SurRoL's `setup.py` lists
`python_requires='>=3.7'`, so we are not actually pinned to 3.7.

```
conda env create -f envs/bsp.yml
conda activate bsp

# clone, patch (macOS EGL guard), and install SurRoL editable --no-deps
bash scripts/setup_surrol.sh

# install this project
pip install -e .
```

`scripts/setup_surrol.sh` is idempotent — safe to re-run. It skips
`panda3d==1.10.11` (no arm64 wheel) and `kivymd` (unused for NeedleReach)
and pins `numpy<2` for ABI compatibility with `roboticstoolbox-python`.

## Phase 1: collect data

```
python scripts/collect.py --task NeedleReach --num-episodes 500 \
    --max-steps 100 --resolution 128 --seed 0 \
    --out-dir data/needle_reach
```

Failures are dropped by default (`--keep-failures` to retain). Each episode
is saved as `data/needle_reach/ep_NNNNN.npz`.

## Status

Empty scaffold. Phase 1 (data collection wrapper) is next.
