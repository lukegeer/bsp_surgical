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

# install SurRoL editable, without its heavy deps (panda3d/kivymd/rtb).
# We install only what NeedleReach actually imports.
pip install -e third_party/SurRoL/Benchmark/state_based --no-deps

# install this project
pip install -e .
```

If a SurRoL import blows up for a missing module, pip-install that module
individually. We skip `panda3d==1.10.11` (no arm64 wheel), `kivymd`, and
`roboticstoolbox-python` unless something we actually use needs them.

## Status

Empty scaffold. Phase 1 (data collection wrapper) is next.
