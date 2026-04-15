#!/usr/bin/env bash
# Clone SurRoL, patch it to build on macOS arm64, and install into the active
# env. Idempotent: safe to re-run. Assumes `bsp` conda env is activated.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/third_party"
SURROL_DIR="$VENDOR_DIR/SurRoL"

mkdir -p "$VENDOR_DIR"

if [ ! -d "$SURROL_DIR/.git" ]; then
    echo "[1/4] Cloning SurRoL..."
    git clone --depth=1 https://github.com/med-air/SurRoL.git "$SURROL_DIR"
else
    echo "[1/4] SurRoL already cloned — skipping."
fi

# Patch: on macOS, pkgutil.get_loader('eglRenderer') returns None. Guard it.
ENV_FILE="$SURROL_DIR/Benchmark/state_based/surrol/gym/surrol_env.py"
echo "[2/4] Patching $ENV_FILE for macOS (EGL guard)..."
python3 - "$ENV_FILE" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
old = (
    "            if socket.gethostname().startswith('pc') or True:\n"
    "                # TODO: not able to run on remote server\n"
    "                egl = pkgutil.get_loader('eglRenderer')\n"
    "                plugin = p.loadPlugin(egl.get_filename(), \"_eglRendererPlugin\")"
)
new = (
    "            # EGL is Linux-only; on macOS pkgutil.get_loader returns None.\n"
    "            egl = pkgutil.get_loader('eglRenderer')\n"
    "            if egl is not None:\n"
    "                plugin = p.loadPlugin(egl.get_filename(), \"_eglRendererPlugin\")"
)
if old in text:
    p.write_text(text.replace(old, new))
    print("  patched.")
elif new in text:
    print("  already patched — skipping.")
else:
    sys.exit("  unexpected content; upstream SurRoL may have changed. Aborting.")
PY

echo "[3/4] Installing runtime deps (skipping panda3d/kivymd — not needed for NeedleReach)..."
pip install --quiet \
    "gym==0.26.2" \
    "numpy<2" \
    "roboticstoolbox-python" \
    "pandas" \
    "trimesh"

echo "[4/4] Installing SurRoL editable (--no-deps)..."
pip install --quiet --no-deps -e "$SURROL_DIR/Benchmark/state_based"

echo
echo "Done. Verify with:"
echo "  python -c 'from surrol.tasks.needle_reach_RL import NeedleReach; NeedleReach(render_mode=None).reset(); print(\"ok\")'"
