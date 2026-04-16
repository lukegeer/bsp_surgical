#!/usr/bin/env bash
# Watch for data collection to complete, then kick off training.
# Usage: bash scripts/train_when_ready.sh <data_dir> <ckpt_dir>
set -u
export KMP_DUPLICATE_LIB_OK=TRUE

if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
fi
conda activate bsp

data_dir="$1"
ckpt_dir="$2"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for 500 episodes in $data_dir..."
while [ "$(ls -1 $data_dir/ep_*.npz 2>/dev/null | wc -l | tr -d ' ')" -lt 500 ]; do
    sleep 30
done
log "500 episodes ready in $data_dir — starting training"

python scripts/train_rgbd_dynamics.py --data-dir "$data_dir" \
    --out-dir "$ckpt_dir/dynamics" \
    --epochs 30 --batch-size 32 --chunk-size 5 --weight-decay 1e-4

log "dynamics training done, starting diffusion subgoal"
python scripts/train_diffusion_subgoal.py --data-dir "$data_dir" \
    --encoder-ckpt "$ckpt_dir/dynamics/rgbd_dynamics.pt" \
    --out-dir "$ckpt_dir/subgoal" \
    --epochs 80 --batch-size 64

log "=== $ckpt_dir READY ==="
