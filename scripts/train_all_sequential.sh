#!/usr/bin/env bash
# Wait for each task's data to finish collecting, then train that
# task's encoder + diffusion subgoal in sequence.
# Usage: bash scripts/train_all_sequential.sh
set -u
export KMP_DUPLICATE_LIB_OK=TRUE

if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
fi
conda activate bsp

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_data() {
    local dir="$1"
    local n="$2"
    log "waiting for $n episodes in $dir..."
    while [ "$(ls -1 $dir/ep_*.npz 2>/dev/null | wc -l | tr -d ' ')" -lt "$n" ]; do
        sleep 30
    done
    log "  -> $n episodes ready in $dir"
}

train_task() {
    local data_dir="$1"
    local ckpt_dir="$2"
    local task_label="$3"
    log "=== TRAINING $task_label (data: $data_dir) ==="
    python scripts/train_rgbd_dynamics.py --data-dir "$data_dir" \
        --out-dir "$ckpt_dir/dynamics" \
        --epochs 30 --batch-size 32 --chunk-size 5 --weight-decay 1e-4
    python scripts/train_diffusion_subgoal.py --data-dir "$data_dir" \
        --encoder-ckpt "$ckpt_dir/dynamics/rgbd_dynamics.pt" \
        --out-dir "$ckpt_dir/subgoal" \
        --epochs 80 --batch-size 64
    log "=== $task_label DONE ==="
}

# Sequential: each task waits for its own data, trains, then next task starts
wait_for_data data/needle_reach_sd    500
train_task   data/needle_reach_sd    checkpoints/reach_sd   NeedleReach

wait_for_data data/needle_pick_sd     500
train_task   data/needle_pick_sd     checkpoints/pick_sd    NeedlePick

wait_for_data data/gauze_retrieve_sd  500
train_task   data/gauze_retrieve_sd  checkpoints/gauze_sd   GauzeRetrieve

log "=== ALL TRAINING COMPLETE ==="
