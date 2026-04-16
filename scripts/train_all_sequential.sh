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
    # Usage: train_task <ckpt_dir> <task_label> <data_dir1> [<data_dir2> ...]
    local ckpt_dir="$1"; shift
    local task_label="$1"; shift
    local data_dirs=("$@")
    log "=== TRAINING $task_label (data: ${data_dirs[*]}) ==="
    if [ -f "$ckpt_dir/dynamics/rgbd_dynamics.pt" ]; then
        log "  dynamics checkpoint exists, skipping"
    else
        python scripts/train_rgbd_dynamics.py --data-dir "${data_dirs[@]}" \
            --out-dir "$ckpt_dir/dynamics" \
            --epochs 30 --batch-size 32 --chunk-size 5 --weight-decay 1e-4
    fi
    if [ -f "$ckpt_dir/subgoal/subgoal_diffusion.pt" ]; then
        log "  subgoal diffusion checkpoint exists, skipping"
    else
        python scripts/train_diffusion_subgoal.py --data-dir "${data_dirs[@]}" \
            --encoder-ckpt "$ckpt_dir/dynamics/rgbd_dynamics.pt" \
            --out-dir "$ckpt_dir/subgoal" \
            --epochs 80 --batch-size 64
    fi
    log "=== $task_label DONE ==="
}

# Per-task sanity models (optional — each task trained on its own data)
wait_for_data data/needle_reach_sd 500
train_task checkpoints/reach_sd NeedleReach data/needle_reach_sd

wait_for_data data/needle_pick_sd 500
train_task checkpoints/pick_sd NeedlePick data/needle_pick_sd

# M_simple: the compositional model — jointly trained on reach + pick. This is
# the model we evaluate zero-shot on GauzeRetrieve in the compositional test.
train_task checkpoints/simple_sd "M_simple (reach+pick JOINT)" \
    data/needle_reach_sd data/needle_pick_sd

# Anchor bank built from simple-task training states for rerank at eval
if [ ! -f checkpoints/simple_sd/anchors.npz ]; then
    log "=== BUILDING ANCHOR BANK (reach + pick training states) ==="
    python scripts/build_anchor_bank.py \
        --data-dirs data/needle_reach_sd data/needle_pick_sd \
        --encoder-ckpt checkpoints/simple_sd/dynamics/rgbd_dynamics.pt \
        --out checkpoints/simple_sd/anchors.npz
fi

# M_complex: upper bound — same architecture but trained directly on the target
wait_for_data data/gauze_retrieve_sd 500
train_task checkpoints/gauze_sd "M_complex (gauze directly)" data/gauze_retrieve_sd

log "=== ALL TRAINING COMPLETE ==="
log "    compositional: checkpoints/simple_sd"
log "    upper bound:   checkpoints/gauze_sd"
