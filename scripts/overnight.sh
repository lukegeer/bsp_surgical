#!/usr/bin/env bash
# Overnight chain: recollect all 3 tasks with proprioception, precompute
# features, train dynamics (with --with-proprio --chunk-size 5) + subgoal,
# eval all 4 planners per task. Writes timestamped summary at end.
#
# Run as:  caffeinate -i bash scripts/overnight.sh 2>&1 | tee overnight.log

export KMP_DUPLICATE_LIB_OK=TRUE

# conda sourcing before set -e — these scripts use patterns set -e trips on
if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh ]; then
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
fi
conda activate bsp

set -u
set -o pipefail
# Don't `set -e` — we want the script to continue past any single stage failure
# and keep producing data for downstream stages.

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# -------- data collection with proprio --------
for task_spec in "NeedleReach:100:data/needle_reach_p" "NeedlePick:150:data/needle_pick_p" "GauzeRetrieve:120:data/gauze_retrieve_p"; do
    IFS=':' read -r task max_steps out_dir <<< "$task_spec"
    if [ -d "$out_dir" ] && [ "$(ls -1 $out_dir/ep_*.npz 2>/dev/null | wc -l)" -ge 500 ]; then
        log "$task: already collected ($(ls -1 $out_dir/ep_*.npz | wc -l) eps), skipping"
    else
        log "collecting $task -> $out_dir"
        python scripts/collect.py --task "$task" --num-episodes 500 --max-steps "$max_steps" \
            --resolution 128 --seed 0 --out-dir "$out_dir"
    fi
done

# -------- feature precompute --------
for out_dir in data/needle_reach_p data/needle_pick_p data/gauze_retrieve_p; do
    feat_dir="$out_dir/features/dinov2-base"
    if [ -d "$feat_dir" ] && [ "$(ls -1 $feat_dir/ep_*.npz 2>/dev/null | wc -l)" -ge 500 ]; then
        log "$out_dir features: already done, skipping"
    else
        log "precomputing features -> $feat_dir"
        python scripts/precompute_features.py --data-dir "$out_dir" \
            --out-dir "$out_dir/features" --backbone dinov2-base --amp
    fi
done

# -------- train dynamics + subgoal, per task --------
for task_spec in "NeedleReach:needle_reach_p:reach_p:80" "NeedlePick:needle_pick_p:pick_p:150" "GauzeRetrieve:gauze_retrieve_p:gauze_p:150"; do
    IFS=':' read -r task raw_sub ckpt_sub max_eval <<< "$task_spec"
    raw_dir="data/$raw_sub"
    feat_dir="$raw_dir/features/dinov2-base"
    ckpt_dir="checkpoints/$ckpt_sub"

    log "training dynamics for $task"
    python scripts/train_dynamics.py \
        --data-dir "$raw_dir" --feature-dir "$feat_dir" \
        --out-dir "$ckpt_dir/dynamics" \
        --epochs 50 --log-every 200 --chunk-size 5 --weight-decay 1e-4 --with-proprio

    log "training subgoal for $task"
    python scripts/train_subgoal.py \
        --data-dir "$raw_dir" --feature-dir "$feat_dir" \
        --out-dir "$ckpt_dir/subgoal" \
        --epochs 100 --log-every 50 --random-windows --weight-decay 1e-4
done

# -------- eval each task with its own models --------
for task_spec in "NeedleReach:reach_p:80" "NeedlePick:pick_p:150" "GauzeRetrieve:gauze_p:150"; do
    IFS=':' read -r task ckpt_sub max_eval <<< "$task_spec"
    ckpt_dir="checkpoints/$ckpt_sub"

    log "evaluating $task"
    python scripts/evaluate_features.py --task "$task" --num-episodes 30 \
        --dynamics-ckpt "$ckpt_dir/dynamics/dynamics.pt" \
        --subgoal-ckpt "$ckpt_dir/subgoal/subgoal.pt" \
        --out-dir "eval/${ckpt_sub}_run_0" --max-steps "$max_eval" --match-res 128
done

# -------- summary --------
log "=== OVERNIGHT DONE ==="
for task in reach_p pick_p gauze_p; do
    f="eval/${task}_run_0/results.json"
    if [ -f "$f" ]; then
        echo "--- $task ---"
        python -c "import json; d=json.load(open('$f')); [print(f'{k:10s}: {v[\"success_rate\"]*100:5.1f}%  steps={v[\"mean_steps\"]:.1f}') for k,v in d.items()]"
    fi
done
