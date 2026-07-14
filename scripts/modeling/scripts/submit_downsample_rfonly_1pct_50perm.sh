#!/bin/bash
#============================================================
# SGE Submission - Downsampling analysis
#   RF ONLY | 1% steps | 50 nested permutations (for 95% CI)
#============================================================
# Repeats the nested downsampling over 50 independent random
# permutations of the ORIGINAL 76.5% train split so the mean
# learning curve gets a 95% interval. Gradient boosting (XGB)
# is skipped. Holdout stays fixed (15%, seed 42) and verified
# against the saved training holdout indices; only the training
# subset varies between replicates.
#
# Replicate r uses permutation seed 42+r (r=0..49), so replicate
# 0 reproduces the original single-draw curve exactly.
#
# Outputs go to a NEW dir so all prior results are preserved.
# models/rfxgb/full/downsampling_rfonly_1pct_50perm/:
#   downsampling_metrics.csv   RAW: one row per (replicate, fraction),
#                              all 5 metrics -> compute any CI downstream
#   downsampling_ci.csv        mean/SD/SE, SE-based 95% CI, 2.5/50/97.5 band
#   learning_curve.png         mean + shaded percentile band
#   downsampling_summary.json  config + seed list + timing
# (per-replicate predictions are NOT saved by design)
#
# Usage: qsub submit_downsample_rfonly_1pct_50perm.sh
#============================================================

#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -l mem_free=48G
#$ -pe serial 8
#$ -o logs/downsample_rfonly_1pct_50perm_$JOB_ID.log
#$ -e logs/downsample_rfonly_1pct_50perm_$JOB_ID.err
#$ -N downsample_rf_50perm

# PROJECT_ROOT is the directory you submit from. Run all qsub commands from
# the repo root, e.g.:  qsub scripts/submit_preprocess.sh
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
DATA_PATH="$PROJECT_ROOT/data/processed/categorical/full_processed_categorical.csv"
RF_DIR="$PROJECT_ROOT/models/rfxgb/full"
OUT_DIR="$RF_DIR/downsampling_rfonly_1pct_50perm"

# Auto-pick the most recent best RF model in the full models dir.
RF_MODEL=$(ls -t "$RF_DIR"/rf_best_*.pkl 2>/dev/null | head -n 1)
if [ -z "$RF_MODEL" ]; then
    echo "ERROR: missing rf_best_*.pkl in $RF_DIR" >&2
    exit 1
fi

# Holdout indices saved at training time (same timestamp as the models).
HOLDOUT_IDX=$(ls -t "$RF_DIR"/results/holdout_indices_*.csv 2>/dev/null | head -n 1)

mkdir -p "$PROJECT_ROOT/logs" "$OUT_DIR"

# --- Activate the LAMPRA_ML conda environment (portable: finds conda on PATH) ---
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found on PATH (install Miniconda/Anaconda first)" >&2; exit 1
fi
conda activate LAMPRA_ML || { echo "ERROR: conda activate LAMPRA_ML failed" >&2; exit 1; }

echo "============================================================"
echo "Downsampling — RF only, 1% steps, 50 nested permutations"
echo "Date: $(date)  Host: $(hostname)  Job: $JOB_ID"
echo "RF model:   $RF_MODEL"
echo "Holdout idx: ${HOLDOUT_IDX:-<none, skipping verification>}"
echo "Output:     $OUT_DIR"
echo "============================================================"

HOLDOUT_ARG=""
[ -n "$HOLDOUT_IDX" ] && HOLDOUT_ARG="--holdout-indices $HOLDOUT_IDX"

python "$SCRIPT_DIR/downsample_replicates.py" \
    --rf-model "$RF_MODEL" \
    --skip-xgb \
    --data-path "$DATA_PATH" \
    --output-dir "$OUT_DIR" \
    --target avg_Rep \
    --holdout-size 0.15 \
    --val-size 0.10 \
    --random-state 42 \
    --n-replicates 50 \
    --base-downsample-seed 42 \
    --frac-start 0.01 --frac-step 0.01 --frac-stop 1.00 \
    $HOLDOUT_ARG

EXIT_CODE=$?
echo "============================================================"
echo "DONE  exit=$EXIT_CODE  Finished: $(date)"
echo "============================================================"
exit $EXIT_CODE
