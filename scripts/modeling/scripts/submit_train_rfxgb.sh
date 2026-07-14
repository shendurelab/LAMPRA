#!/bin/bash
#============================================================
# SGE Submission - Train RF/XGB Models (categorical input)
#============================================================
# RF:  int codes fed directly (sklearn has no native categorical)
# XGB: enable_categorical=True, tree_method='hist', pd.Categorical cast
#
# Loops over {full, iiiF_only, except_iiiF}.
#
# Usage: qsub submit_train_rfxgb.sh
#============================================================

#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -l mem_free=32G
#$ -pe serial 8
#$ -o logs/train_rfxgb_$JOB_ID.log
#$ -e logs/train_rfxgb_$JOB_ID.err
#$ -N train_rfxgb_cat

# PROJECT_ROOT is the directory you submit from. Run all qsub commands from
# the repo root, e.g.:  qsub scripts/submit_preprocess.sh
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
PROC_DIR="$PROJECT_ROOT/data/processed/categorical"

mkdir -p "$PROJECT_ROOT/logs"

# --- Activate the LAMPRA_ML conda environment (portable: finds conda on PATH) ---
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found on PATH (install Miniconda/Anaconda first)" >&2; exit 1
fi
conda activate LAMPRA_ML || { echo "ERROR: conda activate LAMPRA_ML failed" >&2; exit 1; }

echo "============================================================"
echo "TRAIN RF/XGB (categorical)"
echo "Date: $(date)  Host: $(hostname)  Job: $JOB_ID"
echo "============================================================"

cd "$SCRIPT_DIR"

declare -A INPUTS=(
    [full]="$PROC_DIR/full_processed_categorical.csv"
    [iiiF_only]="$PROC_DIR/subset_iiiF_only.csv"
    [except_iiiF]="$PROC_DIR/subset_except_iiiF.csv"
)

OVERALL=0
for SUBSET in full iiiF_only except_iiiF; do
    INPUT="${INPUTS[$SUBSET]}"
    if [ ! -f "$INPUT" ]; then
        echo "SKIP $SUBSET: input not found ($INPUT)"
        continue
    fi
    OUT="$PROJECT_ROOT/models/rfxgb/$SUBSET"
    mkdir -p "$OUT"

    echo ""
    echo "------------------------------------------------------------"
    echo "RF/XGB — subset=$SUBSET -> $OUT"
    echo "------------------------------------------------------------"
    python train_rfxgb.py \
        --input "$INPUT" --output "$OUT" \
        --target avg_Rep --holdout_size 0.15 --val_size 0.10 \
        --random_state 42 --n_jobs 8 \
        --scoring neg_mean_squared_error --verbose 1
    [ $? -ne 0 ] && OVERALL=1
done

echo ""
echo "============================================================"
echo "DONE  Overall exit: $OVERALL  Finished: $(date)"
echo "============================================================"
exit $OVERALL
