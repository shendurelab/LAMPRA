#!/bin/bash
#============================================================
# SGE Submission - Train Linear Models (categorical input)
#============================================================
# Trains OLS, Ridge, Lasso, ElasticNet on the 10-int-column CSVs.
# OneHotEncoder(drop='first') is applied INSIDE each pipeline so
# dinucleotide_shuffled_i (id_code=0) and forward (ori_code=0) are the
# per-slot reference categories.
#
# For each of {full, iiiF_only, except_iiiF} the script runs twice:
#   1. without interactions  -> models/linear/<subset>/
#   2. with pairwise interactions over expanded dummies
#                            -> models/linear-interactions/<subset>/
#
# Usage:
#   qsub submit_train_linear.sh
#============================================================

#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -l mem_free=16G
#$ -pe serial 4
#$ -o logs/train_linear_$JOB_ID.log
#$ -e logs/train_linear_$JOB_ID.err
#$ -N train_linear_cat

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
echo "TRAIN LINEAR (categorical)"
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

    OUT_BASE="$PROJECT_ROOT/models/linear/$SUBSET"
    OUT_INT="$PROJECT_ROOT/models/linear-interactions/$SUBSET"
    mkdir -p "$OUT_BASE" "$OUT_INT"

    echo ""
    echo "------------------------------------------------------------"
    echo "Linear (no interactions) — subset=$SUBSET -> $OUT_BASE"
    echo "------------------------------------------------------------"
    python train_linear.py \
        --input "$INPUT" --output "$OUT_BASE" \
        --target avg_Rep --holdout_size 0.15 --val_size 0.10 \
        --random_state 42 --n_jobs 4 \
        --scoring neg_mean_squared_error --verbose 1
    [ $? -ne 0 ] && OVERALL=1

    echo ""
    echo "------------------------------------------------------------"
    echo "Linear-interactions — subset=$SUBSET -> $OUT_INT"
    echo "------------------------------------------------------------"
    python train_linear.py \
        --input "$INPUT" --output "$OUT_INT" \
        --target avg_Rep --holdout_size 0.15 --val_size 0.10 \
        --random_state 42 --n_jobs 4 \
        --scoring neg_mean_squared_error --add_interactions --verbose 1
    [ $? -ne 0 ] && OVERALL=1
done

echo ""
echo "============================================================"
echo "DONE  Overall exit: $OVERALL  Finished: $(date)"
echo "============================================================"
exit $OVERALL
