#!/bin/bash
#============================================================
# SGE Submission - Holdout Evaluation for ALL best models
#============================================================
# Iterates *_best_*.pkl files across:
#   linear/{full,iiiF_only,except_iiiF}              -> linear processed CSV
#   linear-interactions/{full,iiiF_only,except_iiiF} -> same CSV (pipeline expands)
#   rfxgb/{full,iiiF_only,except_iiiF}               -> same CSV
#
# Usage: qsub submit_evaluate_best.sh
#============================================================

#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -l mem_free=16G
#$ -pe serial 4
#$ -o logs/eval_best_$JOB_ID.log
#$ -e logs/eval_best_$JOB_ID.err
#$ -N eval_best_cat

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
echo "EVALUATE ALL BEST MODELS — categorical"
echo "Date: $(date)  Host: $(hostname)  Job: $JOB_ID"
echo "============================================================"

declare -A SUBSET_DATA=(
    [full]="$PROC_DIR/full_processed_categorical.csv"
    [iiiF_only]="$PROC_DIR/subset_iiiF_only.csv"
    [except_iiiF]="$PROC_DIR/subset_except_iiiF.csv"
)

SPLIT_ARGS="--holdout-size 0.15 --val-size 0.10 --random-state 42"
COUNT=0
FAIL=0
OVERALL=0

evaluate_dir() {
    local MODEL_DIR="$1"
    local DATA="$2"
    local LABEL="$3"
    echo ""
    echo "------------------------------------------------------------"
    echo "$LABEL"
    echo "  Models: $MODEL_DIR"
    echo "  Data:   $DATA"
    echo "------------------------------------------------------------"
    if [ ! -d "$MODEL_DIR" ]; then echo "  SKIP: missing dir"; return; fi
    if [ ! -f "$DATA" ]; then echo "  SKIP: missing data"; return; fi

    for PKL in "$MODEL_DIR"/*_best_*.pkl; do
        [ -f "$PKL" ] || continue
        COUNT=$((COUNT + 1))
        echo "  [$COUNT] $(basename "$PKL")"
        python "$SCRIPT_DIR/evaluate_holdout.py" \
            --model-path "$PKL" --data-path "$DATA" $SPLIT_ARGS
        EC=$?
        if [ $EC -ne 0 ]; then
            FAIL=$((FAIL + 1))
            OVERALL=$EC
            echo "  FAILED ($EC)"
        fi
    done
}

for FAMILY in linear linear-interactions rfxgb; do
    for SUBSET in full iiiF_only except_iiiF; do
        evaluate_dir \
            "$PROJECT_ROOT/models/$FAMILY/$SUBSET" \
            "${SUBSET_DATA[$SUBSET]}" \
            "$FAMILY / $SUBSET"
    done
done

echo ""
echo "============================================================"
echo "DONE  evaluated=$COUNT  failures=$FAIL  exit=$OVERALL"
echo "Finished: $(date)"
echo "============================================================"
exit $OVERALL
