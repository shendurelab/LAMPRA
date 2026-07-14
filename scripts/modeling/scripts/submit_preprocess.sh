#!/bin/bash
#============================================================
# SGE Submission - Preprocess (categorical, 10 int cols)
#============================================================
#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -l mem_free=8G
#$ -pe serial 1
#$ -o logs/preprocess_$JOB_ID.log
#$ -e logs/preprocess_$JOB_ID.err
#$ -N preprocess_cat

# PROJECT_ROOT is the directory you submit from. Run all qsub commands from
# the repo root, e.g.:  qsub scripts/submit_preprocess.sh
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
RAW_DATA="$PROJECT_ROOT/data/raw/v2_longMPRA_scores_with_orientation_20260127.txt"
OUTPUT_DIR="$PROJECT_ROOT/data/processed/categorical"

mkdir -p "$PROJECT_ROOT/logs" "$OUTPUT_DIR"

# --- Activate the LAMPRA_ML conda environment (portable: finds conda on PATH) ---
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found on PATH (install Miniconda/Anaconda first)" >&2; exit 1
fi
conda activate LAMPRA_ML || { echo "ERROR: conda activate LAMPRA_ML failed" >&2; exit 1; }

echo "============================================================"
echo "PREPROCESS (categorical, 10 int cols)"
echo "Date: $(date)  Host: $(hostname)  Job: $JOB_ID"
echo "Python: $(which python)"
echo "============================================================"

cd "$SCRIPT_DIR"
python preprocess.py \
    --input "$RAW_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --target avg_Rep \
    --holdout-size 0.15 \
    --val-size 0.10 \
    --random-state 42

EC=$?
echo "Exit: $EC"
exit $EC
