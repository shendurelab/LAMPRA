#!/bin/bash
#============================================================
# SGE Submission - Model comparison tables
#============================================================
# Runs after training + holdout evaluation. Produces:
#   data/combined_df.csv               (model_comparison_merge.py)
#   data/holdout_eval_combined.csv     (model_comparison_v3.1.py)
#   data/model_comparison_v3.1_*/holdout_pred_*.csv
#
# Usage (submit from the repo root):
#   qsub scripts/submit_model_comparison.sh
#============================================================

#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -l mem_free=16G
#$ -pe serial 2
#$ -o logs/model_comparison_$JOB_ID.log
#$ -e logs/model_comparison_$JOB_ID.err
#$ -N model_comparison_cat

# PROJECT_ROOT is the directory you submit from. Run all qsub commands from
# the repo root, e.g.:  qsub scripts/submit_model_comparison.sh
PROJECT_ROOT="$(pwd)"

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
echo "MODEL COMPARISON — categorical"
echo "Date: $(date)  Host: $(hostname)  Job: $JOB_ID"
echo "============================================================"

cd "$PROJECT_ROOT"
python notebooks/model_comparison_merge.py
EC1=$?
python notebooks/model_comparison_v3.1.py
EC2=$?

OVERALL=0
[ $EC1 -ne 0 ] && OVERALL=$EC1
[ $EC2 -ne 0 ] && OVERALL=$EC2
echo "============================================================"
echo "DONE  merge_exit=$EC1  v3.1_exit=$EC2  overall=$OVERALL  Finished: $(date)"
echo "============================================================"
exit $OVERALL
