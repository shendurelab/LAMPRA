# LAMPRA modeling

---

## Encoding scheme

Each construct is represented by **10 integer columns** + the target:

```
insert_1_id_code, insert_1_ori_code,
insert_2_id_code, insert_2_ori_code,
insert_3_id_code, insert_3_ori_code,
insert_4_id_code, insert_4_ori_code,
insert_5_id_code, insert_5_ori_code,
avg_Rep                                   # target
```

- `id_code`: 0–10 over the 11 insert types (alphabetical:
  `dinucleotide_shuffled_{i..v}`, `eNMU_region_{i..v}`, `synthetic_insulator`).
- `ori_code`: 0 = forward, 1 = reverse.
- The code ↔ name maps are fixed in `scripts/utils.py` (`ID_LEVELS`, `ORI_LEVELS`,
  …) and shared by every script so categories never drift between preprocessing,
  training, and evaluation.

### How each model family consumes the 10 columns

| Family | Internal handling |
|---|---|
| **Linear** (OLS, Ridge, Lasso, ElasticNet) | `OneHotEncoder(drop='first')` inside the sklearn pipeline → one dummy per non-reference level (references: `dinucleotide_shuffled_i`, `forward`). |
| **Linear-interactions** | OHE → `PolynomialFeatures(degree=2, interaction_only=True)` over the dummies → `StandardScaler`. |
| **RandomForestRegressor** | Integer codes fed directly (sklearn RF has no native categorical support). |
| **XGBRegressor** | `enable_categorical=True, tree_method='hist'`; columns cast to `pd.Categorical` with stable categories before `.fit`. |


### Splits (single split, no cross-validation)

- holdout = 15% of total (never touched during training/tuning)
- val = 10% of the remaining pool
- `random_state = 42`

---

## Installation

Requires [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create -f environment.yml   # creates an env named LAMPRA_ML
conda activate LAMPRA_ML
```

(Or, into an existing Python 3.12 environment: `pip install -r requirements.txt`.)

The `.pkl` models are serialized under scikit-learn 1.6.1 / xgboost 3.1.2;
loading them under other versions may warn or fail — use the pinned env.

---

## Repository layout

```
LAMPRA_ML/
├── environment.yml / requirements.txt   # pinned software stack
├── README.md
├── data/
│   ├── raw/
│   │   └── v2_longMPRA_scores_with_orientation_20260127.txt   # raw MPRA scores (input)
│   └── processed/categorical/           # 10-int-column CSVs (produced by preprocess.py)
│       ├── full_processed_categorical.csv        (36 033 rows)
│       ├── subset_iiiF_only.csv                  (1 665 rows)
│       ├── subset_except_iiiF.csv                (34 368 rows)
│       └── *_metadata.json / *_splits.json
├── scripts/
│   ├── utils.py                         # shared encoding constants + helpers
│   ├── preprocess.py                    # raw → processed categorical CSVs
│   ├── train_linear.py                  # OLS/Ridge/Lasso/ElasticNet (±interactions)
│   ├── train_rfxgb.py                   # RandomForest + XGBoost
│   ├── evaluate_holdout.py              # holdout metrics for any best model
│   ├── downsample_replicates.py         # replicated learning curve (RF, 95% CI)
│   └── submit_*.sh                      # SGE/qsub wrappers (one per stage)
├── notebooks/
│   ├── model_comparison_merge.py        # → data/combined_df.csv
│   └── model_comparison_v3.1.py         # → data/holdout_eval_combined.csv (+ per-model preds)
├── logs/                                # SGE job logs land here
└── models/                              # created by training (RF/XGB/linear .pkl + results/)
```

---

## Running the pipeline

Run the stages in order. Stages 2–3 depend on stage 1; stages 4–6 depend on the
trained models from 2–3.

### Option A — on an SGE cluster (qsub)

Submit **from the repo root** (the scripts use `-cwd`, so relative paths and
`logs/` resolve correctly):

```bash
conda activate LAMPRA_ML
qsub scripts/submit_preprocess.sh                    # 1. raw → processed CSVs
qsub scripts/submit_train_linear.sh                  # 2. linear + linear-interactions, all 3 subsets
qsub scripts/submit_train_rfxgb.sh                   # 3. RF + XGB, all 3 subsets
qsub scripts/submit_evaluate_best.sh                 # 4. holdout eval for every best model
qsub scripts/submit_downsample_rfonly_1pct_50perm.sh # 5. RF learning curve, 50 permutations (95% CI)
python notebooks/model_comparison_merge.py           # 6a. → data/combined_df.csv
python notebooks/model_comparison_v3.1.py            # 6b. → data/holdout_eval_combined.csv
```

### Option B — without a cluster (plain Python)

The Python scripts are cluster-independent; the qsub wrappers only add scheduling.
Equivalent direct calls (run from the repo root, `LAMPRA_ML` env active):

```bash
cd scripts

# 1. Preprocess (already provided under data/processed/; rerun to regenerate)
python preprocess.py \
    --input ../data/raw/v2_longMPRA_scores_with_orientation_20260127.txt \
    --output-dir ../data/processed/categorical --target avg_Rep \
    --holdout-size 0.15 --val-size 0.10 --random-state 42

# 2. Linear models — repeat per subset (full / iiiF_only / except_iiiF),
#    and again with --add_interactions for the linear-interactions family.
python train_linear.py -i ../data/processed/categorical/full_processed_categorical.csv \
    -o ../models/linear/full
python train_linear.py -i ../data/processed/categorical/full_processed_categorical.csv \
    -o ../models/linear-interactions/full --add_interactions

# 3. Random forest + XGBoost — repeat per subset.
python train_rfxgb.py -i ../data/processed/categorical/full_processed_categorical.csv \
    -o ../models/rfxgb/full

# 4. Holdout evaluation of a saved best model (writes *_holdout_eval.json).
python evaluate_holdout.py \
    --model-path '../models/rfxgb/full/rf_best_*.pkl' \
    --data-path ../data/processed/categorical/full_processed_categorical.csv

# 5. Replicated RF learning curve (fixed holdout, 50 nested permutations → 95% CI).
python downsample_replicates.py \
    --rf-model '../models/rfxgb/full/rf_best_*.pkl' --skip-xgb \
    --data-path ../data/processed/categorical/full_processed_categorical.csv \
    --output-dir ../models/rfxgb/full/downsampling_rfonly_1pct_50perm \
    --n-replicates 50 --base-downsample-seed 42 \
    --frac-start 0.01 --frac-step 0.01 --frac-stop 1.00

# 6. Model comparison tables (run from repo root, not scripts/).
cd .. && python notebooks/model_comparison_merge.py && python notebooks/model_comparison_v3.1.py
```

The `submit_*.sh` files show the exact arguments used for every subset in the
published run.

---

## Outputs

- **Per model family/subset** (`models/<family>/<subset>/`): best model `.pkl`,
  `results/<model>_gridsearch_results_*.csv`, `results/<model>_summary_*.json`,
  and (linear only) `results/<model>_coefficients_*.csv`.
- **Holdout evaluation**: `results/<model>_best_*_holdout_eval.json`.
- **Learning curve** (`models/rfxgb/full/downsampling_rfonly_1pct_50perm/`):
  `downsampling_metrics.csv` (raw per-replicate metrics — one row per
  `model × replicate × fraction`). Confidence intervals and the learning-curve
  plot are produced downstream (in R) from these raw metrics, not stored here.
- **Comparison tables**: `data/combined_df.csv`, `data/holdout_eval_combined.csv`.

