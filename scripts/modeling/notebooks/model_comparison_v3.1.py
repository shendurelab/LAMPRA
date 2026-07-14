#!/usr/bin/env python3
"""Model Comparison v3.1 — Holdout Predictions for Full + iiiF_only.

- All families (linear, linear-interactions, rfxgb) read from the SAME
  10-int-column CSV (data/processed/categorical/full_processed_categorical.csv
  or its iiiF_only variant). Linear pipelines handle one-hot expansion
  internally; RF takes ints directly; XGB needs pd.Categorical casting.
- Single-split only (no cross-validation).

Outputs:
    data/model_comparison_v3.1_20260407/holdout_eval_combined.csv
    data/holdout_eval_combined.csv  (top-level convenience copy)
    plus per-model holdout_pred_*.csv files in the v3.1 dir.
"""

import os
import sys
from glob import glob

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Make the scripts package importable for utils
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
sys.path.insert(0, SCRIPT_DIR)
from utils import CAT_COLS, cast_to_categorical, make_iiiF_mask  # noqa: E402

# Repo root = parent of this notebooks/ directory (portable; no hardcoded path).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROC_DIR = os.path.join(DATA_DIR, 'processed', 'categorical')
RAW_DATA = os.path.join(DATA_DIR, 'raw', 'v2_longMPRA_scores_with_orientation_20260127.txt')

HOLDOUT_SIZE = 0.15
VAL_SIZE = 0.10
RANDOM_STATE = 42

V31_DIR = os.path.join(DATA_DIR, 'model_comparison_v3.1_20260407')
os.makedirs(V31_DIR, exist_ok=True)


# ----------------------------- helpers --------------------------------------

def load_best_model(family, subset, model_name):
    """Load the latest best model .pkl for a given family/subset/model."""
    pattern = os.path.join(MODELS_DIR, family, subset, f'{model_name}_best_*.pkl')
    matches = sorted(glob(pattern))
    if not matches:
        return None, None
    return joblib.load(matches[-1]), matches[-1]


def load_data_and_split(subset):
    """Load processed categorical data and recreate train/val/holdout splits.

    Returns X (10 int cols), y, and a dict of splits. The same X/y is used
    for every family — only the model itself differs.
    """
    if subset == 'full':
        filename = 'full_processed_categorical.csv'
    elif subset == 'iiiF_only':
        filename = 'subset_iiiF_only.csv'
    elif subset == 'except_iiiF':
        filename = 'subset_except_iiiF.csv'
    else:
        raise ValueError(subset)
    df = pd.read_csv(os.path.join(PROC_DIR, filename))
    X = df[CAT_COLS]
    y = df['avg_Rep'].astype(float)
    X_pool, X_h, y_pool, y_h = train_test_split(
        X, y, test_size=HOLDOUT_SIZE, random_state=RANDOM_STATE)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_pool, y_pool, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    return X, y, {'X_train': X_tr, 'X_val': X_v, 'X_holdout': X_h,
                  'y_train': y_tr, 'y_val': y_v, 'y_holdout': y_h}


def build_subset_metadata(raw_df, subset, meta_cols, target='avg_Rep'):
    """Raw metadata rows in the SAME order as the processed subset CSV.

    Mirrors preprocess.clean_data (numeric-coerce target, drop NaN-target
    rows, reset_index) and the subset mask, so the resulting RangeIndex
    aligns 1:1 with the split labels (X_holdout.index). Without this, the
    split labels (subset 0..n-1) were resolved against the full raw frame,
    attaching the wrong rows' metadata.
    """
    clean = raw_df.copy()
    clean[target] = pd.to_numeric(clean[target], errors='coerce')
    clean = clean.dropna(subset=[target]).reset_index(drop=True)
    if subset == 'full':
        sub = clean
    elif subset == 'iiiF_only':
        sub = clean[make_iiiF_mask(clean)].reset_index(drop=True)
    elif subset == 'except_iiiF':
        sub = clean[~make_iiiF_mask(clean)].reset_index(drop=True)
    else:
        raise ValueError(subset)
    return sub[[c for c in meta_cols if c in sub.columns]]


def is_xgb(model):
    return type(model).__name__.startswith('XGB')


# ----------------------------- main -----------------------------------------

def main():
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')
    print(f'V3.1 output:  {V31_DIR}\n')

    raw_df = pd.read_csv(RAW_DATA, sep='\t')
    metadata_cols = [
        'insert_combo',
        'insert_1', 'insert_1_id', 'insert_1_ori',
        'insert_2', 'insert_2_id', 'insert_2_ori',
        'insert_3', 'insert_3_id', 'insert_3_ori',
        'insert_4', 'insert_4_id', 'insert_4_ori',
        'insert_5', 'insert_5_id', 'insert_5_ori',
    ]
    available_meta = [c for c in metadata_cols if c in raw_df.columns]
    print(f'Raw rows: {len(raw_df)}')

    # Same 10-col splits drive every family. Metadata is rebuilt per subset so
    # its row order matches the processed CSV (and thus the split labels).
    splits = {}
    subset_metadata = {}
    for subset in ['full', 'iiiF_only']:
        print(f'Loading splits for subset={subset}...')
        _, _, sp = load_data_and_split(subset)
        splits[subset] = sp
        subset_metadata[subset] = build_subset_metadata(raw_df, subset, available_meta)
        print(f'  Holdout: {len(sp["X_holdout"])} samples  '
              f'(metadata rows: {len(subset_metadata[subset])})')

    model_configs = [
        # --- full subset ---
        ('rfxgb',              'full', 'rf',         'RF'),
        ('rfxgb',              'full', 'xgb',        'XGB'),
        ('linear',             'full', 'ridge',      'Ridge'),
        ('linear',             'full', 'lasso',      'Lasso'),
        ('linear',             'full', 'elasticnet', 'ElasticNet'),
        ('linear',             'full', 'ols',        'OLS'),
        ('linear-interactions','full', 'ridge',      'Ridge+Int'),
        ('linear-interactions','full', 'elasticnet', 'ElasticNet+Int'),
        ('linear-interactions','full', 'lasso',      'Lasso+Int'),
        ('linear-interactions','full', 'ols',        'OLS+Int'),
        # --- iiiF_only subset ---
        ('rfxgb',              'iiiF_only', 'rf',         'RF'),
        ('rfxgb',              'iiiF_only', 'xgb',        'XGB'),
        ('linear',             'iiiF_only', 'ridge',      'Ridge'),
        ('linear',             'iiiF_only', 'lasso',      'Lasso'),
        ('linear',             'iiiF_only', 'elasticnet', 'ElasticNet'),
        ('linear',             'iiiF_only', 'ols',        'OLS'),
        ('linear-interactions','iiiF_only', 'ridge',      'Ridge+Int'),
        ('linear-interactions','iiiF_only', 'elasticnet', 'ElasticNet+Int'),
        ('linear-interactions','iiiF_only', 'lasso',      'Lasso+Int'),
        ('linear-interactions','iiiF_only', 'ols',        'OLS+Int'),
    ]

    all_records = []
    for family, data_subset, model_name, display_name in model_configs:
        model, pkl_path = load_best_model(family, data_subset, model_name)
        if model is None:
            print(f'  SKIP {family}/{data_subset}/{model_name}: no model found')
            continue

        sp = splits[data_subset]
        X_h = sp['X_holdout']
        y_h = sp['y_holdout']
        holdout_idx = X_h.index

        Xp = cast_to_categorical(X_h) if is_xgb(model) else X_h

        try:
            yp = model.predict(Xp)
            r2 = r2_score(y_h, yp)
        except Exception as e:
            print(f'  SKIP {display_name} ({family}/{data_subset}): {e}')
            continue

        df = pd.DataFrame({
            'actual_avg_Rep': y_h.values,
            'predicted_avg_Rep': yp,
        })
        meta = subset_metadata[data_subset].loc[holdout_idx].reset_index(drop=True)
        df = pd.concat([df, meta], axis=1)
        df['family'] = family
        df['data_subset'] = data_subset
        df['model'] = model_name
        df['display_name'] = display_name
        df['pkl_file'] = os.path.basename(pkl_path)
        df['pkl_path'] = pkl_path
        df['holdout_r2'] = r2

        per_model_csv = os.path.join(V31_DIR, f'holdout_pred_{family}_{data_subset}_{model_name}.csv')
        df.to_csv(per_model_csv, index=False)
        print(f'  {family}/{data_subset}/{display_name}: holdout R2 = {r2:.4f}  ({len(df)} rows)')
        all_records.append(df)

    print(f'\n{len(all_records)} models evaluated.')

    combined = pd.concat(all_records, ignore_index=True)

    v31_path = os.path.join(V31_DIR, 'holdout_eval_combined.csv')
    combined.to_csv(v31_path, index=False)
    print(f'Saved: {v31_path}')

    top_path = os.path.join(DATA_DIR, 'holdout_eval_combined.csv')
    combined.to_csv(top_path, index=False)
    print(f'Saved: {top_path}')

    summary = combined.groupby(
        ['family', 'data_subset', 'model', 'pkl_file']
    ).agg(
        n_samples=('actual_avg_Rep', 'count'),
        holdout_r2=('holdout_r2', 'first'),
    ).reset_index()
    print('\n--- Holdout R2 summary ---')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
