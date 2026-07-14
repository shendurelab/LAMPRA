#!/usr/bin/env python3
"""LAMPRA RF/XGB Training — Categorical input.

Reads the 10-int-column categorical CSV and trains:
- RandomForestRegressor: feeds int codes directly (sklearn RF has no
  native categorical support; integer-as-ordinal is the best it can do).
- XGBRegressor: enable_categorical=True, tree_method='hist', and we cast
  the 10 columns to pandas Categorical with stable categories (defined
  in utils.py) before fitting so XGBoost performs optimal categorical
  splits rather than ordinal splits.

Single train/val/holdout split (no CV); only the best model per family
is saved (no top-3 .pkl candidates).

Usage:
    python train_rfxgb.py \
        --input ../data/processed/categorical/full_processed_categorical.csv \
        --output ../models/rfxgb/full
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from utils import CAT_COLS, cast_to_categorical

warnings.filterwarnings('ignore')

SEP = "=" * 70


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', '-i', type=str, required=True)
    p.add_argument('--output', '-o', type=str, required=True)
    p.add_argument('--target', '-t', type=str, default='avg_Rep')
    p.add_argument('--holdout_size', type=float, default=0.15)
    p.add_argument('--val_size', type=float, default=0.10)
    p.add_argument('--random_state', type=int, default=42)
    p.add_argument('--n_jobs', type=int, default=8)
    p.add_argument('--scoring', type=str, default='neg_mean_squared_error',
                   choices=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'])
    p.add_argument('--skip_rf', action='store_true')
    p.add_argument('--skip_xgb', action='store_true')
    p.add_argument('--verbose', '-v', type=int, default=1)
    return p.parse_args()


def validate(a):
    if not os.path.exists(a.input):
        sys.exit(f"ERROR: input not found: {a.input}")
    if not (0 < a.holdout_size < 1):
        sys.exit("ERROR: holdout_size out of range")
    if not (0 < a.val_size < 1):
        sys.exit("ERROR: val_size out of range")
    if a.skip_rf and a.skip_xgb:
        sys.exit("ERROR: cannot skip both RF and XGB")
    if not a.skip_xgb and not HAS_XGB:
        sys.exit("ERROR: xgboost not installed; use --skip_xgb")


def load_data(path, target):
    print(f"Loading data: {path}")
    df = pd.read_csv(path)
    if target not in df.columns:
        sys.exit(f"ERROR: target {target} not in columns")
    X = df[CAT_COLS].copy()
    y = df[target].astype(float)
    print(f"  Features: {X.shape[1]}, samples: {X.shape[0]}")
    return X, y


def create_splits(X, y, holdout_size, val_size, random_state):
    X_pool, X_h, y_pool, y_h = train_test_split(
        X, y, test_size=holdout_size, random_state=random_state)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_pool, y_pool, test_size=val_size, random_state=random_state)
    return {'X_train': X_tr, 'X_val': X_v, 'X_holdout': X_h,
            'y_train': y_tr, 'y_val': y_v, 'y_holdout': y_h}


def score(y, yp, metric):
    if metric == 'neg_mean_squared_error':
        return -mean_squared_error(y, yp)
    if metric == 'r2':
        return r2_score(y, yp)
    if metric == 'neg_mean_absolute_error':
        return -mean_absolute_error(y, yp)
    raise ValueError(metric)


def metrics(model, splits, transform=None):
    out = {}
    for split in ['train', 'val']:
        X = splits[f'X_{split}']
        if transform is not None:
            X = transform(X)
        yp = model.predict(X)
        y = splits[f'y_{split}']
        out[f'{split}_r2'] = float(r2_score(y, yp))
        out[f'{split}_rmse'] = float(np.sqrt(mean_squared_error(y, yp)))
        out[f'{split}_mae'] = float(mean_absolute_error(y, yp))
    out['overfit_r2_gap'] = out['train_r2'] - out['val_r2']
    out['overfit_rmse_ratio'] = out['val_rmse'] / out['train_rmse'] if out['train_rmse'] > 0 else float('inf')
    return out


def grid_search(model_factory, param_grid, splits, scoring, verbose, transform=None):
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"  Searching {len(combos)} parameter combinations...")

    rows = []
    best_score = float('-inf')
    best_model = None
    best_params = None

    Xtr = splits['X_train'] if transform is None else transform(splits['X_train'])
    Xv = splits['X_val'] if transform is None else transform(splits['X_val'])

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        model = model_factory(**params)
        model.fit(Xtr, splits['y_train'])
        yp_tr = model.predict(Xtr)
        yp_v = model.predict(Xv)

        train_score = score(splits['y_train'], yp_tr, scoring)
        val_score = score(splits['y_val'], yp_v, scoring)
        train_r2 = r2_score(splits['y_train'], yp_tr)
        val_r2 = r2_score(splits['y_val'], yp_v)
        train_rmse = float(np.sqrt(mean_squared_error(splits['y_train'], yp_tr)))
        val_rmse = float(np.sqrt(mean_squared_error(splits['y_val'], yp_v)))
        train_mae = mean_absolute_error(splits['y_train'], yp_tr)
        val_mae = mean_absolute_error(splits['y_val'], yp_v)

        rows.append({
            **params,
            'train_score': train_score, 'val_score': val_score,
            'train_r2': train_r2, 'val_r2': val_r2,
            'train_rmse': train_rmse, 'val_rmse': val_rmse,
            'train_mae': train_mae, 'val_mae': val_mae,
            'overfit_gap': train_r2 - val_r2,
        })

        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_params = params

        if verbose >= 2:
            print(f"    [{i}/{len(combos)}] val={val_score:.4f} {params}")
        elif verbose >= 1 and i % 50 == 0:
            print(f"    [{i}/{len(combos)}] best val so far: {best_score:.4f}")

    results_df = pd.DataFrame(rows).sort_values('val_score', ascending=False)
    return best_model, best_params, best_score, results_df


def get_rf_grid():
    return {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5],
    }


def get_xgb_grid():
    return {
        'n_estimators': [100, 200, 500],
        'max_depth': [2, 3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5, 7],
    }


def save_model(model, name, output_dir, timestamp):
    pkl = os.path.join(output_dir, f"{name}_best_{timestamp}.pkl")
    joblib.dump(model, pkl)
    print(f"  Saved model: {pkl}")


def save_gridsearch(df, name, results_dir, timestamp):
    p = os.path.join(results_dir, f"{name}_gridsearch_results_{timestamp}.csv")
    df.to_csv(p, index=False)
    print(f"  Saved gridsearch: {p}")


def save_summary(name, params, val_score, met, results_dir, timestamp):
    s = {
        'model_name': name,
        'best_params': params,
        'best_val_score': float(val_score),
        'metrics': met,
        'timestamp': timestamp,
        'tuning_method': 'single_validation_set_no_cv',
    }
    p = os.path.join(results_dir, f"{name}_summary_{timestamp}.json")
    with open(p, 'w') as f:
        json.dump(s, f, indent=2)
    print(f"  Saved summary: {p}")


def save_split_indices(splits, output_dir, timestamp):
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    for split in ['train', 'val', 'holdout']:
        idx = splits[f'X_{split}'].index
        path = os.path.join(results_dir, f'{split}_indices_{timestamp}.csv')
        pd.DataFrame({'original_index': idx}).to_csv(path, index=False)


def print_metrics_block(name, params, val_score, met):
    print(f"\n{name} best params: {params}")
    print(f"Best val score: {val_score:.4f}")
    print(f"  {'Split':<10} {'R2':<10} {'RMSE':<10} {'MAE':<10}")
    for s in ['train', 'val']:
        print(f"  {s:<10} {met[f'{s}_r2']:<10.4f} {met[f'{s}_rmse']:<10.4f} {met[f'{s}_mae']:<10.4f}")
    print(f"  Overfit R2 gap: {met['overfit_r2_gap']:.4f}")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validate(args)
    os.makedirs(args.output, exist_ok=True)
    results_dir = os.path.join(args.output, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print(SEP)
    print("LAMPRA RF/XGB — Categorical input (10 int cols), single split")
    print(SEP)
    print(f"Timestamp: {timestamp}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Holdout: {args.holdout_size}  Val(of pool): {args.val_size}  Seed: {args.random_state}")
    print(SEP)

    X, y = load_data(args.input, args.target)
    splits = create_splits(X, y, args.holdout_size, args.val_size, args.random_state)
    save_split_indices(splits, args.output, timestamp)

    print(f"  Train: {len(splits['X_train'])}  Val: {len(splits['X_val'])}  Holdout: {len(splits['X_holdout'])}")

    all_results = {}

    if not args.skip_rf:
        print("\n" + SEP)
        print("RandomForestRegressor (int-coded inputs)")
        print(SEP)
        rf_factory = lambda **kw: RandomForestRegressor(
            random_state=args.random_state, n_jobs=args.n_jobs, **kw)
        rf_model, rf_params, rf_score, rf_df = grid_search(
            rf_factory, get_rf_grid(), splits, args.scoring, args.verbose, transform=None)
        rf_met = metrics(rf_model, splits, transform=None)
        print_metrics_block("RF", rf_params, rf_score, rf_met)
        save_model(rf_model, 'rf', args.output, timestamp)
        save_gridsearch(rf_df, 'rf', results_dir, timestamp)
        save_summary('rf', rf_params, rf_score, rf_met, results_dir, timestamp)
        all_results['rf'] = rf_met

    if not args.skip_xgb:
        print("\n" + SEP)
        print("XGBRegressor (enable_categorical=True, tree_method='hist')")
        print(SEP)
        xgb_factory = lambda **kw: xgb.XGBRegressor(
            random_state=args.random_state, n_jobs=args.n_jobs,
            objective='reg:squarederror', verbosity=0,
            enable_categorical=True, tree_method='hist', **kw)
        # Cast inputs to pd.Categorical with stable categories so XGB sees true categoricals.
        xgb_model, xgb_params, xgb_score, xgb_df = grid_search(
            xgb_factory, get_xgb_grid(), splits, args.scoring, args.verbose,
            transform=cast_to_categorical)
        xgb_met = metrics(xgb_model, splits, transform=cast_to_categorical)
        print_metrics_block("XGB", xgb_params, xgb_score, xgb_met)
        save_model(xgb_model, 'xgb', args.output, timestamp)
        save_gridsearch(xgb_df, 'xgb', results_dir, timestamp)
        save_summary('xgb', xgb_params, xgb_score, xgb_met, results_dir, timestamp)
        all_results['xgb'] = xgb_met

    summary = {
        'timestamp': timestamp,
        'version': 'lampra_categorical_rfxgb',
        'input_file': os.path.abspath(args.input),
        'output_dir': os.path.abspath(args.output),
        'config': {
            'target': args.target,
            'holdout_size': args.holdout_size,
            'val_size': args.val_size,
            'tuning_method': 'single_validation_set_no_cv',
            'random_state': args.random_state,
            'scoring': args.scoring,
            'n_jobs': args.n_jobs,
            'rf_input_encoding': 'integer_ordinal_no_native_categorical',
            'xgb_input_encoding': 'pandas_Categorical_with_enable_categorical',
        },
        'data': {
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'train_samples': int(len(splits['X_train'])),
            'val_samples': int(len(splits['X_val'])),
            'holdout_samples': int(len(splits['X_holdout'])),
        },
        'results': all_results,
    }
    out_path = os.path.join(results_dir, f'training_summary_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved overall summary: {out_path}")
    print(SEP)
    print("DONE")
    print(SEP)


if __name__ == '__main__':
    sys.exit(main())
