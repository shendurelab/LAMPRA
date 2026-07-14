#!/usr/bin/env python3
"""LAMPRA Linear Model Training — Categorical input.

Reads the 10-int-column categorical CSV. Each fitted model is a sklearn
Pipeline that one-hot expands the 10 columns inside itself:

    ColumnTransformer(OneHotEncoder(drop='first'), all 10 cols)
      -> [optional] PolynomialFeatures(degree=2, interaction_only=True)
      -> StandardScaler(with_mean=False)
      -> {OLS, Ridge, Lasso, ElasticNet}

drop='first' makes dinucleotide_shuffled_i (id_code=0) and forward
(ori_code=0) the per-slot reference categories.

Single train/val/holdout split (no CV); only the best model per family
is saved (no top-3 .pkl candidates).

Usage:
    python train_linear.py \
        --input ../data/processed/categorical/full_processed_categorical.csv \
        --output ../models/linear/full

    python train_linear.py \
        --input ../data/processed/categorical/full_processed_categorical.csv \
        --output ../models/linear-interactions/full \
        --add_interactions
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from utils import CAT_COLS, ID_LEVELS, ORI_LEVELS

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
    p.add_argument('--n_jobs', type=int, default=4)
    p.add_argument('--scoring', type=str, default='neg_mean_squared_error',
                   choices=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'])
    p.add_argument('--skip_ols', action='store_true')
    p.add_argument('--skip_ridge', action='store_true')
    p.add_argument('--skip_lasso', action='store_true')
    p.add_argument('--skip_elasticnet', action='store_true')
    p.add_argument('--add_interactions', action='store_true',
                   help='Insert PolynomialFeatures(degree=2, interaction_only=True) AFTER one-hot expansion')
    p.add_argument('--verbose', '-v', type=int, default=1)
    return p.parse_args()


def validate_args(a):
    if not os.path.exists(a.input):
        sys.exit(f"ERROR: input not found: {a.input}")
    if not (0 < a.holdout_size < 1):
        sys.exit("ERROR: holdout_size out of range")
    if not (0 < a.val_size < 1):
        sys.exit("ERROR: val_size out of range")
    if a.holdout_size + a.val_size >= 1:
        sys.exit("ERROR: holdout_size + val_size must be < 1")
    if a.skip_ols and a.skip_ridge and a.skip_lasso and a.skip_elasticnet:
        sys.exit("ERROR: at least one model must be enabled")


def load_data(path, target):
    print(f"Loading data: {path}")
    df = pd.read_csv(path)
    if target not in df.columns:
        sys.exit(f"ERROR: target {target} not in columns")
    cat_cols_present = [c for c in CAT_COLS if c in df.columns]
    if len(cat_cols_present) != len(CAT_COLS):
        missing = set(CAT_COLS) - set(cat_cols_present)
        sys.exit(f"ERROR: missing expected categorical columns: {missing}")
    X = df[CAT_COLS].copy()
    y = df[target].astype(float)
    print(f"  Features: {X.shape[1]} (categorical int cols)")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    return X, y


def create_splits(X, y, holdout_size, val_size, random_state):
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=holdout_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=val_size, random_state=random_state)
    return {'X_train': X_train, 'X_val': X_val, 'X_holdout': X_holdout,
            'y_train': y_train, 'y_val': y_val, 'y_holdout': y_holdout}


def make_preprocessor(add_interactions):
    """OHE(drop='first') on all 10 cat cols, optionally followed by pairwise interactions."""
    id_cats = list(range(len(ID_LEVELS)))
    ori_cats = list(range(len(ORI_LEVELS)))
    ohe_id = OneHotEncoder(categories=[id_cats] * 5, drop='first', sparse_output=False, dtype=np.float64)
    ohe_ori = OneHotEncoder(categories=[ori_cats] * 5, drop='first', sparse_output=False, dtype=np.float64)
    ct = ColumnTransformer([
        ('ohe_id', ohe_id, [c for c in CAT_COLS if c.endswith('_id_code')]),
        ('ohe_ori', ohe_ori, [c for c in CAT_COLS if c.endswith('_ori_code')]),
    ], remainder='drop', verbose_feature_names_out=False)

    steps = [('ohe', ct)]
    if add_interactions:
        steps.append(('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    steps.append(('scaler', StandardScaler(with_mean=False)))
    return Pipeline(steps)


def make_pipeline(model, add_interactions):
    pre = make_preprocessor(add_interactions)
    return Pipeline(pre.steps + [('model', model)])


def get_expanded_feature_names(pipeline):
    """Walk the pipeline up to (but not including) the model step and return output feature names."""
    transform_steps = [(n, t) for n, t in pipeline.steps if n != 'model']
    n_features_seen = None
    names = None
    for i, (name, step) in enumerate(transform_steps):
        if i == 0:
            names = step.get_feature_names_out()
        else:
            names = step.get_feature_names_out(names)
    return list(names)


def compute_metrics(model, X, y, prefix=''):
    yp = model.predict(X)
    return {
        f'{prefix}r2': float(r2_score(y, yp)),
        f'{prefix}rmse': float(np.sqrt(mean_squared_error(y, yp))),
        f'{prefix}mae': float(mean_absolute_error(y, yp)),
    }


def all_metrics(model, splits):
    m = {}
    m.update(compute_metrics(model, splits['X_train'], splits['y_train'], 'train_'))
    m.update(compute_metrics(model, splits['X_val'], splits['y_val'], 'val_'))
    m['overfit_r2_gap'] = m['train_r2'] - m['val_r2']
    m['overfit_rmse_ratio'] = m['val_rmse'] / m['train_rmse'] if m['train_rmse'] > 0 else float('inf')
    return m


def score(y_true, y_pred, metric):
    if metric == 'neg_mean_squared_error':
        return -mean_squared_error(y_true, y_pred)
    if metric == 'r2':
        return r2_score(y_true, y_pred)
    if metric == 'neg_mean_absolute_error':
        return -mean_absolute_error(y_true, y_pred)
    raise ValueError(metric)


def grid_search(model_factory, param_grid, splits, scoring, add_interactions, verbose):
    """Plain grid search on a single validation set. Returns best model + results df."""
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"  Searching {len(combos)} parameter combinations...")

    rows = []
    best_score = float('-inf')
    best_pipeline = None
    best_params = None

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        model = model_factory(**params)
        pipeline = make_pipeline(model, add_interactions)
        pipeline.fit(splits['X_train'], splits['y_train'])

        y_train_pred = pipeline.predict(splits['X_train'])
        y_val_pred = pipeline.predict(splits['X_val'])

        train_score = score(splits['y_train'], y_train_pred, scoring)
        val_score = score(splits['y_val'], y_val_pred, scoring)
        train_r2 = r2_score(splits['y_train'], y_train_pred)
        val_r2 = r2_score(splits['y_val'], y_val_pred)
        train_rmse = float(np.sqrt(mean_squared_error(splits['y_train'], y_train_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(splits['y_val'], y_val_pred)))
        train_mae = mean_absolute_error(splits['y_train'], y_train_pred)
        val_mae = mean_absolute_error(splits['y_val'], y_val_pred)

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
            best_pipeline = pipeline
            best_params = params

        if verbose >= 2:
            print(f"    [{i}/{len(combos)}] val={val_score:.4f} {params}")
        elif verbose >= 1 and i % 10 == 0:
            print(f"    [{i}/{len(combos)}] best val so far: {best_score:.4f}")

    results_df = pd.DataFrame(rows).sort_values('val_score', ascending=False)
    return best_pipeline, best_params, best_score, results_df


def train_ols(splits, add_interactions):
    print("\n" + SEP)
    print("OLS Linear Regression")
    print(SEP)
    pipeline = make_pipeline(LinearRegression(), add_interactions)
    pipeline.fit(splits['X_train'], splits['y_train'])
    metrics = all_metrics(pipeline, splits)
    print_metrics({}, 0.0, metrics)
    return pipeline, {}, 0.0, pd.DataFrame(), metrics


def train_regularized(name, model_factory, param_grid, splits, args):
    print("\n" + SEP)
    print(name)
    print(SEP)
    print(f"  Grid size: {int(np.prod([len(v) for v in param_grid.values()]))} combos")
    best_pipeline, best_params, best_score, results_df = grid_search(
        model_factory, param_grid, splits, args.scoring, args.add_interactions, args.verbose)
    metrics = all_metrics(best_pipeline, splits)
    print_metrics(best_params, best_score, metrics)
    return best_pipeline, best_params, best_score, results_df, metrics


def print_metrics(best_params, best_val_score, metrics):
    print(f"\nBest params: {best_params}")
    if best_val_score:
        print(f"Best val score: {best_val_score:.4f}")
    print(f"  {'Split':<10} {'R2':<10} {'RMSE':<10} {'MAE':<10}")
    for s in ['train', 'val']:
        print(f"  {s:<10} {metrics[f'{s}_r2']:<10.4f} {metrics[f'{s}_rmse']:<10.4f} {metrics[f'{s}_mae']:<10.4f}")
    print(f"  Overfit R2 gap (train-val): {metrics['overfit_r2_gap']:.4f}")


def save_results(pipeline, model_name, params, val_score, results_df, metrics, output_dir, timestamp):
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    pkl_path = os.path.join(output_dir, f"{model_name}_best_{timestamp}.pkl")
    joblib.dump(pipeline, pkl_path)
    print(f"  Saved model: {pkl_path}")

    if len(results_df):
        results_path = os.path.join(results_dir, f"{model_name}_gridsearch_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"  Saved gridsearch: {results_path}")

    feature_names = get_expanded_feature_names(pipeline)
    coefs = pipeline.named_steps['model'].coef_
    intercept = float(pipeline.named_steps['model'].intercept_)
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs),
    }).sort_values('abs_coefficient', ascending=False)
    coef_path = os.path.join(results_dir, f"{model_name}_coefficients_{timestamp}.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"  Saved coefficients: {coef_path}  ({len(coef_df)} features)")

    intercept_path = os.path.join(results_dir, f"{model_name}_intercept_{timestamp}.txt")
    with open(intercept_path, 'w') as f:
        f.write(f"intercept: {intercept}\n")

    summary = {
        'model_name': model_name,
        'best_params': params,
        'best_val_score': float(val_score) if val_score else None,
        'metrics': metrics,
        'intercept': intercept,
        'n_nonzero_coefficients': int((coef_df['coefficient'] != 0).sum()),
        'n_features_post_expansion': int(len(coef_df)),
        'timestamp': timestamp,
        'tuning_method': 'single_validation_set_no_cv',
    }
    summary_path = os.path.join(results_dir, f"{model_name}_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")


def save_split_indices(splits, output_dir, timestamp):
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    for split in ['train', 'val', 'holdout']:
        idx = splits[f'X_{split}'].index
        path = os.path.join(results_dir, f'{split}_indices_{timestamp}.csv')
        pd.DataFrame({'original_index': idx}).to_csv(path, index=False)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validate_args(args)
    os.makedirs(args.output, exist_ok=True)

    print(SEP)
    print("LAMPRA LINEAR MODELS — Categorical input (10 int cols), single split")
    print(SEP)
    print(f"Timestamp: {timestamp}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Target: {args.target}")
    print(f"Holdout: {args.holdout_size}  Val (of pool): {args.val_size}  Seed: {args.random_state}")
    print(f"Add interactions: {args.add_interactions}")
    print(SEP)

    X, y = load_data(args.input, args.target)
    splits = create_splits(X, y, args.holdout_size, args.val_size, args.random_state)
    save_split_indices(splits, args.output, timestamp)

    pre = make_preprocessor(args.add_interactions)
    pre.fit(splits['X_train'])
    n_post = len(get_expanded_feature_names(Pipeline(pre.steps + [('model', LinearRegression())])))
    print(f"\nPost-OHE{' + interactions' if args.add_interactions else ''} feature count: {n_post}")

    all_results = {}

    if not args.skip_ols:
        m, p, s, r, met = train_ols(splits, args.add_interactions)
        save_results(m, 'ols', p, s, r, met, args.output, timestamp)
        all_results['OLS'] = met

    if not args.skip_ridge:
        m, p, s, r, met = train_regularized(
            "Ridge",
            lambda **kw: Ridge(random_state=args.random_state, **kw),
            {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
            splits, args)
        save_results(m, 'ridge', p, s, r, met, args.output, timestamp)
        all_results['Ridge'] = met

    if not args.skip_lasso:
        m, p, s, r, met = train_regularized(
            "Lasso",
            lambda **kw: Lasso(random_state=args.random_state, **kw),
            {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
            splits, args)
        save_results(m, 'lasso', p, s, r, met, args.output, timestamp)
        all_results['Lasso'] = met

    if not args.skip_elasticnet:
        m, p, s, r, met = train_regularized(
            "ElasticNet",
            lambda **kw: ElasticNet(random_state=args.random_state, **kw),
            {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
             'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]},
            splits, args)
        save_results(m, 'elasticnet', p, s, r, met, args.output, timestamp)
        all_results['ElasticNet'] = met

    summary = {
        'timestamp': timestamp,
        'version': 'lampra_categorical_linear',
        'input_file': os.path.abspath(args.input),
        'output_dir': os.path.abspath(args.output),
        'config': {
            'target': args.target,
            'holdout_size': args.holdout_size,
            'val_size': args.val_size,
            'tuning_method': 'single_validation_set_no_cv',
            'random_state': args.random_state,
            'scoring': args.scoring,
            'add_interactions': args.add_interactions,
        },
        'data': {
            'n_samples': int(len(X)),
            'n_input_categorical_cols': int(X.shape[1]),
            'n_features_post_expansion': int(n_post),
            'train_samples': int(len(splits['X_train'])),
            'val_samples': int(len(splits['X_val'])),
            'holdout_samples': int(len(splits['X_holdout'])),
        },
        'results': all_results,
    }
    out_path = os.path.join(args.output, 'results', f'training_summary_{timestamp}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved overall summary: {out_path}")
    print(SEP)
    print("DONE")
    print(SEP)


if __name__ == '__main__':
    sys.exit(main())
