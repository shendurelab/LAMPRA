#!/usr/bin/env python3
"""LAMPRA Holdout Evaluation — Categorical edition.

Loads any best model (linear pipeline / RF / XGB-native-cat), reproduces
the train/val/holdout split, and evaluates on all three.

For XGBoost models we cast the input to pd.Categorical with the stable
categories defined in utils.py (XGB stores its own categories internally
but we re-cast for safety against pandas dtype drift).

Usage:
    python evaluate_holdout.py \
        --model-path ../models/rfxgb/full/rf_best_*.pkl \
        --data-path ../data/processed/categorical/full_processed_categorical.csv

    python evaluate_holdout.py \
        --model-path ../models/linear/full/ridge_best_*.pkl \
        --data-path ../data/processed/categorical/full_processed_categorical.csv
"""

import argparse
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils import CAT_COLS, cast_to_categorical


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model-path', type=str, required=True)
    p.add_argument('--data-path', type=str, required=True)
    p.add_argument('--target', type=str, default='avg_Rep')
    p.add_argument('--holdout-size', type=float, default=0.15)
    p.add_argument('--val-size', type=float, default=0.10)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--output', type=str, default=None)
    return p.parse_args()


def resolve(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        sys.exit(f"ERROR: no model files matching {pattern}")
    if len(matches) > 1:
        print(f"Multiple matches; using most recent: {matches[-1]}")
    return matches[-1]


def is_xgb(model):
    cls = type(model).__name__
    if cls in ('XGBRegressor', 'XGBClassifier'):
        return True
    # pipeline-wrapped (not used for our XGB; included for safety)
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if type(step).__name__.startswith('XGB'):
                return True
    return False


def main():
    args = parse_args()
    model_path = resolve(args.model_path)
    print(f"Model: {model_path}")
    print(f"Data:  {args.data_path}")

    if not os.path.exists(args.data_path):
        sys.exit(f"ERROR: data not found: {args.data_path}")

    model = joblib.load(model_path)
    print(f"Model type: {type(model).__name__}")

    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        sys.exit(f"ERROR: target {args.target} not in columns")
    X = df[CAT_COLS].copy()
    y = df[args.target].astype(float)

    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=args.holdout_size, random_state=args.random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool, test_size=args.val_size, random_state=args.random_state)

    print(f"\nSplits (reproduced): train={len(X_train)} val={len(X_val)} holdout={len(X_holdout)}")

    needs_cat = is_xgb(model)
    if needs_cat:
        print("Detected XGBoost model — casting inputs to pd.Categorical")

    results = {}
    for split, Xs, ys in [('train', X_train, y_train),
                          ('val', X_val, y_val),
                          ('holdout', X_holdout, y_holdout)]:
        Xp = cast_to_categorical(Xs) if needs_cat else Xs
        yp = model.predict(Xp)
        results[split] = {
            'r2': float(r2_score(ys, yp)),
            'rmse': float(np.sqrt(mean_squared_error(ys, yp))),
            'mae': float(mean_absolute_error(ys, yp)),
            'n_samples': int(len(ys)),
        }

    print(f"\n{'Split':<10} {'R2':<10} {'RMSE':<10} {'MAE':<10} {'N':<8}")
    for s in ['train', 'val', 'holdout']:
        r = results[s]
        print(f"{s:<10} {r['r2']:<10.4f} {r['rmse']:<10.4f} {r['mae']:<10.4f} {r['n_samples']:<8}")

    overfit_r2_gap = results['train']['r2'] - results['val']['r2']
    overfit_rmse_ratio = (results['val']['rmse'] / results['train']['rmse']
                          if results['train']['rmse'] > 0 else float('inf'))
    generalization_gap = results['train']['r2'] - results['holdout']['r2']

    print(f"\nOverfit R2 gap (train-val):     {overfit_r2_gap:.4f}")
    print(f"Overfit RMSE ratio (val/train):  {overfit_rmse_ratio:.4f}")
    print(f"Generalization gap (train-hold): {generalization_gap:.4f}")

    out = {
        'model_path': os.path.abspath(model_path),
        'data_path': os.path.abspath(args.data_path),
        'holdout_size': args.holdout_size,
        'val_size': args.val_size,
        'random_state': args.random_state,
        'results': results,
        'overfit_r2_gap': overfit_r2_gap,
        'overfit_rmse_ratio': overfit_rmse_ratio,
        'generalization_gap': generalization_gap,
    }

    if args.output:
        out_path = args.output
    else:
        model_dir = os.path.dirname(model_path)
        base = os.path.basename(model_path).replace('.pkl', '')
        out_path = os.path.join(model_dir, 'results', f"{base}_holdout_eval.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved holdout evaluation: {out_path}")


if __name__ == '__main__':
    sys.exit(main())
