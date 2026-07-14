#!/usr/bin/env python3
"""LAMPRA Downsampling Analysis — replicated (multi-permutation) edition.

Same nested-downsampling design as downsample_analysis.py, but repeated over
many independent permutations of the training pool so that per-fraction
performance can be summarized with a mean curve and a 95% interval. Built for
the RF-only case (gradient boosting skipped) but model-agnostic.

Design:
- Holdout: 15% of total, random_state=42 (identical two-stage split as
  train_rfxgb.py), held FIXED across every replicate and verified against the
  saved holdout_indices CSV. Only the training subset varies between replicates.
- Training pool that gets downsampled: the original 76.5% TRAIN split only.
- REPLICATES: --n-replicates independent permutations of the train rows, seed
  r = --base-downsample-seed + rep (rep = 0..n-1). Each replicate is drawn once
  and used NESTED/cumulative across fractions, so within a replicate
  1% subset of 2% subset of ... of 100% (each curve is smooth). Across
  replicates the permutations are independent, giving the spread for the CI.
  NOTE: with base seed 42, replicate 0 reproduces downsample_analysis.py's
  single-draw curve exactly at shared fractions (sanity check).
- Model is re-initialized from the saved best .pkl via sklearn.base.clone
  (keeps every tuned hyperparameter incl. its own random_state), fit fresh on
  each (replicate, fraction) subset. The model's internal seed is held fixed,
  so measured variability is purely from which training rows were drawn.

Outputs (in --output-dir):
    downsampling_metrics.csv   RAW, one row per (model, replicate, fraction):
                                 replicate, seed, model, fraction, n_train,
                                 r2_coef_determination, pearson_r, pearson_r2,
                                 rmse, mae
                               -> the only data output. CIs and the learning-
                                 curve plot are produced separately in R, so
                                 neither downsampling_ci.csv nor
                                 learning_curve.png is written.
    downsampling_summary.json  config + best params used + seed list + timing.

Per user spec: predictions are NOT saved (per-replicate holdout predictions
would be ~1.5 GB at 100 fractions x 50 reps and are unneeded for a metric CI).

Usage:
    python downsample_replicates.py \
        --rf-model  ../models/rfxgb/full/rf_best_*.pkl \
        --data-path ../data/processed/categorical/full_processed_categorical.csv \
        --output-dir ../models/rfxgb/full/downsampling_rfonly_1pct_50perm/ \
        --holdout-indices ../models/rfxgb/full/results/holdout_indices_*.csv \
        --n-replicates 50 --base-downsample-seed 42 \
        --frac-start 0.01 --frac-step 0.01 --frac-stop 1.00
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from glob import glob

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from utils import CAT_COLS, cast_to_categorical

METRIC_COLS = ['r2_coef_determination', 'pearson_r', 'pearson_r2', 'rmse', 'mae']


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--rf-model', default=None,
                   help='Path (glob ok) to rf_best_*.pkl. Omit or use --skip-rf to skip.')
    p.add_argument('--xgb-model', default=None,
                   help='Path (glob ok) to xgb_best_*.pkl. Omit or use --skip-xgb to skip.')
    p.add_argument('--data-path', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--target', default='avg_Rep')
    # Split reproduction — must match train_rfxgb.py defaults exactly.
    p.add_argument('--holdout-size', type=float, default=0.15)
    p.add_argument('--val-size', type=float, default=0.10)
    p.add_argument('--random-state', type=int, default=42,
                   help='Seed for the train/val/holdout split (must match training).')
    # Replicated nested downsampling.
    p.add_argument('--n-replicates', type=int, default=50,
                   help='Number of independent training-permutation replicates.')
    p.add_argument('--base-downsample-seed', type=int, default=42,
                   help='Replicate r uses permutation seed base+r (r=0..n-1).')
    # Fraction grid.
    p.add_argument('--frac-start', type=float, default=0.01)
    p.add_argument('--frac-step', type=float, default=0.01)
    p.add_argument('--frac-stop', type=float, default=1.00)
    # Optional verification against the saved holdout indices.
    p.add_argument('--holdout-indices', default=None,
                   help='CSV with original_index column from training; asserts identical holdout.')
    p.add_argument('--skip-rf', action='store_true')
    p.add_argument('--skip-xgb', action='store_true')
    return p.parse_args()


def resolve(path):
    if path and '*' in path:
        m = sorted(glob(path))
        if not m:
            raise FileNotFoundError(path)
        path = m[-1]
        log(f"Resolved: {path}")
    return path


def load_data(path, target):
    df = pd.read_csv(path)
    X = df[CAT_COLS].copy()
    y = df[target].astype(float)
    return X, y


def create_splits(X, y, holdout_size, val_size, random_state):
    """Identical two-stage split to train_rfxgb.create_splits."""
    X_pool, X_h, y_pool, y_h = train_test_split(
        X, y, test_size=holdout_size, random_state=random_state)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_pool, y_pool, test_size=val_size, random_state=random_state)
    return {'X_train': X_tr, 'X_val': X_v, 'X_holdout': X_h,
            'y_train': y_tr, 'y_val': y_v, 'y_holdout': y_h}


def verify_holdout(X_holdout, holdout_indices_path):
    """Assert the reproduced holdout matches the indices saved at training time."""
    saved = pd.read_csv(holdout_indices_path)['original_index'].values
    got = X_holdout.index.values
    if len(saved) != len(got) or not np.array_equal(saved, got):
        if set(saved.tolist()) == set(got.tolist()):
            log(f"WARNING: holdout matches as a SET but order differs "
                f"({len(got)} rows). Metrics still valid (same rows).")
        else:
            raise SystemExit(
                "ERROR: reproduced holdout does NOT match saved holdout indices. "
                f"saved n={len(saved)}, got n={len(got)}, "
                f"overlap={len(set(saved) & set(got))}. "
                "Check --holdout-size / --random-state against training.")
    else:
        log(f"Holdout verified identical to training ({len(got)} rows, exact order).")


def is_xgb(model):
    return type(model).__name__.startswith('XGB')


def fraction_grid(start, step, stop):
    n = int(round((stop - start) / step)) + 1
    fracs = [round(start + i * step, 4) for i in range(n)]
    return [f for f in fracs if f <= stop + 1e-9]


def replicate_subset_indices(n_train, fracs, seed):
    """One nested/cumulative permutation for a single replicate."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_train)
    out = {}
    for f in fracs:
        k = max(1, int(round(f * n_train)))
        k = min(k, n_train)
        out[f] = perm[:k]
    return out


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        pr = float('nan')
    else:
        pr = pearsonr(y_true, y_pred)[0]
    return {
        'r2_coef_determination': float(r2),
        'pearson_r': float(pr),
        'pearson_r2': float(pr ** 2),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
    }


def run_model(name, best_model, splits, fracs, n_reps, base_seed):
    """Re-fit a cloned model on each (replicate, fraction) nested subset."""
    log(f"\n{'='*60}\n{name.upper()}: {n_reps} replicates x {len(fracs)} fractions "
        f"= {n_reps * len(fracs)} fits\n{'='*60}")
    needs_cat = is_xgb(best_model)

    X_train, y_train = splits['X_train'], splits['y_train']
    X_holdout, y_holdout = splits['X_holdout'], splits['y_holdout']
    n_train_full = len(X_train)

    X_hold_in = cast_to_categorical(X_holdout) if needs_cat else X_holdout
    y_hold_arr = y_holdout.values

    metric_rows = []
    for rep in range(n_reps):
        seed = base_seed + rep
        subset_idx = replicate_subset_indices(n_train_full, fracs, seed)
        t_rep = time.time()
        for f in fracs:
            idx = subset_idx[f]
            X_sub = X_train.iloc[idx]
            y_sub = y_train.iloc[idx]
            X_sub_in = cast_to_categorical(X_sub) if needs_cat else X_sub

            model = clone(best_model)  # fresh, unfitted, same hyperparameters
            model.fit(X_sub_in, y_sub)
            y_pred = model.predict(X_hold_in)

            m = evaluate(y_hold_arr, y_pred)
            m.update({'replicate': rep, 'seed': seed, 'model': name,
                      'fraction': f, 'n_train': int(len(idx))})
            metric_rows.append(m)
        log(f"  replicate {rep:>2}/{n_reps} (seed={seed}) done "
            f"[{len(fracs)} fits, {time.time() - t_rep:.1f}s]")
    return metric_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log("=" * 70)
    log("LAMPRA Downsampling Analysis — replicated (multi-permutation)")
    log("=" * 70)
    log(f"Data:       {args.data_path}")
    log(f"Output:     {args.output_dir}")
    log(f"Split:      holdout={args.holdout_size}, val={args.val_size}, seed={args.random_state}")
    log(f"Replicates: {args.n_replicates}, base seed={args.base_downsample_seed}, "
        f"nested; fracs {args.frac_start}->{args.frac_stop} step {args.frac_step}")
    log("=" * 70)

    if args.skip_rf and args.skip_xgb:
        sys.exit("ERROR: both models skipped; nothing to do.")
    if args.n_replicates < 1:
        sys.exit("ERROR: --n-replicates must be >= 1.")

    X, y = load_data(args.data_path, args.target)
    log(f"Loaded {len(X)} rows, {X.shape[1]} features")

    splits = create_splits(X, y, args.holdout_size, args.val_size, args.random_state)
    log(f"Splits: train={len(splits['X_train'])} val={len(splits['X_val'])} "
        f"holdout={len(splits['X_holdout'])}")

    if args.holdout_indices:
        verify_holdout(splits['X_holdout'], args.holdout_indices)

    fracs = fraction_grid(args.frac_start, args.frac_step, args.frac_stop)
    log(f"Fractions ({len(fracs)}): {fracs[0]}..{fracs[-1]}")
    seeds = [args.base_downsample_seed + r for r in range(args.n_replicates)]

    all_metrics = []
    best_params_used = {}
    t_all = time.time()

    if not args.skip_rf:
        rf_path = resolve(args.rf_model)
        if not rf_path:
            sys.exit("ERROR: --rf-model required unless --skip-rf.")
        rf_best = joblib.load(rf_path)
        best_params_used['rf'] = {k: v for k, v in rf_best.get_params().items()}
        all_metrics += run_model('rf', rf_best, splits, fracs,
                                 args.n_replicates, args.base_downsample_seed)

    if not args.skip_xgb:
        xgb_path = resolve(args.xgb_model)
        if not xgb_path:
            sys.exit("ERROR: --xgb-model required unless --skip-xgb.")
        xgb_best = joblib.load(xgb_path)
        best_params_used['xgb'] = {
            k: v for k, v in xgb_best.get_params().items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }
        all_metrics += run_model('xgb', xgb_best, splits, fracs,
                                 args.n_replicates, args.base_downsample_seed)

    # --- Raw per-(model, replicate, fraction) metrics ---
    metrics_df = pd.DataFrame(all_metrics)[
        ['model', 'replicate', 'seed', 'fraction', 'n_train'] + METRIC_COLS
    ].sort_values(['model', 'replicate', 'fraction'])
    metrics_path = os.path.join(args.output_dir, 'downsampling_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    log(f"\nSaved: {metrics_path}  ({len(metrics_df):,} rows)")

    # NOTE: confidence intervals and the learning-curve plot are produced
    # downstream in R from downsampling_metrics.csv; this script writes only the
    # raw metrics (no downsampling_ci.csv, no learning_curve.png).

    # --- Summary JSON ---
    summary = {
        'timestamp': timestamp,
        'analysis': 'downsampling_nested_replicated_fixed_holdout',
        'data_path': os.path.abspath(args.data_path),
        'config': {
            'holdout_size': args.holdout_size,
            'val_size': args.val_size,
            'random_state': args.random_state,
            'n_replicates': args.n_replicates,
            'base_downsample_seed': args.base_downsample_seed,
            'replicate_seeds': seeds,
            'training_pool': 'original_train_split_only_76.5pct',
            'sampling_scheme': 'nested_cumulative_multi_permutation',
            'predictions_saved': False,
            'fractions': fracs,
        },
        'n_total': int(len(X)),
        'n_train_full': int(len(splits['X_train'])),
        'n_holdout': int(len(splits['X_holdout'])),
        'elapsed_sec': round(time.time() - t_all, 1),
        'best_params_used': best_params_used,
    }
    summary_path = os.path.join(args.output_dir, 'downsampling_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Saved: {summary_path}")

    log("=" * 70)
    log(f"DONE  ({summary['elapsed_sec']}s total)")
    log("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
