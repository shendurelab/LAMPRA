#!/usr/bin/env python3
"""Model Comparison — Merge single-split results into combined_df.csv.

Collects training-summary JSONs and holdout-evaluation JSONs from each
family/subset model directory and writes a single combined_df.csv that
serves as the input for downstream R analyses.

Single-split only (no cross-validation); no top-k candidate rows are emitted.

Output:
    data/combined_df.csv

Usage:
    python model_comparison_merge.py
"""

import json
import os
from glob import glob

import numpy as np
import pandas as pd

# Repo root = parent of this notebooks/ directory (portable; no hardcoded path).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

FAMILIES = ['linear', 'linear-interactions', 'rfxgb']
SUBSETS = ['full', 'iiiF_only', 'except_iiiF']


def load_summaries(models_dir):
    rows = []
    for family in FAMILIES:
        for subset in SUBSETS:
            results_dir = os.path.join(models_dir, family, subset, 'results')
            if not os.path.exists(results_dir):
                continue
            for json_path in sorted(glob(os.path.join(results_dir, '*_summary_*.json'))):
                base = os.path.basename(json_path)
                if base.startswith('training_summary'):
                    continue
                with open(json_path) as f:
                    summary = json.load(f)
                model_name = summary.get('model_name', base.split('_summary')[0])
                metrics = summary.get('metrics', {})
                rows.append({
                    'family': family,
                    'subset': subset,
                    'model': model_name,
                    'rank': 'best',
                    'training_method': 'single_split',
                    'label': f'{family}/{subset}/{model_name}',
                    'best_params': str(summary.get('best_params', {})),
                    'train_r2': metrics.get('train_r2'),
                    'val_r2': metrics.get('val_r2'),
                    'train_rmse': metrics.get('train_rmse'),
                    'val_rmse': metrics.get('val_rmse'),
                    'overfit_r2_gap': metrics.get('overfit_r2_gap'),
                    'overfit_rmse_ratio': metrics.get('overfit_rmse_ratio'),
                })
    return pd.DataFrame(rows)


def load_holdout_evaluations(models_dir):
    evals = []
    for json_path in glob(os.path.join(models_dir, '**/*_holdout_eval.json'), recursive=True):
        with open(json_path) as f:
            data = json.load(f)
        rel = os.path.relpath(json_path, models_dir).split(os.sep)
        family = rel[0] if rel else 'unknown'
        subset = rel[1] if len(rel) > 1 else 'unknown'

        base = os.path.basename(json_path).replace('_holdout_eval.json', '')
        if '_best_' in base:
            model_name = base.split('_best_')[0]
        else:
            model_name = base
        rank = 'best'

        h = data.get('results', {}).get('holdout', {})
        evals.append({
            'family': family,
            'subset': subset,
            'model': model_name,
            'rank': rank,
            'holdout_r2': h.get('r2'),
            'holdout_rmse': h.get('rmse'),
            'holdout_mae': h.get('mae'),
            'holdout_n': h.get('n_samples'),
            'overfit_r2_gap_holdout': data.get('overfit_r2_gap'),
            'generalization_gap': data.get('generalization_gap'),
            'json_path': json_path,
        })
    return pd.DataFrame(evals)


def merge(summary_df, holdout_df):
    out = summary_df.copy()
    if len(holdout_df):
        cols = ['family', 'subset', 'model', 'rank',
                'holdout_r2', 'holdout_rmse', 'holdout_mae', 'generalization_gap']
        out = out.merge(holdout_df[cols], on=['family', 'subset', 'model', 'rank'], how='left')
    else:
        out['holdout_r2'] = np.nan
        out['holdout_rmse'] = np.nan
        out['holdout_mae'] = np.nan
        out['generalization_gap'] = np.nan
    out['training_method'] = out['training_method'].fillna('single_split')
    return out


def main():
    print(f'PROJECT_ROOT: {PROJECT_ROOT}\n')

    summary_df = load_summaries(MODELS_DIR)
    print(f'1. {len(summary_df)} single-split summaries loaded')

    holdout_df = load_holdout_evaluations(MODELS_DIR)
    print(f'2. {len(holdout_df)} holdout evaluations loaded')

    combined = merge(summary_df, holdout_df)
    print(f'\ncombined_df: {len(combined)} rows, {len(combined.columns)} cols')

    out_path = os.path.join(DATA_DIR, 'combined_df.csv')
    combined.to_csv(out_path, index=False)
    print(f'\nSaved {len(combined)} rows to {out_path}')

    # Quick summary
    print('\n--- Best per subset (by val_r2) ---')
    best = combined[combined['rank'] == 'best']
    for subset in sorted(best['subset'].unique()):
        sub = best[best['subset'] == subset]
        if not len(sub):
            continue
        bidx = sub['val_r2'].idxmax()
        b = sub.loc[bidx]
        h = f', holdout R\u00b2={b["holdout_r2"]:.4f}' if pd.notna(b.get('holdout_r2')) else ''
        print(f'  {subset}: {b["family"]}/{b["model"]} \u2014 val R\u00b2={b["val_r2"]:.4f}{h}')


if __name__ == '__main__':
    main()
