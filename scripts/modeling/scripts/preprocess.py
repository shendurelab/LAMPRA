#!/usr/bin/env python3
"""LAMPRA Data Preprocessing — Categorical Encoding.

Reads raw MPRA data and emits a single 10-column integer-coded CSV per
dataset (full + iiiF_only + except_iiiF). Both the linear and rfxgb
training scripts consume the same processed CSVs:
- linear:  one-hot expands these columns inside its sklearn pipeline.
- rfxgb:   feeds int codes directly (RF) or via pd.Categorical (XGB).

Output columns:
    insert_1_id_code, insert_1_ori_code,
    insert_2_id_code, insert_2_ori_code,
    insert_3_id_code, insert_3_ori_code,
    insert_4_id_code, insert_4_ori_code,
    insert_5_id_code, insert_5_ori_code,
    avg_Rep

Usage:
    python preprocess.py \
        --input ../data/raw/v2_longMPRA_scores_with_orientation_20260127.txt \
        --output-dir ../data/processed/categorical/

SGE submission:
    qsub submit_preprocess.sh
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    CAT_COLS,
    ID_LEVELS,
    ORI_LEVELS,
    ID_CODE_BY_NAME,
    ORI_CODE_BY_NAME,
    encode_raw_to_categorical,
    make_iiiF_mask,
)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess MPRA data with categorical (int-coded) encoding")
    p.add_argument('--input', '-i', type=str, required=True)
    p.add_argument('--output-dir', '-o', type=str, required=True)
    p.add_argument('--target', type=str, default='avg_Rep')
    p.add_argument('--holdout-size', type=float, default=0.15)
    p.add_argument('--val-size', type=float, default=0.10)
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()


def load_raw_data(filepath):
    print(f"Loading raw data from: {filepath}")
    df = pd.read_table(filepath)
    print(f"  Shape: {df.shape}")
    return df


def clean_data(df, target_col):
    keep_cols = [target_col]
    for pos in range(1, 6):
        keep_cols.extend([f'insert_{pos}_id', f'insert_{pos}_ori'])

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns: {missing}")

    out = df[keep_cols].copy()
    out[target_col] = pd.to_numeric(out[target_col], errors='coerce')
    n_missing = out[target_col].isna().sum()
    if n_missing > 0:
        print(f"WARNING: Dropping {n_missing} rows with missing/invalid target")
        out = out.dropna(subset=[target_col]).reset_index(drop=True)
    return out


def create_split_indices(n, holdout_size, val_size, random_state):
    idx = np.arange(n)
    pool, holdout = train_test_split(idx, test_size=holdout_size, random_state=random_state)
    train, val = train_test_split(pool, test_size=val_size, random_state=random_state)
    return {'train': train, 'val': val, 'holdout': holdout}


def write_dataset(df_raw, name, out_dir, args, timestamp):
    """Encode + split + save one dataset."""
    encoded = encode_raw_to_categorical(df_raw, target_col=args.target)
    print(f"  {name}: {encoded.shape[0]} rows, {encoded.shape[1]-1} categorical features")

    csv_path = os.path.join(out_dir, f'{name}.csv')
    encoded.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    splits = create_split_indices(len(encoded), args.holdout_size, args.val_size, args.random_state)
    splits_path = os.path.join(out_dir, f'{name}_splits.json')
    with open(splits_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in splits.items()}, f)
    print(f"    Saved: {splits_path}")

    meta = {
        'dataset': name,
        'source_file': os.path.basename(args.input),
        'n_samples': int(len(encoded)),
        'n_features': int(encoded.shape[1] - 1),
        'encoding': 'integer_categorical',
        'feature_columns': CAT_COLS,
        'id_levels': ID_LEVELS,
        'ori_levels': ORI_LEVELS,
        'id_code_by_name': ID_CODE_BY_NAME,
        'ori_code_by_name': ORI_CODE_BY_NAME,
        'target_col': args.target,
        'target_stats': {
            'mean': float(encoded[args.target].mean()),
            'std': float(encoded[args.target].std()),
            'min': float(encoded[args.target].min()),
            'max': float(encoded[args.target].max()),
        },
        'splits': {
            'holdout_size': args.holdout_size,
            'val_size': args.val_size,
            'random_state': args.random_state,
            'train_n': int(len(splits['train'])),
            'val_n': int(len(splits['val'])),
            'holdout_n': int(len(splits['holdout'])),
        },
        'columns': list(encoded.columns),
        'created_at': timestamp,
    }
    meta_path = os.path.join(out_dir, f'{name}_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"    Saved: {meta_path}")


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("LAMPRA DATA PREPROCESSING — Categorical (10 int columns)")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Holdout: {args.holdout_size}, val (of pool): {args.val_size}, seed: {args.random_state}")
    print("=" * 70)

    raw = load_raw_data(args.input)
    clean = clean_data(raw, args.target)

    print("\nUnique insert types per slot (raw):")
    for pos in range(1, 6):
        u = sorted(clean[f'insert_{pos}_id'].unique())
        print(f"  Position {pos}: {u}")
    print("\nUnique orientations per slot (raw):")
    for pos in range(1, 6):
        u = sorted(clean[f'insert_{pos}_ori'].unique())
        print(f"  Position {pos}: {u}")

    print("\n--- FULL ---")
    write_dataset(clean, 'full_processed_categorical', args.output_dir, args, timestamp)

    mask_iiiF = make_iiiF_mask(clean)
    print(f"\n--- iiiF_only ({mask_iiiF.sum()} rows) ---")
    write_dataset(clean[mask_iiiF].reset_index(drop=True),
                  'subset_iiiF_only', args.output_dir, args, timestamp)

    print(f"\n--- except_iiiF ({(~mask_iiiF).sum()} rows) ---")
    write_dataset(clean[~mask_iiiF].reset_index(drop=True),
                  'subset_except_iiiF', args.output_dir, args, timestamp)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    for f in sorted(os.listdir(args.output_dir)):
        size_kb = os.path.getsize(os.path.join(args.output_dir, f)) / 1024
        print(f"  {f}: {size_kb:.1f} KB")


if __name__ == '__main__':
    sys.exit(main())
