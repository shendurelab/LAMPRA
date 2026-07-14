"""Shared helpers for LAMPRA_ML.

The processed CSVs in this project store insert id and orientation as
integer codes (10 columns total per construct). Code <-> name maps live
here so every script (preprocess, train, evaluate) agrees on the
encoding.

Categorical level ordering is FIXED — never re-sorted by something downstream
(e.g., sorted(unique())) — so models trained at one time can be evaluated
later without categorical drift.
"""

import numpy as np
import pandas as pd

ID_LEVELS = [
    'dinucleotide_shuffled_i',
    'dinucleotide_shuffled_ii',
    'dinucleotide_shuffled_iii',
    'dinucleotide_shuffled_iv',
    'dinucleotide_shuffled_v',
    'eNMU_region_i',
    'eNMU_region_ii',
    'eNMU_region_iii',
    'eNMU_region_iv',
    'eNMU_region_v',
    'synthetic_insulator',
]
ORI_LEVELS = ['forward', 'reverse']

ID_CODE_BY_NAME = {name: i for i, name in enumerate(ID_LEVELS)}
ORI_CODE_BY_NAME = {name: i for i, name in enumerate(ORI_LEVELS)}
ID_NAME_BY_CODE = {i: name for name, i in ID_CODE_BY_NAME.items()}
ORI_NAME_BY_CODE = {i: name for name, i in ORI_CODE_BY_NAME.items()}

CAT_COLS = []
for _i in range(1, 6):
    CAT_COLS.append(f'insert_{_i}_id_code')
    CAT_COLS.append(f'insert_{_i}_ori_code')

ID_COLS = [c for c in CAT_COLS if c.endswith('_id_code')]
ORI_COLS = [c for c in CAT_COLS if c.endswith('_ori_code')]


def encode_raw_to_categorical(df_raw, target_col='avg_Rep'):
    """Convert a raw-data DataFrame into the 10-int-column categorical layout.

    Input columns required: insert_{1..5}_id, insert_{1..5}_ori (and target_col).
    Output columns: insert_{1..5}_id_code, insert_{1..5}_ori_code, target_col.
    """
    out = pd.DataFrame(index=df_raw.index)
    for i in range(1, 6):
        id_col = f'insert_{i}_id'
        ori_col = f'insert_{i}_ori'
        out[f'insert_{i}_id_code'] = df_raw[id_col].map(ID_CODE_BY_NAME).astype('Int8')
        out[f'insert_{i}_ori_code'] = df_raw[ori_col].map(ORI_CODE_BY_NAME).astype('Int8')

    if out[CAT_COLS].isna().any().any():
        bad = out[CAT_COLS].isna().any(axis=1)
        raise ValueError(
            f"Unmapped categorical levels in {bad.sum()} rows. "
            f"Check ID_LEVELS / ORI_LEVELS for missing entries."
        )
    out[CAT_COLS] = out[CAT_COLS].astype(np.int8)

    if target_col in df_raw.columns:
        out[target_col] = pd.to_numeric(df_raw[target_col], errors='coerce').astype(float)
    return out


def cast_to_categorical(X, cols=None):
    """Cast int-coded columns to pandas Categorical with stable categories.

    Required for XGBoost native categorical support (enable_categorical=True)
    and for pandas-aware downstream tools that benefit from category dtype.
    """
    cols = cols if cols is not None else [c for c in CAT_COLS if c in X.columns]
    X = X.copy()
    id_cats = list(range(len(ID_LEVELS)))
    ori_cats = list(range(len(ORI_LEVELS)))
    for c in cols:
        if c.endswith('_id_code'):
            X[c] = pd.Categorical(X[c], categories=id_cats)
        elif c.endswith('_ori_code'):
            X[c] = pd.Categorical(X[c], categories=ori_cats)
    return X


def make_iiiF_mask(df):
    """Boolean mask: insert_5 is eNMU_region_iii AND forward.

    Works for raw (insert_5_id) or encoded (insert_5_id_code) DataFrames.
    """
    if 'insert_5_id' in df.columns and 'insert_5_ori' in df.columns:
        return ((df['insert_5_id'] == 'eNMU_region_iii') &
                (df['insert_5_ori'] == 'forward')).values
    if 'insert_5_id_code' in df.columns and 'insert_5_ori_code' in df.columns:
        iii = ID_CODE_BY_NAME['eNMU_region_iii']
        fwd = ORI_CODE_BY_NAME['forward']
        return ((df['insert_5_id_code'] == iii) &
                (df['insert_5_ori_code'] == fwd)).values
    raise ValueError("DataFrame missing both insert_5_id and insert_5_id_code")
