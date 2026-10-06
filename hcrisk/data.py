"""Loading the 7 Home Credit tables with low memory.

The raw CSVs are ~2.5 GB. Read naively, pandas uses ~10 GB (float64 / object). We
read each table once, downcast numbers (float64 -> float32, int64 -> smallest int),
turn text columns into categories, and cache the result as Parquet, so later runs
load in seconds and use ~3x less RAM.

Table relationships (keys):

    application_{train,test}   SK_ID_CURR (1 row per applicant; TARGET only in train)
    ├── bureau                 SK_ID_CURR -> SK_ID_BUREAU (loans at other institutions)
    │   └── bureau_balance     SK_ID_BUREAU (monthly status of those loans)
    └── previous_application   SK_ID_CURR -> SK_ID_PREV (earlier Home Credit applications)
        ├── POS_CASH_balance       SK_ID_PREV (monthly POS / cash loan snapshots)
        ├── installments_payments  SK_ID_PREV (every scheduled vs actual payment)
        └── credit_card_balance    SK_ID_PREV (monthly credit card snapshots)
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, ints -> smallest int, text -> category. Keys stay int32/int64."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            df[col] = s.astype("category")
    return df


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / 1e6


def load_table(name: str, raw_dir: str | Path, cache_dir: Optional[str | Path] = None,
               ids: Optional[set] = None, id_col: str = "SK_ID_CURR") -> pd.DataFrame:
    """Load one table (from the Parquet cache if present). `ids` keeps only those applicants
    (used by --sample to run on a laptop with little RAM)."""
    raw_dir = Path(raw_dir)
    cache = Path(cache_dir) / f"{name}.parquet" if cache_dir else None
    if cache is not None and cache.exists():
        df = pd.read_parquet(cache)
    else:
        path = raw_dir / TABLES[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - put the Kaggle CSVs in {raw_dir}")
        # read text columns straight into categories (bureau_balance has 27M rows)
        head = pd.read_csv(path, nrows=2000)
        dtypes = {c: "category" for c in head.columns
                  if pd.api.types.is_object_dtype(head[c]) or pd.api.types.is_string_dtype(head[c])}
        df = downcast(pd.read_csv(path, dtype=dtypes, low_memory=False))
        del head
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
    if ids is not None and id_col in df.columns:
        df = df[df[id_col].isin(ids)].reset_index(drop=True)
    gc.collect()
    return df


def sample_ids(raw_dir: str | Path, frac: float, seed: int = 42, cache_dir=None) -> Optional[set]:
    """A stratified sample of applicant ids (train keeps the default rate) plus all of test."""
    if frac >= 1.0:
        return None
    train = load_table("application_train", raw_dir, cache_dir)[["SK_ID_CURR", "TARGET"]]
    test = load_table("application_test", raw_dir, cache_dir)[["SK_ID_CURR"]]
    picked = train.groupby("TARGET", group_keys=False).sample(frac=frac, random_state=seed)
    return set(picked["SK_ID_CURR"]) | set(test["SK_ID_CURR"].sample(frac=frac, random_state=seed))
