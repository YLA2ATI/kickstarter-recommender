"""Sanity check: load all artifacts and confirm shapes + GPU availability."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

X_train = pd.read_parquet(DATA / "X_train.parquet")
X_test = pd.read_parquet(DATA / "X_test.parquet")
y_train = pd.read_parquet(DATA / "y_train.parquet")["label"].values
y_test = pd.read_parquet(DATA / "y_test.parquet")["label"].values

X_train_full = sp.load_npz(DATA / "X_train_full.npz")
X_test_full = sp.load_npz(DATA / "X_test_full.npz")

clean = pd.read_parquet(DATA / "kickstarter_clean.parquet")

print("=== Tabular ===")
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}  y_test: {y_test.shape}")
print(f"y_train pos rate: {y_train.mean():.4f}  y_test pos rate: {y_test.mean():.4f}")
print(f"tabular cols: {list(X_train.columns)[:10]} ...")
print()

print("=== Full sparse ===")
print(f"X_train_full: {X_train_full.shape}  nnz={X_train_full.nnz:,}")
print(f"X_test_full:  {X_test_full.shape}  nnz={X_test_full.nnz:,}")
print(f"density train: {X_train_full.nnz / (X_train_full.shape[0]*X_train_full.shape[1]):.6e}")
print()

print("=== Master clean ===")
print(f"rows: {len(clean):,}  cols: {clean.shape[1]}")
print(f"launch_year range: {clean['launch_year'].min()} - {clean['launch_year'].max()}")
print(f"label rate: {clean['label'].mean():.4f}")
if "is_lebanon_related" in clean.columns:
    print(f"lebanon-related: {clean['is_lebanon_related'].sum():,}")
print()

print("=== XGBoost / GPU ===")
print(f"xgboost version: {xgb.__version__}")
try:
    booster = xgb.train(
        {"device": "cuda", "objective": "binary:logistic", "max_depth": 2, "verbosity": 0},
        xgb.DMatrix(np.random.rand(100, 5), label=np.random.randint(0, 2, 100)),
        num_boost_round=2,
    )
    print("GPU train smoke test: OK")
except Exception as e:
    print(f"GPU train smoke test FAILED: {e}")

with open(DATA / "dataset_summary.json") as f:
    summary = json.load(f)
print()
print("=== dataset_summary.json ===")
print(json.dumps(summary, indent=2))
