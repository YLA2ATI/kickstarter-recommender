"""Audit X_train.parquet columns for outcome leakage."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

X_train = pd.read_parquet(DATA / "X_train.parquet")
y_train = pd.read_parquet(DATA / "y_train.parquet")["label"].values

print(f"X_train shape: {X_train.shape}")
print(f"\nAll {X_train.shape[1]} columns:")
for c in X_train.columns:
    print(f"  {c}")

# correlation with label for binary/numeric cols (quick leak detector)
print("\n--- Univariate correlation with label (top 25 by abs corr) ---")
corrs = X_train.corrwith(pd.Series(y_train, index=X_train.index)).abs().sort_values(ascending=False)
print(corrs.head(25).to_string())

# also check the master clean table for what got dropped vs kept
print("\n--- kickstarter_clean.parquet columns ---")
clean = pd.read_parquet(DATA / "kickstarter_clean.parquet")
print(f"  shape: {clean.shape}")
for c in clean.columns:
    print(f"  {c}")
