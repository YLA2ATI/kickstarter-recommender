"""
Remove outcome-leaking / non-actionable columns and rewrite clean artifacts.

Leaks dropped:
  is_spotlight  - corr=1.0 with label; only set for SUCCESSFUL campaigns post-outcome.
  is_staff_pick - Kickstarter curatorial decision, not a creator-controllable
                  pre-launch input. For a pre-launch recommendation tool this
                  is unrealistic input.

Strategy:
  - parquet: drop columns in pandas
  - npz full sparse: tabular features occupy columns [0:54] in the same order as
    X_train.parquet; we delete the indices of dropped cols and keep TF-IDF tail
    intact.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

LEAKS = ["is_spotlight", "is_staff_pick"]

X_train = pd.read_parquet(DATA / "X_train.parquet")
X_test = pd.read_parquet(DATA / "X_test.parquet")
print(f"original tabular cols: {X_train.shape[1]}")

leak_idx = [X_train.columns.get_loc(c) for c in LEAKS]
print(f"leak column indices in tabular: {dict(zip(LEAKS, leak_idx))}")

X_train_clean = X_train.drop(columns=LEAKS)
X_test_clean = X_test.drop(columns=LEAKS)
print(f"clean tabular cols: {X_train_clean.shape[1]}")

X_train_clean.to_parquet(DATA / "X_train_clean.parquet")
X_test_clean.to_parquet(DATA / "X_test_clean.parquet")
print(f"wrote {DATA / 'X_train_clean.parquet'}, {DATA / 'X_test_clean.parquet'}")

# also produce cleaned full sparse matrices
N_TAB_ORIG = X_train.shape[1]  # 54
print(f"\nLoading full sparse matrices ...")
X_train_full = sp.load_npz(DATA / "X_train_full.npz").tocsc()
X_test_full = sp.load_npz(DATA / "X_test_full.npz").tocsc()
print(f"  X_train_full: {X_train_full.shape}  X_test_full: {X_test_full.shape}")
assert X_train_full.shape[1] >= N_TAB_ORIG

keep_mask = np.ones(X_train_full.shape[1], dtype=bool)
for i in leak_idx:
    keep_mask[i] = False
keep_indices = np.where(keep_mask)[0]
print(f"  keeping {keep_mask.sum():,} of {len(keep_mask):,} columns")

X_train_full_clean = X_train_full[:, keep_indices].tocsr()
X_test_full_clean = X_test_full[:, keep_indices].tocsr()
print(f"  cleaned shapes: train={X_train_full_clean.shape}  test={X_test_full_clean.shape}")

sp.save_npz(DATA / "X_train_full_clean.npz", X_train_full_clean)
sp.save_npz(DATA / "X_test_full_clean.npz", X_test_full_clean)
print(f"wrote cleaned full sparse matrices")

# also build a feature-name index for the cleaned full matrix (tabular + tfidf)
import joblib

wv = joblib.load(DATA / "tfidf_word_vectorizer.joblib")
cv = joblib.load(DATA / "tfidf_char_vectorizer.joblib")
tab_names = list(X_train_clean.columns)
word_names = [f"w::{w}" for w in wv.get_feature_names_out()]
char_names = [f"c::{c}" for c in cv.get_feature_names_out()]
all_names = tab_names + word_names + char_names
assert len(all_names) == X_train_full_clean.shape[1], (
    f"feature-name count {len(all_names)} != cols {X_train_full_clean.shape[1]}"
)
np.save(DATA / "feature_names_full_clean.npy", np.array(all_names, dtype=object))
print(f"wrote feature-name index ({len(all_names):,} names)")
