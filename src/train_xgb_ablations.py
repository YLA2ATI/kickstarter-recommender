"""
Ablation runs:
  A) XGBoost on tabular-only (52 cols) -> measures non-linear lift vs LR baseline.
  B) XGBoost on full features minus Google Trends columns
     (trend_score, has_trend) -> isolates Trends contribution.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def metrics(y_true, proba, pred):
    return {
        "auroc": float(roc_auc_score(y_true, proba)),
        "ap": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "logloss": float(log_loss(y_true, proba)),
        "accuracy": float((pred == y_true).mean()),
        "confusion": confusion_matrix(y_true, pred).tolist(),
        "report": classification_report(y_true, pred, output_dict=True),
    }


def split_train_val(y, frac=0.10, seed=42):
    idx_tr, idx_val = train_test_split(
        np.arange(len(y)), test_size=frac, stratify=y, random_state=seed
    )
    return idx_tr, idx_val


y_train = pd.read_parquet(DATA / "y_train.parquet")["label"].values.astype(int)
y_test = pd.read_parquet(DATA / "y_test.parquet")["label"].values.astype(int)
idx_tr, idx_val = split_train_val(y_train)
y_tr = y_train[idx_tr]
y_val = y_train[idx_val]


# ============================================================================
# A) XGBoost tabular only
# ============================================================================
print("=" * 70)
print("ABLATION A: XGBoost on tabular only (52 features)")
print("=" * 70)

X_train_tab = pd.read_parquet(DATA / "X_train_clean.parquet")
X_test_tab = pd.read_parquet(DATA / "X_test_clean.parquet")
print(f"  X_train_tab={X_train_tab.shape}  X_test_tab={X_test_tab.shape}")

X_tr_tab = X_train_tab.iloc[idx_tr].values
X_val_tab = X_train_tab.iloc[idx_val].values
X_test_tab_arr = X_test_tab.values

dtrain_tab = xgb.QuantileDMatrix(X_tr_tab, label=y_tr, max_bin=256)
dval_tab = xgb.QuantileDMatrix(X_val_tab, label=y_val, ref=dtrain_tab, max_bin=256)
dtest_tab = xgb.QuantileDMatrix(X_test_tab_arr, label=y_test, ref=dtrain_tab, max_bin=256)

# tabular is small (52 cols) — can afford larger trees, more bins, higher colsample
params_tab = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_bin": 256,
    "verbosity": 0,
    "seed": 42,
}

t0 = time.time()
booster_tab = xgb.train(
    params_tab,
    dtrain_tab,
    num_boost_round=2000,
    evals=[(dtrain_tab, "train"), (dval_tab, "val")],
    early_stopping_rounds=50,
    verbose_eval=100,
)
fit_secs_tab = time.time() - t0
print(f"  fit time: {fit_secs_tab:.1f}s  best_iter={booster_tab.best_iteration}")

proba_test_tab = booster_tab.predict(
    dtest_tab, iteration_range=(0, booster_tab.best_iteration + 1)
)
pred_test_tab = (proba_test_tab >= 0.5).astype(int)
m_test_tab = metrics(y_test, proba_test_tab, pred_test_tab)
print(f"  TEST AUROC={m_test_tab['auroc']:.4f}  AP={m_test_tab['ap']:.4f}  "
      f"Brier={m_test_tab['brier']:.4f}")

booster_tab.save_model(MODELS / "xgb_tabular.json")
np.save(MODELS / "xgb_tabular_proba_test.npy", proba_test_tab)

results_tab = {
    "model": "xgboost_tabular",
    "n_features": int(X_train_tab.shape[1]),
    "best_iteration": int(booster_tab.best_iteration),
    "fit_time_seconds": fit_secs_tab,
    "params": params_tab,
    "test": {k: v for k, v in m_test_tab.items() if k != "report"},
}
with open(REPORTS / "metrics_xgb_tabular.json", "w") as f:
    json.dump(results_tab, f, indent=2, default=float)


# ============================================================================
# B) XGBoost full minus Trends (drop trend_score + has_trend)
# ============================================================================
print("\n" + "=" * 70)
print("ABLATION B: XGBoost full minus Trends (drop trend_score + has_trend)")
print("=" * 70)

trend_cols = ["trend_score", "has_trend"]
trend_idx = [X_train_tab.columns.get_loc(c) for c in trend_cols]
print(f"  Trend col indices in tabular: {dict(zip(trend_cols, trend_idx))}")

X_train_full = sp.load_npz(DATA / "X_train_full_clean.npz")
X_test_full = sp.load_npz(DATA / "X_test_full_clean.npz")
print(f"  X_train_full={X_train_full.shape}  X_test_full={X_test_full.shape}")

keep_mask = np.ones(X_train_full.shape[1], dtype=bool)
for i in trend_idx:
    keep_mask[i] = False
keep_indices = np.where(keep_mask)[0]
X_train_no_trend = X_train_full.tocsc()[:, keep_indices].tocsr()
X_test_no_trend = X_test_full.tocsc()[:, keep_indices].tocsr()
print(f"  no-trend shapes: train={X_train_no_trend.shape}  test={X_test_no_trend.shape}")

X_tr_nt = X_train_no_trend[idx_tr]
X_val_nt = X_train_no_trend[idx_val]

dtrain_nt = xgb.QuantileDMatrix(X_tr_nt, label=y_tr, max_bin=64)
dval_nt = xgb.QuantileDMatrix(X_val_nt, label=y_val, ref=dtrain_nt, max_bin=64)
dtest_nt = xgb.QuantileDMatrix(X_test_no_trend, label=y_test, ref=dtrain_nt, max_bin=64)

# match the full-model params for fair comparison
params_nt = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.3,
    "reg_lambda": 1.0,
    "max_bin": 64,
    "verbosity": 0,
    "seed": 42,
}

t0 = time.time()
booster_nt = xgb.train(
    params_nt,
    dtrain_nt,
    num_boost_round=2000,
    evals=[(dtrain_nt, "train"), (dval_nt, "val")],
    early_stopping_rounds=50,
    verbose_eval=100,
)
fit_secs_nt = time.time() - t0
print(f"  fit time: {fit_secs_nt:.1f}s  best_iter={booster_nt.best_iteration}")

proba_test_nt = booster_nt.predict(
    dtest_nt, iteration_range=(0, booster_nt.best_iteration + 1)
)
pred_test_nt = (proba_test_nt >= 0.5).astype(int)
m_test_nt = metrics(y_test, proba_test_nt, pred_test_nt)
print(f"  TEST AUROC={m_test_nt['auroc']:.4f}  AP={m_test_nt['ap']:.4f}  "
      f"Brier={m_test_nt['brier']:.4f}")

booster_nt.save_model(MODELS / "xgb_full_no_trends.json")
np.save(MODELS / "xgb_full_no_trends_proba_test.npy", proba_test_nt)

results_nt = {
    "model": "xgboost_full_no_trends",
    "n_features": int(X_train_no_trend.shape[1]),
    "best_iteration": int(booster_nt.best_iteration),
    "fit_time_seconds": fit_secs_nt,
    "params": params_nt,
    "test": {k: v for k, v in m_test_nt.items() if k != "report"},
}
with open(REPORTS / "metrics_xgb_full_no_trends.json", "w") as f:
    json.dump(results_nt, f, indent=2, default=float)

print("\nDone.")
