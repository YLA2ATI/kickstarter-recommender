"""
XGBoost on the full feature matrix (tabular + word TF-IDF + char TF-IDF).

We carve a held-out validation slice (10%) from train for early stopping
so we never touch test during selection. Training runs on GPU.
"""
import json
import time
from pathlib import Path

import joblib
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
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

print("Loading clean full sparse matrices ...")
X_train = sp.load_npz(DATA / "X_train_full_clean.npz")
X_test = sp.load_npz(DATA / "X_test_full_clean.npz")
y_train = pd.read_parquet(DATA / "y_train.parquet")["label"].values.astype(int)
y_test = pd.read_parquet(DATA / "y_test.parquet")["label"].values.astype(int)
print(f"  X_train={X_train.shape} nnz={X_train.nnz:,}")
print(f"  X_test ={X_test.shape}  nnz={X_test.nnz:,}")
print(f"  y_train pos={y_train.mean():.4f}  y_test pos={y_test.mean():.4f}")

print("Carving 10% validation slice from train ...")
idx_tr, idx_val = train_test_split(
    np.arange(len(y_train)),
    test_size=0.10,
    stratify=y_train,
    random_state=42,
)
X_tr = X_train[idx_tr]
X_val = X_train[idx_val]
y_tr = y_train[idx_tr]
y_val = y_train[idx_val]
print(f"  tr={X_tr.shape}  val={X_val.shape}")

print("Building QuantileDMatrices (memory-efficient for high-dim sparse) ...")
t0 = time.time()
dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, max_bin=64)
dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain, max_bin=64)
dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain, max_bin=64)
print(f"  QuantileDMatrix build: {time.time() - t0:.1f}s")

params = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.3,
    "reg_lambda": 1.0,
    "max_bin": 64,
    "verbosity": 1,
    "seed": 42,
}

print(f"Training XGBoost (GPU) ...")
print(f"  params: {params}")
t0 = time.time()
booster = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=25,
)
fit_secs = time.time() - t0
print(f"  fit time: {fit_secs:.1f}s")
print(f"  best_iteration: {booster.best_iteration}")
print(f"  best_score (val AUC): {booster.best_score}")

print("Predicting ...")
proba_val = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
proba_test = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
pred_test = (proba_test >= 0.5).astype(int)


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


m_val = metrics(y_val, proba_val, (proba_val >= 0.5).astype(int))
m_test = metrics(y_test, proba_test, pred_test)

print("\n=== Val (held out from train) ===")
print(f"AUROC={m_val['auroc']:.4f}  AP={m_val['ap']:.4f}  Brier={m_val['brier']:.4f}")
print("\n=== Test (post-2022, held out) ===")
print(f"AUROC={m_test['auroc']:.4f}  AP={m_test['ap']:.4f}  Brier={m_test['brier']:.4f}")
print(f"Accuracy={m_test['accuracy']:.4f}")
print(f"Confusion matrix:\n{np.array(m_test['confusion'])}")

# top features by gain
print("\nTop 25 features by total gain:")
feat_names = np.load(DATA / "feature_names_full_clean.npy", allow_pickle=True)
score = booster.get_score(importance_type="total_gain")
gain_df = (
    pd.DataFrame({"feat_idx": list(score.keys()), "gain": list(score.values())})
    .assign(feat_idx=lambda d: d["feat_idx"].str.lstrip("f").astype(int))
    .assign(name=lambda d: feat_names[d["feat_idx"].values])
    .sort_values("gain", ascending=False)
)
print(gain_df.head(25)[["name", "gain"]].to_string(index=False))

# save artifacts
booster.save_model(MODELS / "xgb_full.json")
np.save(MODELS / "xgb_full_proba_test.npy", proba_test)
gain_df.head(200).to_csv(REPORTS / "xgb_full_top_gain_features.csv", index=False)

results = {
    "model": "xgboost_full",
    "n_features": int(X_train.shape[1]),
    "params": params,
    "best_iteration": int(booster.best_iteration),
    "best_val_auc": float(booster.best_score),
    "fit_time_seconds": fit_secs,
    "val": {k: v for k, v in m_val.items() if k != "report"},
    "test": {k: v for k, v in m_test.items() if k != "report"},
}
with open(REPORTS / "metrics_xgb_full.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nSaved: {MODELS / 'xgb_full.json'}")
print(f"Saved: {REPORTS / 'metrics_xgb_full.json'}")
