"""
Baseline: Logistic Regression on tabular features only (54 cols).

This is the non-ML baseline reference from the proposal — what a naive
prediction tool would produce. We standardize numeric features, fit L2 LR,
report AUROC / precision / recall / Brier / confusion matrix at 0.5,
and save the model + metrics.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

print("Loading tabular data ...")
X_train = pd.read_parquet(DATA / "X_train_clean.parquet")
X_test = pd.read_parquet(DATA / "X_test_clean.parquet")
y_train = pd.read_parquet(DATA / "y_train.parquet")["label"].values.astype(int)
y_test = pd.read_parquet(DATA / "y_test.parquet")["label"].values.astype(int)
print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

feat_cols = list(X_train.columns)

pipe = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=True)),
        (
            "lr",
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=2000,
                random_state=42,
            ),
        ),
    ]
)

print("Training logistic regression ...")
t0 = time.time()
pipe.fit(X_train.values, y_train)
fit_secs = time.time() - t0
print(f"  fit time: {fit_secs:.1f}s")

print("Predicting ...")
proba_train = pipe.predict_proba(X_train.values)[:, 1]
proba_test = pipe.predict_proba(X_test.values)[:, 1]
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


pred_train = (proba_train >= 0.5).astype(int)
m_train = metrics(y_train, proba_train, pred_train)
m_test = metrics(y_test, proba_test, pred_test)

print("\n=== Train ===")
print(f"AUROC={m_train['auroc']:.4f}  AP={m_train['ap']:.4f}  Brier={m_train['brier']:.4f}")
print("\n=== Test ===")
print(f"AUROC={m_test['auroc']:.4f}  AP={m_test['ap']:.4f}  Brier={m_test['brier']:.4f}")
print(f"Accuracy={m_test['accuracy']:.4f}")
print(f"Confusion matrix:\n{np.array(m_test['confusion'])}")

# top-magnitude coefficients (post-scaling, so directly comparable)
lr = pipe.named_steps["lr"]
coefs = pd.DataFrame(
    {"feature": feat_cols, "coef": lr.coef_[0]}
).sort_values("coef", key=lambda s: s.abs(), ascending=False)
print("\nTop 15 coefficients (by magnitude):")
print(coefs.head(15).to_string(index=False))

# save artifacts
joblib.dump(pipe, MODELS / "lr_baseline.joblib")
np.save(MODELS / "lr_baseline_proba_test.npy", proba_test)

results = {
    "model": "logistic_regression_tabular",
    "n_features": X_train.shape[1],
    "fit_time_seconds": fit_secs,
    "train": {k: v for k, v in m_train.items() if k not in {"report", "confusion"}},
    "test": {k: v for k, v in m_test.items() if k != "report"},
    "top_coefficients": coefs.head(25).to_dict(orient="records"),
}
with open(REPORTS / "metrics_baseline_lr.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved: {MODELS / 'lr_baseline.joblib'}")
print(f"Saved: {REPORTS / 'metrics_baseline_lr.json'}")
