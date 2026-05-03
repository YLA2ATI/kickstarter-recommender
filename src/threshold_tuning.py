"""
Threshold tuning for the recommender model (XGBoost tabular).

Surfaces:
- precision/recall/F1 across all thresholds
- best threshold for several creator-relevant objectives
- PR curve and threshold-vs-metric plot
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

proba = np.load(MODELS / "xgb_tabular_proba_test.npy")
y = pd.read_parquet(DATA / "y_test.parquet")["label"].values.astype(int)

# sweep
thresholds = np.linspace(0.05, 0.95, 91)
rows = []
for t in thresholds:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p = precision_score(y, pred, zero_division=0)
    r = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    rows.append({
        "threshold": float(t),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "predicted_positive_rate": float(pred.mean()),
        "accuracy": float((pred == y).mean()),
    })
df = pd.DataFrame(rows)
df.to_csv(REPORTS / "threshold_sweep.csv", index=False)

# objective-specific picks
def best_threshold(criterion: callable, name: str) -> dict:
    best = max(rows, key=criterion)
    return {"objective": name, **best}

picks = []
picks.append(best_threshold(lambda r: r["f1"], "max_f1"))
picks.append(best_threshold(lambda r: r["accuracy"], "max_accuracy"))
# strict screener: ≥0.85 precision (model only flags very-likely-success), max recall under that
strict = [r for r in rows if r["precision"] >= 0.85]
if strict:
    picks.append(max(strict, key=lambda r: r["recall"]) | {"objective": "precision_>=0.85_max_recall"})
# safety net: ≥0.85 recall (catch most successful), max precision under that
safety = [r for r in rows if r["recall"] >= 0.85]
if safety:
    picks.append(max(safety, key=lambda r: r["precision"]) | {"objective": "recall_>=0.85_max_precision"})

# show picks
print("=== Best threshold per objective ===")
keep_cols = ["objective", "threshold", "precision", "recall", "f1", "accuracy",
             "predicted_positive_rate"]
picks_df = pd.DataFrame(picks)[keep_cols]
print(picks_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
picks_df.to_csv(REPORTS / "threshold_best_picks.csv", index=False)

# PR curve
prec_curve, rec_curve, _ = precision_recall_curve(y, proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec_curve, prec_curve, lw=2)
base = y.mean()
ax.axhline(base, color="grey", linestyle="--", lw=0.8, label=f"base rate ({base:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall — XGBoost tabular (test set)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS / "pr_curve.png", dpi=140)
plt.close()
print(f"saved {REPORTS / 'pr_curve.png'}")

# threshold sweep plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["threshold"], df["precision"], label="precision")
ax.plot(df["threshold"], df["recall"], label="recall")
ax.plot(df["threshold"], df["f1"], label="F1", lw=2)
ax.plot(df["threshold"], df["accuracy"], label="accuracy", linestyle="--")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_title("Precision/Recall/F1/Accuracy vs threshold (XGB tabular, test)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS / "threshold_sweep.png", dpi=140)
plt.close()
print(f"saved {REPORTS / 'threshold_sweep.png'}")

# headline summary
summary = {
    "default_threshold_0.5": {
        "precision": float(df[df["threshold"].between(0.495, 0.505)].iloc[0]["precision"]),
        "recall": float(df[df["threshold"].between(0.495, 0.505)].iloc[0]["recall"]),
        "f1": float(df[df["threshold"].between(0.495, 0.505)].iloc[0]["f1"]),
    },
    "picks": picks,
}
with open(REPORTS / "threshold_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"\nsaved {REPORTS / 'threshold_summary.json'}")
