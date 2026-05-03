"""
Build the headline ablation table + calibration figure.

Reads metric JSONs and saved test-set probability arrays, produces:
  reports/ablation_table.csv
  reports/ablation_table.md
  reports/calibration.png
  reports/roc_curves.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

y_test = pd.read_parquet(DATA / "y_test.parquet")["label"].values.astype(int)

variants = [
    ("LR baseline (52 tabular)", REPORTS / "metrics_baseline_lr.json", MODELS / "lr_baseline_proba_test.npy"),
    ("XGBoost (52 tabular)", REPORTS / "metrics_xgb_tabular.json", MODELS / "xgb_tabular_proba_test.npy"),
    ("XGBoost full minus Trends", REPORTS / "metrics_xgb_full_no_trends.json", MODELS / "xgb_full_no_trends_proba_test.npy"),
    ("XGBoost full (596k features)", REPORTS / "metrics_xgb_full.json", MODELS / "xgb_full_proba_test.npy"),
]

rows = []
probas = {}
for name, mfile, pfile in variants:
    if not mfile.exists() or not pfile.exists():
        print(f"  [skip] {name}: missing {mfile.name} or {pfile.name}")
        continue
    with open(mfile) as f:
        m = json.load(f)
    test = m["test"]
    rows.append(
        {
            "model": name,
            "n_features": m.get("n_features", "?"),
            "auroc": test["auroc"],
            "ap": test["ap"],
            "brier": test["brier"],
            "logloss": test["logloss"],
            "accuracy": test["accuracy"],
            "fit_time_s": m.get("fit_time_seconds", float("nan")),
        }
    )
    probas[name] = np.load(pfile)

table = pd.DataFrame(rows)
print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
table.to_csv(REPORTS / "ablation_table.csv", index=False)
with open(REPORTS / "ablation_table.md", "w") as f:
    f.write("# Ablation results (test set, post-2022)\n\n")
    cols = list(table.columns)
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for _, r in table.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        f.write("| " + " | ".join(cells) + " |\n")

# ROC curves
plt.figure(figsize=(8, 6))
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=name, lw=1.6)
plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC — held-out test set (post-2022)")
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS / "roc_curves.png", dpi=140)
plt.close()
print(f"saved {REPORTS / 'roc_curves.png'}")

# Calibration (reliability diagram)
plt.figure(figsize=(8, 6))
for name, p in probas.items():
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=15, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=name, lw=1.6)
plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
plt.xlabel("Mean predicted probability (per quantile bin)")
plt.ylabel("Empirical success rate")
plt.title("Calibration — held-out test set (post-2022)")
plt.legend(loc="upper left", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS / "calibration.png", dpi=140)
plt.close()
print(f"saved {REPORTS / 'calibration.png'}")
print(f"saved {REPORTS / 'ablation_table.csv'}")
print(f"saved {REPORTS / 'ablation_table.md'}")
