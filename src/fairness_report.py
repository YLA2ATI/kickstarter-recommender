"""
Fairness / disparity report.

For the recommender model (XGBoost tabular), measure:
  - test-set success rate per category and per country
  - per-group AUROC
  - per-group calibration error
to surface where the model performs unevenly.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

X_test = pd.read_parquet(DATA / "X_test_clean.parquet")
y_test = pd.read_parquet(DATA / "y_test.parquet")["label"].values.astype(int)

booster = xgb.Booster()
booster.load_model(str(MODELS / "xgb_tabular.json"))
proba = booster.predict(xgb.DMatrix(X_test.values, feature_names=list(X_test.columns)))


def group_metrics(mask: np.ndarray, name: str):
    n = int(mask.sum())
    if n < 30:
        return None
    yt = y_test[mask]
    pp = proba[mask]
    if yt.min() == yt.max():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(yt, pp))
    return {
        "group": name,
        "n": n,
        "base_rate": float(yt.mean()),
        "mean_pred": float(pp.mean()),
        "auroc": auc,
        "brier": float(brier_score_loss(yt, pp)),
        "calibration_error": float(pp.mean() - yt.mean()),
    }


# categories — column starts with category_name_
cat_rows = []
for c in [c for c in X_test.columns if c.startswith("category_name_")]:
    name = c.replace("category_name_", "")
    m = X_test[c].values == 1
    g = group_metrics(m, name)
    if g:
        cat_rows.append(g)
# Plus Art (the dropped baseline category — represented by all zeros across
# category_name_*).
all_cat_cols = [c for c in X_test.columns if c.startswith("category_name_")]
m_art = (X_test[all_cat_cols].sum(axis=1) == 0).values
g = group_metrics(m_art, "Art (baseline)")
if g:
    cat_rows.append(g)

cat_df = pd.DataFrame(cat_rows).sort_values("auroc", ascending=False)
print("=== Per-category metrics (test set) ===")
print(cat_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
cat_df.to_csv(REPORTS / "fairness_by_category.csv", index=False)

# countries — likewise
ctry_rows = []
for c in [c for c in X_test.columns if c.startswith("country_")]:
    name = c.replace("country_", "")
    m = X_test[c].values == 1
    g = group_metrics(m, name)
    if g:
        ctry_rows.append(g)
all_ctry_cols = [c for c in X_test.columns if c.startswith("country_")]
m_baseline = (X_test[all_ctry_cols].sum(axis=1) == 0).values
g = group_metrics(m_baseline, "(other / baseline)")
if g:
    ctry_rows.append(g)

ctry_df = pd.DataFrame(ctry_rows).sort_values("auroc", ascending=False)
print("\n=== Per-country metrics (test set) ===")
print(ctry_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
ctry_df.to_csv(REPORTS / "fairness_by_country.csv", index=False)

# headline summary
summary = {
    "category_auroc_min_max": [float(cat_df["auroc"].min()), float(cat_df["auroc"].max())],
    "category_base_rate_min_max": [float(cat_df["base_rate"].min()), float(cat_df["base_rate"].max())],
    "country_auroc_min_max": [float(ctry_df["auroc"].min()), float(ctry_df["auroc"].max())],
    "country_base_rate_min_max": [float(ctry_df["base_rate"].min()), float(ctry_df["base_rate"].max())],
    "max_calibration_error_category": float(cat_df["calibration_error"].abs().max()),
    "max_calibration_error_country": float(ctry_df["calibration_error"].abs().max()),
}
with open(REPORTS / "fairness_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n=== Summary ===")
print(json.dumps(summary, indent=2))
