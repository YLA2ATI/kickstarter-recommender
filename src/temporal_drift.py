"""
Temporal-drift analysis: train XGBoost-tabular on different time windows and
measure (a) within-window AUROC and (b) AUROC when scoring a fixed forward
test set (post-2022). Compare top-feature importances across windows.

Also reports the "transfer gap": how much does each model degrade when applied
to post-2022 data vs its own held-out year?
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

LEAKS = {"is_spotlight", "is_staff_pick", "creator_is_superbacker"}
META = {
    "id", "name", "blurb", "state", "label", "launch_year", "launched",
    "scrape_date", "is_lebanon_related",
}

clean = pd.read_parquet(DATA / "kickstarter_clean.parquet")
print(f"clean: {clean.shape}")
print(f"year range: {clean['launch_year'].min()} - {clean['launch_year'].max()}")

feature_cols = [c for c in clean.columns if c not in META and c not in LEAKS]
print(f"feature cols: {len(feature_cols)}")

# windows
WINDOWS = [
    ("2010-2014", 2010, 2014),
    ("2015-2017", 2015, 2017),
    ("2018-2020", 2018, 2020),
    ("2021+",     2021, 9999),
]

# fixed forward test set (the "future" all windows are scored against)
fwd = clean[clean["launch_year"] >= 2022].copy()
X_fwd = fwd[feature_cols].values
y_fwd = fwd["label"].values.astype(int)
print(f"forward test set (>=2022): {len(fwd):,}  pos={y_fwd.mean():.3f}")

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "max_bin": 256,
    "verbosity": 0,
    "seed": 42,
}

results = []
gain_per_window = {}

for name, lo, hi in WINDOWS:
    sub = clean[(clean["launch_year"] >= lo) & (clean["launch_year"] <= hi)]
    if len(sub) < 1000:
        print(f"  [skip] {name}: only {len(sub)} rows")
        continue
    X = sub[feature_cols].values
    y = sub["label"].values.astype(int)
    print(f"\n=== {name} ===  n={len(sub):,}  pos={y.mean():.3f}")

    # 80/20 within-window split for own-AUROC
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    dtr = xgb.QuantileDMatrix(X[idx_tr], label=y[idx_tr], max_bin=256)
    dte = xgb.QuantileDMatrix(X[idx_te], label=y[idx_te], ref=dtr, max_bin=256)
    dfwd = xgb.QuantileDMatrix(X_fwd, label=y_fwd, ref=dtr, max_bin=256)

    booster = xgb.train(
        params,
        dtr,
        num_boost_round=600,
        evals=[(dte, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    p_own = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1))
    p_fwd = booster.predict(dfwd, iteration_range=(0, booster.best_iteration + 1))
    auc_own = roc_auc_score(y[idx_te], p_own)
    auc_fwd = roc_auc_score(y_fwd, p_fwd)
    print(f"  own-window AUROC: {auc_own:.4f}")
    print(f"  forward (>=2022) AUROC: {auc_fwd:.4f}")
    print(f"  transfer gap: {auc_own - auc_fwd:+.4f}")

    score = booster.get_score(importance_type="total_gain")
    gain = pd.DataFrame({"feat_idx": list(score.keys()), "gain": list(score.values())})
    gain["feat_idx"] = gain["feat_idx"].str.lstrip("f").astype(int)
    gain["name"] = [feature_cols[i] for i in gain["feat_idx"]]
    gain = gain.sort_values("gain", ascending=False).reset_index(drop=True)
    gain_per_window[name] = gain

    results.append(
        {
            "window": name,
            "n_rows": int(len(sub)),
            "base_rate": float(y.mean()),
            "best_iteration": int(booster.best_iteration),
            "auroc_own_holdout": float(auc_own),
            "auroc_forward_2022plus": float(auc_fwd),
            "transfer_gap": float(auc_own - auc_fwd),
        }
    )
    booster.save_model(MODELS / f"xgb_drift_{name.replace('+','plus').replace('-','_')}.json")

# ---- summary csv -----------------------------------------------------------
res_df = pd.DataFrame(results)
print("\n=== Summary ===")
print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
res_df.to_csv(REPORTS / "temporal_drift_summary.csv", index=False)

# ---- side-by-side top-feature comparison -----------------------------------
TOP_K = 12
top_features = sorted(
    set().union(*[set(g.head(TOP_K)["name"]) for g in gain_per_window.values()])
)
matrix = pd.DataFrame(index=top_features)
for name, g in gain_per_window.items():
    s = g.set_index("name")["gain"]
    matrix[name] = s.reindex(top_features).fillna(0)
matrix["max_gain"] = matrix.max(axis=1)
matrix = matrix.sort_values("max_gain", ascending=False).drop(columns="max_gain")
matrix.to_csv(REPORTS / "temporal_drift_feature_gain.csv")
print("\nSaved feature-gain matrix (12 cols/window):")
print(matrix.head(15).to_string(float_format=lambda x: f"{x:.0f}"))

# ---- plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
display_features = matrix.head(15).index.tolist()
disp = matrix.loc[display_features].copy()
# normalize per window so colors are comparable
disp_norm = disp.div(disp.max(axis=0).replace(0, 1), axis=1)
im = ax.imshow(disp_norm.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(disp.shape[1]))
ax.set_xticklabels(disp.columns, rotation=15, ha="right")
ax.set_yticks(range(len(display_features)))
ax.set_yticklabels(display_features)
ax.set_title("Feature importance (normalized per window) across launch eras")
plt.colorbar(im, ax=ax, label="relative gain within window")
plt.tight_layout()
plt.savefig(REPORTS / "temporal_drift_heatmap.png", dpi=140)
plt.close()
print(f"saved {REPORTS / 'temporal_drift_heatmap.png'}")

# AUROC per window
fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(len(res_df))
ax.bar(xs - 0.2, res_df["auroc_own_holdout"], width=0.4, label="own-window holdout")
ax.bar(xs + 0.2, res_df["auroc_forward_2022plus"], width=0.4, label="forward (>=2022)")
ax.set_xticks(xs)
ax.set_xticklabels(res_df["window"])
ax.set_ylabel("AUROC")
ax.set_title("Model AUROC by training window")
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.7, 0.9)
plt.tight_layout()
plt.savefig(REPORTS / "temporal_drift_auroc.png", dpi=140)
plt.close()
print(f"saved {REPORTS / 'temporal_drift_auroc.png'}")

with open(REPORTS / "temporal_drift_summary.json", "w") as f:
    json.dump(results, f, indent=2)
