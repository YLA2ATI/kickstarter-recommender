"""
Lebanon-related campaigns: small subgroup (~49 campaigns). Run each through
the recommender, compare predicted vs actual outcomes, save markdown appendix.

Per the proposal: this is a case-study angle for Lebanese diaspora creators
who use crowdfunding for relief/reconstruction/cultural projects but lack
guidance for international audiences. Sample is too small for reliable
subgroup metrics — we treat it qualitatively.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from recommend import load_recommender  # noqa: E402

clean = pd.read_parquet(DATA / "kickstarter_clean.parquet")
leb = clean[clean["is_lebanon_related"] == 1].copy().reset_index(drop=True)
print(f"Lebanon-related campaigns: {len(leb)}")
print(f"Train (<2022): {(leb['launch_year'] < 2022).sum()}")
print(f"Test (>=2022): {(leb['launch_year'] >= 2022).sum()}")

# success rate
print(f"\nLebanon success rate: {leb['label'].mean():.3f}")
print(f"Overall success rate: {clean['label'].mean():.3f}")

rec = load_recommender()


def to_input(row):
    cat = next(
        (c.replace("category_name_", "")
         for c in clean.columns if c.startswith("category_name_") and row.get(c, 0) == 1),
        "Art",
    )
    ctry = next(
        (c.replace("country_", "")
         for c in clean.columns if c.startswith("country_") and row.get(c, 0) == 1),
        "US",
    )
    return {
        "goal_usd": float(np.expm1(row["log_goal"])),
        "duration_days": float(row["duration_days"]),
        "category": cat,
        "country": ctry,
        "has_video": int(row.get("has_video", 0)),
        "blurb": row.get("blurb", "") or "",
        "launched_dow": int(row.get("launch_dow", 1)),
        "launched_hour": int(row.get("launch_hour", 11)),
        "launched_month": int(row.get("launch_month", 4)),
        "launch_year": int(row.get("launch_year", 2024)),
    }


# score every campaign
records = []
for _, row in leb.iterrows():
    inp = to_input(row)
    out = rec.predict(inp)
    records.append(
        {
            "id": row["id"],
            "name": row.get("name", "")[:120],
            "category": inp["category"],
            "country": inp["country"],
            "goal_usd": inp["goal_usd"],
            "duration_days": inp["duration_days"],
            "launch_year": inp["launch_year"],
            "actual": int(row["label"]),
            "predicted_p": out["probability"],
            "n_recs": len(out["recommendations"]),
            "top_rec": out["recommendations"][0]["message"] if out["recommendations"] else "",
            "top_rec_feature": out["recommendations"][0]["feature"] if out["recommendations"] else "",
        }
    )
df = pd.DataFrame(records)

# subgroup metrics (acknowledge small N)
print("\n=== Lebanon subgroup metrics ===")
print(f"  n={len(df)}  base rate={df['actual'].mean():.3f}")
if df["actual"].nunique() > 1:
    from sklearn.metrics import roc_auc_score
    print(f"  subgroup AUROC: {roc_auc_score(df['actual'], df['predicted_p']):.4f}")
print(
    f"  predicted >=0.5 actually succeeded: "
    f"{((df['predicted_p'] >= 0.5) == df['actual'].astype(bool)).mean():.3f}"
)

# breakdown
print("\n=== By category ===")
print(df.groupby("category").agg(n=("id", "count"), success=("actual", "mean"),
                                  pred_mean=("predicted_p", "mean")).to_string())
print("\n=== By country ===")
print(df.groupby("country").agg(n=("id", "count"), success=("actual", "mean"),
                                 pred_mean=("predicted_p", "mean")).to_string())

# save full table
df.to_csv(REPORTS / "lebanon_case_studies.csv", index=False)

# write a markdown appendix
md_lines = [
    "# Lebanon-related campaigns — case-study appendix",
    "",
    f"This appendix runs all **{len(df)} campaigns** in the master table that "
    "matched the Lebanon-related keyword filter (`is_lebanon_related == 1`) "
    "through the pre-launch recommender. The sample is too small for reliable "
    "subgroup metrics, so we treat it qualitatively.",
    "",
    f"- Lebanon subgroup base rate: **{df['actual'].mean():.1%}** "
    f"(vs full dataset {clean['label'].mean():.1%}).",
    "",
    "## Per-campaign predictions",
    "",
]

agreed = ((df["predicted_p"] >= 0.5).astype(int) == df["actual"]).sum()
md_lines.append(f"- Model agreed with outcome (at threshold 0.5): **{agreed}/{len(df)}**")
md_lines.append("")

# top failures: predicted high, actually failed
fail_high = df[(df["actual"] == 0) & (df["predicted_p"] >= 0.6)].sort_values(
    "predicted_p", ascending=False
)
if len(fail_high) > 0:
    md_lines.append("### High-confidence misses (predicted ≥0.60, actually failed)")
    md_lines.append("")
    md_lines.append("| name | category | country | goal | predicted | actual |")
    md_lines.append("|---|---|---|---:|---:|---:|")
    for _, r in fail_high.head(10).iterrows():
        md_lines.append(
            f"| {r['name']} | {r['category']} | {r['country']} | "
            f"${r['goal_usd']:,.0f} | {r['predicted_p']:.2f} | {r['actual']} |"
        )
    md_lines.append("")

# top correct catches: predicted low, actually failed
catch_low = df[(df["actual"] == 0) & (df["predicted_p"] < 0.4)].sort_values(
    "predicted_p"
)
if len(catch_low) > 0:
    md_lines.append("### Correctly flagged risk (predicted <0.40, actually failed)")
    md_lines.append("")
    md_lines.append("| name | category | country | goal | predicted | top recommendation |")
    md_lines.append("|---|---|---|---:|---:|---|")
    for _, r in catch_low.head(10).iterrows():
        md_lines.append(
            f"| {r['name']} | {r['category']} | {r['country']} | "
            f"${r['goal_usd']:,.0f} | {r['predicted_p']:.2f} | {r['top_rec'][:100]} |"
        )
    md_lines.append("")

# good calls: predicted high, actually succeeded
hits = df[(df["actual"] == 1) & (df["predicted_p"] >= 0.7)].sort_values(
    "predicted_p", ascending=False
)
if len(hits) > 0:
    md_lines.append("### High-confidence wins (predicted ≥0.70, actually succeeded)")
    md_lines.append("")
    md_lines.append("| name | category | country | goal | predicted |")
    md_lines.append("|---|---|---|---:|---:|")
    for _, r in hits.head(10).iterrows():
        md_lines.append(
            f"| {r['name']} | {r['category']} | {r['country']} | "
            f"${r['goal_usd']:,.0f} | {r['predicted_p']:.2f} |"
        )
    md_lines.append("")

with open(REPORTS / "lebanon_case_studies.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"\nsaved {REPORTS / 'lebanon_case_studies.csv'}")
print(f"saved {REPORTS / 'lebanon_case_studies.md'}")
