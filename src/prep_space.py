"""
Prepare the Hugging Face Space deploy folder by slimming down data files.

The full kickstarter_clean.parquet is 44 MB (would need git LFS). The app only
needs post-2022 rows for the "load held-out campaign" feature, which trims it
to a manageable size with no LFS required.
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT.parent / "kickstarter_space"
DATA_SRC = ROOT / "data"
DATA_DST = SPACE / "data"
MODELS_SRC = ROOT / "models"
MODELS_DST = SPACE / "models"

# slim kickstarter_clean: only post-2022 + drop columns the app doesn't use
clean = pd.read_parquet(DATA_SRC / "kickstarter_clean.parquet")
print(f"original clean: {clean.shape}, {DATA_SRC.joinpath('kickstarter_clean.parquet').stat().st_size / 1e6:.1f} MB")

DROP_COLS = {
    "is_staff_pick", "is_spotlight", "creator_is_superbacker",
    "state", "launched", "scrape_date", "id",
    # blurb stats are recomputed at predict time inside recommend.py
    "blurb_readability", "blurb_word_count",
    "blurb_has_number", "blurb_has_exclaim", "blurb_has_question",
    "trend_score", "has_trend",
    "is_lebanon_related",
}
keep_cols = [c for c in clean.columns if c not in DROP_COLS]
# HF rejects binary files in regular git — sample to 1000 rows and ship as CSV
clean_slim = (
    clean[clean["launch_year"] >= 2022][keep_cols]
    .sample(n=1000, random_state=42)
    .reset_index(drop=True)
)
clean_slim.to_csv(DATA_DST / "kickstarter_clean.csv", index=False)
# remove old parquet if present
old_parquet = DATA_DST / "kickstarter_clean.parquet"
if old_parquet.exists():
    old_parquet.unlink()
print(f"slim clean: {clean_slim.shape}, "
      f"{DATA_DST.joinpath('kickstarter_clean.csv').stat().st_size / 1e6:.1f} MB")

# the recommender wants the column list — save it as a tiny json instead of
# shipping the full X_train_clean.parquet
xtrain = pd.read_parquet(DATA_SRC / "X_train_clean.parquet")
import json
with open(DATA_DST / "feature_columns.json", "w") as f:
    json.dump(list(xtrain.columns), f)
print(f"feature_columns.json: {DATA_DST.joinpath('feature_columns.json').stat().st_size / 1e3:.1f} KB")

# tiny files we just copy as-is
for src, name in [
    (DATA_SRC / "trend_lookup.pkl", "trend_lookup.pkl"),
    (MODELS_SRC / "xgb_tabular.json", "xgb_tabular.json"),
]:
    dst = (DATA_DST if "lookup" in name else MODELS_DST) / name
    shutil.copy(src, dst)
    print(f"copied {name}: {dst.stat().st_size / 1e3:.1f} KB")

# copy app code
shutil.copy(ROOT / "app" / "streamlit_app.py", SPACE / "app" / "streamlit_app.py")
print("copied app/streamlit_app.py")

# show final size summary
total = sum(p.stat().st_size for p in SPACE.rglob("*") if p.is_file())
print(f"\nspace total size so far: {total / 1e6:.1f} MB")
