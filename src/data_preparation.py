"""
data_preparation.py
-------------------
Combines multiple WebRobots Kickstarter CSV snapshots into one deduped dataset,
engineers features, attaches Trends, and saves Parquet outputs.

Expected folder layout:
data/raw/
  2019-12-12/*.csv
  2020-12-17/*.csv
  2021-05-13/*.csv
  2022-06-09/*.csv
  2023-10-12/*.csv
  2024-12-12/*.csv
  2025-06-12/*.csv
  2026-04-13/*.csv

Install:
  pip install pandas numpy pyarrow pytrends textstat
"""

import os
import re
import glob
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# textstat is optional; we fallback if missing
try:
    import textstat
    _HAS_TEXTSTAT = True
except Exception:
    _HAS_TEXTSTAT = False

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/raw"
OUTPUT_DIR   = "data"
TRENDS_FILE  = "trend_lookup.pkl"
TRAIN_CUTOFF = 2022  # launched before 2022 => train, 2022+ => test

KEEP_COLS = [
    "id","name","blurb","state","goal","launched_at","deadline",
    "country","category","backers_count","staff_pick","spotlight",
    "usd_pledged","video","creator","source_url"
]

RESOLVED_STATES = {"successful","failed"}

# ── Subcategory to parent mapping ─────────────────────────────────────────────
# Maps any subcategory name to its Kickstarter top-level parent category.
# This ensures trend_score lookup always matches the 15 parent categories.
SUBCATEGORY_TO_PARENT = {
    # Art
    "Conceptual Art": "Art", "Digital Art": "Art", "Fine Art": "Art",
    "Illustration": "Art", "Installations": "Art", "Mixed Media": "Art",
    "Painting": "Art", "Performance Art": "Art", "Public Art": "Art",
    "Sculpture": "Art", "Video Art": "Art",
    # Comics
    "Comic Books": "Comics", "Graphic Novels": "Comics", "Webcomics": "Comics",
    # Crafts
    "Candles": "Crafts", "DIY": "Crafts", "Embroidery": "Crafts",
    "Glass": "Crafts", "Knitting": "Crafts", "Letterpress": "Crafts",
    "Pottery": "Crafts", "Printing": "Crafts", "Quilts": "Crafts",
    "Stationery": "Crafts", "Taxidermy": "Crafts", "Weaving": "Crafts",
    "Woodworking": "Crafts",
    # Dance
    "Performances": "Dance", "Residencies": "Dance", "Spaces": "Dance",
    "Workshops": "Dance",
    # Design
    "Architecture": "Design", "Graphic Design": "Design",
    "Interactive Design": "Design", "Product Design": "Design",
    "Typography": "Design",
    # Fashion
    "Accessories": "Fashion", "Apparel": "Fashion", "Childrenswear": "Fashion",
    "Couture": "Fashion", "Footwear": "Fashion", "Jewelry": "Fashion",
    "Ready-to-wear": "Fashion", "Wearables": "Fashion",
    # Film & Video
    "Action": "Film & Video", "Animation": "Film & Video",
    "Comedy": "Film & Video", "Documentary": "Film & Video",
    "Drama": "Film & Video", "Experimental": "Film & Video",
    "Fantasy": "Film & Video", "Horror": "Film & Video",
    "Movie Theaters": "Film & Video", "Music Videos": "Film & Video",
    "Narrative Film": "Film & Video", "Romance": "Film & Video",
    "Science Fiction": "Film & Video", "Shorts": "Film & Video",
    "Television": "Film & Video", "Thrillers": "Film & Video",
    "Video": "Film & Video", "Webseries": "Film & Video",
    # Food
    "Bacon": "Food", "Community Gardens": "Food", "Drinks": "Food",
    "Events": "Food", "Farms": "Food", "Food Trucks": "Food",
    "Restaurants": "Food", "Small Batch": "Food", "Vegan": "Food",
    # Games
    "Gaming Hardware": "Games", "Live Games": "Games",
    "Mobile Games": "Games", "Playing Cards": "Games",
    "Puzzles": "Games", "Robots": "Games", "Tabletop Games": "Games",
    "Video Games": "Games",
    # Journalism
    "Audio": "Journalism", "Photo": "Journalism", "Print": "Journalism",
    "Video (Journalism)": "Journalism", "Web": "Journalism",
    # Music
    "Blues": "Music", "Chiptune": "Music", "Classical Music": "Music",
    "Country & Folk": "Music", "Electronic Music": "Music",
    "Faith": "Music", "Hip-Hop": "Music", "Indie Rock": "Music",
    "Jazz": "Music", "Kids": "Music", "Latin": "Music", "Metal": "Music",
    "Musical": "Music", "Pop": "Music", "Punk": "Music",
    "R&B": "Music", "Rock": "Music", "Soul": "Music",
    "Soundtracks": "Music", "World Music": "Music",
    # Photography
    "Nature": "Photography", "People": "Photography",
    "Photobooks": "Photography", "Places": "Photography",
    # Publishing
    "Academic": "Publishing", "Anthologies": "Publishing",
    "Art Books": "Publishing", "Children's Books": "Publishing",
    "Cookbooks": "Publishing", "Fiction": "Publishing",
    "Literary Journals": "Publishing", "Literary Spaces": "Publishing",
    "Nonfiction": "Publishing", "Periodicals": "Publishing",
    "Poetry": "Publishing", "Radio & Podcasts": "Publishing",
    "Translations": "Publishing", "Young Adult": "Publishing",
    "Zines": "Publishing",
    # Technology
    "3D Printing": "Technology", "Apps": "Technology",
    "Camera Equipment": "Technology", "DIY Electronics": "Technology",
    "Fabrication Tools": "Technology", "Flight": "Technology",
    "Gadgets": "Technology", "Hardware": "Technology",
    "Software": "Technology", "Sound": "Technology",
    "Space Exploration": "Technology",
    # Theater
    "Comedy (Theater)": "Theater", "Experimental (Theater)": "Theater",
    "Immersive": "Theater", "Musical (Theater)": "Theater",
    "Plays": "Theater",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def infer_scrape_date_from_path(path: str) -> str:
    """Extract YYYY-MM-DD from folder name anywhere in path."""
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", path)  # FIXED: was r"(20\\d{2}-\\d{2}-\\d{2})"
    return m.group(1) if m else "1900-01-01"

def safe_json_loads(val):
    """More robust JSON parsing for messy strings."""
    if not isinstance(val, str) or not val.strip():
        return None
    try:
        return json.loads(val)
    except Exception:
        try:
            return json.loads(val.replace("'", '"'))
        except Exception:
            return None

# ── Step 1: Load CSVs ─────────────────────────────────────────────────────────
def load_all_csvs(data_dir):
    pattern = os.path.join(data_dir, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    print(f"Found {len(files)} CSV files")

    dfs = []
    for f in sorted(files):
        scrape_date = infer_scrape_date_from_path(f)
        try:
            df = pd.read_csv(
                f,
                low_memory=False,
                usecols=lambda c: c in KEEP_COLS
            )
            df["scrape_date"] = scrape_date
            dfs.append(df)
            print(f"  Loaded {os.path.basename(f)} ({scrape_date}): {len(df):,} rows")
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(combined):,}")
    return combined

# ── Step 2: Prefer resolved rows + latest scrape when deduping ────────────────
def deduplicate_prefer_resolved_latest(df):
    df = df.copy()
    df["is_resolved"] = df["state"].isin(list(RESOLVED_STATES)).astype(int)

    df["scrape_date_dt"] = pd.to_datetime(df["scrape_date"], errors="coerce")

    # Sort so best row is last per id: resolved first, then newest scrape_date
    df = df.sort_values(["id","is_resolved","scrape_date_dt"], ascending=[True, True, True])

    before = len(df)
    df = df.drop_duplicates(subset="id", keep="last")
    after = len(df)
    print(f"Dedup: {before:,} -> {after:,} unique campaigns")

    return df.drop(columns=["is_resolved","scrape_date_dt"])

# ── Step 3: Keep resolved only ────────────────────────────────────────────────
def filter_resolved(df):
    df = df[df["state"].isin(list(RESOLVED_STATES))].copy()
    print(f"Resolved campaigns: {len(df):,}")
    print(df["state"].value_counts().to_string())
    return df

# ── Step 4: Timestamps ───────────────────────────────────────────────────────
def parse_timestamps(df):
    df = df.copy()
    df["launched"] = pd.to_datetime(df["launched_at"], unit="s", utc=True, errors="coerce")
    df["deadline_dt"] = pd.to_datetime(df["deadline"], unit="s", utc=True, errors="coerce")
    df["duration_days"] = (df["deadline_dt"] - df["launched"]).dt.days

    df["launch_year"] = df["launched"].dt.year
    df["launch_month"] = df["launched"].dt.month
    df["launch_dow"] = df["launched"].dt.dayofweek
    df["launch_hour"] = df["launched"].dt.hour
    return df

# ── Step 5: Category parsing ──────────────────────────────────────────────────
TOP_LEVEL = set(SUBCATEGORY_TO_PARENT.values())

def parse_category_name(val):
    parsed = safe_json_loads(val)
    if isinstance(parsed, dict):
        parent = parsed.get("parent_name") or ""
        name   = parsed.get("name") or ""
        if parent in TOP_LEVEL:
            return parent
        if name in SUBCATEGORY_TO_PARENT:
            return SUBCATEGORY_TO_PARENT[name]
        if name in TOP_LEVEL:
            return name
        return parent or name or "Unknown"
    return "Unknown"

def add_category(df):
    df = df.copy()
    df["category_name"] = df["category"].apply(parse_category_name)
    print("Top categories:")
    print(df["category_name"].value_counts().head(10).to_string())
    return df

# ── Step 6: Tabular features ──────────────────────────────────────────────────
def engineer_tabular(df):
    df = df.copy()

    df["goal_usd"] = pd.to_numeric(df["goal"], errors="coerce")
    df["log_goal"] = np.log1p(df["goal_usd"].clip(lower=0))

    df["has_video"] = df["video"].notna() & (df["video"].astype(str).str.lower() != "nan")
    df["is_staff_pick"] = df["staff_pick"].fillna(False).astype(bool)
    df["is_spotlight"] = df["spotlight"].fillna(False).astype(bool)

    def creator_is_superbacker(val):
        parsed = safe_json_loads(val)
        if isinstance(parsed, dict):
            return bool(parsed.get("is_superbacker", False))
        return False

    df["creator_is_superbacker"] = df["creator"].apply(creator_is_superbacker)

    pattern = r"(?:lebanon|lebanese|beirut|liban)"
    df["is_lebanon_related"] = (
        df["name"].astype(str).str.contains(pattern, case=False, na=False) |
        df["blurb"].astype(str).str.contains(pattern, case=False, na=False)
    )

    return df

# ── Step 7: NLP features ──────────────────────────────────────────────────────
def nlp_features(text):
    if not isinstance(text, str) or not text.strip():
        return pd.Series({
            "blurb_readability": 50.0,
            "blurb_word_count": 0,
            "blurb_has_number": 0,
            "blurb_has_exclaim": 0,
            "blurb_has_question": 0,
        })

    readability = 50.0
    if _HAS_TEXTSTAT:
        try:
            readability = float(textstat.flesch_reading_ease(text))
        except Exception:
            readability = 50.0

    return pd.Series({
        "blurb_readability": readability,
        "blurb_word_count": len(text.split()),
        "blurb_has_number": int(bool(re.search(r"\d", text))),  # FIXED: was r"\\d"
        "blurb_has_exclaim": int("!" in text),
        "blurb_has_question": int("?" in text),
    })

def add_nlp_features(df):
    print("Extracting NLP features...")
    feats = df["blurb"].apply(nlp_features)
    return pd.concat([df, feats], axis=1)

# ── Step 8: Trends ────────────────────────────────────────────────────────────
def add_trends(df, trends_file):
    df = df.copy()
    if not os.path.exists(trends_file):
        print(f"WARNING: {trends_file} not found. trend_score set to 50.")
        df["trend_score"] = 50
        return df

    with open(trends_file, "rb") as f:
        lookup = pickle.load(f)

    def get_score(row):
        key = (row["category_name"], int(row["launch_year"]), int(row["launch_month"]))
        return lookup.get(key, 50)

    print("Attaching trend_score...")
    df["trend_score"] = df.apply(get_score, axis=1)
    coverage = (df["trend_score"] != 50).mean() * 100
    print(f"Trend match coverage: {coverage:.1f}%")
    return df

# ── Step 9: Missing values ────────────────────────────────────────────────────
def fill_missing(df):
    df = df.copy()
    df["goal_usd"] = df["goal_usd"].fillna(df["goal_usd"].median())
    df["log_goal"] = df["log_goal"].fillna(df["log_goal"].median())
    df["duration_days"] = df["duration_days"].fillna(30)
    df["country"] = df["country"].fillna("US")
    df["category_name"] = df["category_name"].fillna("Unknown")
    df["blurb"] = df["blurb"].fillna("")
    df["launch_dow"] = df["launch_dow"].fillna(0)
    df["launch_hour"] = df["launch_hour"].fillna(0)
    df["launch_month"] = df["launch_month"].fillna(1)
    return df

# ── Step 10: Label ────────────────────────────────────────────────────────────
def add_label(df):
    df = df.copy()
    df["label"] = (df["state"] == "successful").astype(int)
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    print(f"Success rate: {df['label'].mean()*100:.1f}%")
    return df

# ── Step 11: Encode ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "log_goal","duration_days","launch_dow","launch_hour","launch_month",
    "trend_score","blurb_readability","blurb_word_count",
    "has_video","is_staff_pick","is_spotlight","creator_is_superbacker",
    "blurb_has_number","blurb_has_exclaim","blurb_has_question",
    "category_name","country",
]

META_COLS = [
    "id","name","blurb","state","label","launch_year","launched","scrape_date","is_lebanon_related"
]

def encode(df):
    df = df.copy()
    df_out = pd.get_dummies(
        df[FEATURE_COLS + META_COLS],
        columns=["category_name","country"],
        drop_first=True
    )
    return df_out

# ── Step 12: Split ────────────────────────────────────────────────────────────
def split(df_encoded):
    train = df_encoded[df_encoded["launch_year"] < TRAIN_CUTOFF].copy()
    test  = df_encoded[df_encoded["launch_year"] >= TRAIN_CUTOFF].copy()

    drop_cols = {"id","name","blurb","state","label","launch_year","launched","scrape_date","is_lebanon_related"}
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]

    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]

    print(f"Train: {len(train):,} | Test: {len(test):,} | Features: {len(feature_cols)}")
    print(f"Lebanon-related in test: {int(test['is_lebanon_related'].sum())}")
    return X_train, X_test, y_train, y_test, df_encoded

# ── Step 13: Save ─────────────────────────────────────────────────────────────
def save_all(df_clean, X_train, X_test, y_train, y_test):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df_clean.to_parquet(f"{OUTPUT_DIR}/kickstarter_clean.parquet", index=False)
    X_train.to_parquet(f"{OUTPUT_DIR}/X_train.parquet", index=False)
    X_test.to_parquet(f"{OUTPUT_DIR}/X_test.parquet", index=False)
    y_train.to_frame("label").to_parquet(f"{OUTPUT_DIR}/y_train.parquet", index=False)
    y_test.to_frame("label").to_parquet(f"{OUTPUT_DIR}/y_test.parquet", index=False)

    for fn in ["kickstarter_clean.parquet","X_train.parquet","X_test.parquet","y_train.parquet","y_test.parquet"]:
        size_mb = os.path.getsize(f"{OUTPUT_DIR}/{fn}") / 1e6
        print(f"Saved {fn} ({size_mb:.1f} MB)")

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Kickstarter dataset preparation ===")

    df = load_all_csvs(DATA_DIR)
    df = deduplicate_prefer_resolved_latest(df)
    df = filter_resolved(df)
    df = parse_timestamps(df)
    df = add_category(df)
    df = engineer_tabular(df)
    df = add_nlp_features(df)
    df = add_trends(df, TRENDS_FILE)
    df = fill_missing(df)
    df = add_label(df)

    df_encoded = encode(df)
    X_train, X_test, y_train, y_test, df_clean = split(df_encoded)
    save_all(df_clean, X_train, X_test, y_train, y_test)

    print("Done.")
