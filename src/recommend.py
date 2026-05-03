"""
SHAP-based pre-launch recommender for the tabular XGBoost model.

Public API:
  load_recommender() -> Recommender
  Recommender.predict(campaign_dict) -> dict with proba, shap, top recs

A campaign_dict has keys matching tabular feature schema:
  goal_usd: float (raw, will be converted to log_goal internally)
  duration_days: float
  category: str  (one of TOP-LEVEL category names)
  country: str   (e.g. "US", "GB", ...)
  has_video: 0/1
  blurb: str
  launched_dow: 0..6 (Mon=0)
  launched_hour: 0..23
  launched_month: 1..12
  launch_year: int
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

# ---- the natural-language translation layer for SHAP attributions -----------
# Each rule is a dict with:
#   feature: the feature column name
#   applies: callable(current_value, campaign) -> bool — whether the rec is sensible
#   message: format string
# Rules only fire when SHAP < negative_threshold AND applies(...) is True.

ACTIONABLE_RULES = [
    # creator-controllable
    {
        "feature": "log_goal",
        "applies": lambda v, c: c["goal_usd"] >= 5000,
        "message": "Your goal of ${goal} may be too high for this profile. Consider lowering it.",
    },
    {
        "feature": "duration_days",
        "applies": lambda v, c: c["duration_days"] > 35,
        "message": "Your campaign length of {duration} days is hurting your odds. Most successful campaigns run 21-35 days.",
    },
    {
        "feature": "has_video",
        "applies": lambda v, c: int(c.get("has_video", 0)) == 0,
        "message": "Add a video. Campaigns without a video underperform meaningfully in this category.",
    },
    {
        "feature": "blurb_word_count",
        "applies": lambda v, c: True,  # always actionable, but message branches on length
        "message": None,  # filled at runtime based on word count
    },
    {
        "feature": "blurb_readability",
        "applies": lambda v, c: True,
        "message": "Your blurb is hard to read at a glance. Simplify the language.",
    },
    {
        "feature": "blurb_has_number",
        "applies": lambda v, c: not bool(re.search(r"\d", c.get("blurb", ""))),
        "message": "Adding a concrete number to your blurb (e.g., page count, run time, dimensions) tends to help.",
    },
    {
        "feature": "blurb_has_exclaim",
        "applies": lambda v, c: "!" not in (c.get("blurb", "") or ""),
        "message": "A small dose of energy helps - consider rephrasing to sound more confident.",
    },
    {
        "feature": "blurb_has_question",
        "applies": lambda v, c: "?" not in (c.get("blurb", "") or ""),
        "message": "Opening with a question that hooks the reader can lift engagement.",
    },
    # timing
    {
        "feature": "launch_dow",
        "applies": lambda v, c: int(c["launched_dow"]) not in {1, 2},  # Tue/Wed
        "message": "Your planned launch day-of-week ({dow_name}) is associated with lower success. Tuesdays and Wednesdays tend to perform best.",
    },
    {
        "feature": "launch_hour",
        "applies": lambda v, c: int(c["launched_hour"]) < 9 or int(c["launched_hour"]) > 16,
        "message": "Your launch hour ({hour}:00) is associated with lower success. Late morning to early afternoon launches tend to perform better.",
    },
    {
        "feature": "launch_month",
        "applies": lambda v, c: True,
        "message": "Your launch month ({month_name}) is a weaker month for this category. Consider shifting the launch.",
    },
    # context (descriptive only — no obvious creator action, but worth surfacing)
    {
        "feature": "trend_score",
        "applies": lambda v, c: True,
        "message": "Google Trends interest for {category} is currently soft. Either wait for the next interest peak or lean into your differentiators.",
    },
]

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@dataclass
class Recommender:
    booster: xgb.Booster
    feature_names: list[str]
    trend_lookup: dict
    category_cols: list[str]
    country_cols: list[str]

    # ---- feature engineering (mirrors the original prep pipeline) -----------
    def _build_row(self, c: dict[str, Any]) -> pd.DataFrame:
        row = {name: 0.0 for name in self.feature_names}
        row["log_goal"] = float(np.log1p(c["goal_usd"]))
        row["duration_days"] = float(c["duration_days"])
        row["launch_dow"] = int(c["launched_dow"])
        row["launch_hour"] = int(c["launched_hour"])
        row["launch_month"] = int(c["launched_month"])

        cat = c["category"]
        year = int(c["launch_year"])
        month = int(c["launched_month"])
        ts = self.trend_lookup.get((cat, year, month))
        if ts is None:
            row["trend_score"] = 50.0
            row["has_trend"] = 0.0
        else:
            row["trend_score"] = float(ts)
            row["has_trend"] = 1.0

        row["has_video"] = int(c.get("has_video", 0))
        blurb = (c.get("blurb") or "").strip()
        words = re.findall(r"\w+", blurb)
        row["blurb_word_count"] = float(len(words))
        # Flesch reading ease approximation (avoid heavy dep)
        row["blurb_readability"] = float(
            _quick_reading_ease(blurb) if blurb else 60.0
        )
        row["blurb_has_number"] = int(bool(re.search(r"\d", blurb)))
        row["blurb_has_exclaim"] = int("!" in blurb)
        row["blurb_has_question"] = int("?" in blurb)

        cat_col = f"category_name_{cat}"
        if cat_col in row:
            row[cat_col] = 1.0
        ctry_col = f"country_{c['country']}"
        if ctry_col in row:
            row[ctry_col] = 1.0
        return pd.DataFrame([row], columns=self.feature_names)

    # ---- prediction + SHAP --------------------------------------------------
    def predict(self, campaign: dict[str, Any]) -> dict[str, Any]:
        X = self._build_row(campaign)
        d = xgb.DMatrix(X.values, feature_names=self.feature_names)
        proba = float(self.booster.predict(d)[0])

        # SHAP values via XGBoost's exact tree shap
        shap = self.booster.predict(d, pred_contribs=True)[0]
        bias = float(shap[-1])
        contribs = shap[:-1]
        shap_df = (
            pd.DataFrame({"feature": self.feature_names, "shap": contribs, "value": X.values[0]})
            .sort_values("shap", key=lambda s: s.abs(), ascending=False)
        )

        recs = self._build_recommendations(campaign, shap_df)
        return {
            "probability": proba,
            "log_odds_bias": bias,
            "shap_top": shap_df.head(15).to_dict(orient="records"),
            "recommendations": recs,
        }

    def _build_recommendations(self, campaign, shap_df) -> list[dict[str, Any]]:
        recs = []
        # only consider features whose SHAP pushed prediction DOWN substantially
        neg = shap_df[shap_df["shap"] < -0.05].copy()

        for rule in ACTIONABLE_RULES:
            feat = rule["feature"]
            hits = neg[neg["feature"] == feat]
            if hits.empty:
                continue
            row = hits.iloc[0]
            current_value = float(row["value"])
            if not rule["applies"](current_value, campaign):
                continue

            # branch on word count for the blurb message
            if feat == "blurb_word_count":
                wc = int(current_value or 0)
                if wc < 10:
                    msg = (
                        f"Your blurb is only {wc} words. Expand to ~18-28 words "
                        "that state the product, audience, and one differentiator."
                    )
                elif wc > 35:
                    msg = (
                        f"Your blurb is {wc} words - too long. Cut to 18-28 words; "
                        "front-load the hook."
                    )
                else:
                    # SHAP says the count is hurting but the count is in the safe zone:
                    # likely interaction with blurb content, not length. Skip.
                    continue
            else:
                msg = rule["message"].format(
                    goal=f"{int(campaign['goal_usd']):,}",
                    duration=int(campaign["duration_days"]),
                    dow_name=DOW_NAMES[int(campaign["launched_dow"]) % 7],
                    hour=int(campaign["launched_hour"]),
                    month_name=MONTH_NAMES[int(campaign["launched_month"])],
                    category=campaign.get("category", "this category"),
                )

            recs.append(
                {
                    "feature": feat,
                    "shap": float(row["shap"]),
                    "current_value": current_value,
                    "message": msg,
                }
            )
        recs.sort(key=lambda r: r["shap"])  # most damaging first
        return recs[:6]


# ---- utility: light-weight Flesch reading ease (no extra deps) -------------
_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.I)


def _count_syllables(word: str) -> int:
    word = word.lower().rstrip("e")
    return max(1, len(_VOWEL_GROUPS.findall(word)))


def _quick_reading_ease(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"\w+", text)
    if not words:
        return 60.0
    syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / sentences
    asw = syllables / len(words)
    return 206.835 - 1.015 * asl - 84.6 * asw


# ---- loader ----------------------------------------------------------------
def load_recommender(model_path: Path | None = None) -> Recommender:
    booster = xgb.Booster()
    booster.load_model(str(model_path or (MODELS / "xgb_tabular.json")))
    cols = list(pd.read_parquet(DATA / "X_train_clean.parquet").columns)
    trend_lookup = joblib.load(DATA / "trend_lookup.pkl")
    cat_cols = [c for c in cols if c.startswith("category_name_")]
    ctry_cols = [c for c in cols if c.startswith("country_")]
    return Recommender(
        booster=booster,
        feature_names=cols,
        trend_lookup=trend_lookup,
        category_cols=cat_cols,
        country_cols=ctry_cols,
    )


CATEGORIES = [
    "Art", "Comics", "Crafts", "Dance", "Design", "Fashion",
    "Film & Video", "Food", "Games", "Journalism", "Music",
    "Photography", "Publishing", "Technology", "Theater",
]
COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "IT", "NL", "ES", "MX", "JP", "HK"]


if __name__ == "__main__":
    rec = load_recommender()

    # demo: a low-quality campaign
    weak = {
        "goal_usd": 250000,
        "duration_days": 60,
        "category": "Technology",
        "country": "US",
        "has_video": 0,
        "blurb": "smart device innovation",
        "launched_dow": 5,  # Saturday
        "launched_hour": 3,
        "launched_month": 11,
        "launch_year": 2024,
    }
    out = rec.predict(weak)
    print("=== WEAK CAMPAIGN ===")
    print(f"Predicted P(success) = {out['probability']:.3f}")
    print(f"Top recommendations:")
    for r in out["recommendations"]:
        print(f"  [{r['feature']:>22} shap={r['shap']:+.3f}]  {r['message']}")
    print()

    # demo: a strong campaign
    strong = {
        "goal_usd": 8000,
        "duration_days": 30,
        "category": "Games",
        "country": "US",
        "has_video": 1,
        "blurb": "A 96-page hardback solo journaling RPG with 12 unique scenarios. Print run unlocked at $5k.",
        "launched_dow": 1,  # Tuesday
        "launched_hour": 11,
        "launched_month": 4,
        "launch_year": 2024,
    }
    out = rec.predict(strong)
    print("=== STRONG CAMPAIGN ===")
    print(f"Predicted P(success) = {out['probability']:.3f}")
    print(f"Top recommendations (if any):")
    for r in out["recommendations"]:
        print(f"  [{r['feature']:>22} shap={r['shap']:+.3f}]  {r['message']}")
