"""
Kickstarter Pre-Launch Recommender — Streamlit demo.

Run from the project root:
    venv/Scripts/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommend import CATEGORIES, COUNTRIES, DOW_NAMES, MONTH_NAMES, load_recommender  # noqa: E402

st.set_page_config(page_title="Kickstarter Pre-Launch Recommender", layout="wide")

st.title("Kickstarter Pre-Launch Recommender")
st.caption(
    "Pre-launch success-probability prediction with SHAP-explained recommendations. "
    "Trained on 236k resolved campaigns (2009–2021), evaluated on 97k post-2022 campaigns."
)


@st.cache_resource
def _load():
    return load_recommender()


@st.cache_resource
def _load_clean_table():
    csv_path = ROOT / "data" / "kickstarter_clean.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.read_parquet(ROOT / "data" / "kickstarter_clean.parquet")


rec = _load()

# ---- sidebar inputs --------------------------------------------------------
st.sidebar.header("Campaign details")

mode = st.sidebar.radio(
    "Input mode",
    ["Enter manually", "Load a real held-out campaign"],
    index=0,
)

clean = _load_clean_table()
test_pool = clean[clean["launch_year"] >= 2022].reset_index(drop=True)

if mode == "Load a real held-out campaign":
    if "sample_id" not in st.session_state:
        st.session_state["sample_id"] = int(np.random.choice(test_pool.index, 1)[0])
    if st.sidebar.button("Pick another random campaign"):
        st.session_state["sample_id"] = int(np.random.choice(test_pool.index, 1)[0])
    s = test_pool.loc[st.session_state["sample_id"]]
    actual_label = int(s["label"])
    st.sidebar.info(f"Actual outcome (hidden until you predict): **{'?'}** ")
    name_default = s.get("name", "")
    blurb_default = s.get("blurb", "")
    goal_default = float(np.expm1(s["log_goal"]))
    duration_default = float(s["duration_days"])
    cat_default = next((c.replace("category_name_", "") for c in clean.columns if c.startswith("category_name_") and s.get(c, 0) == 1), "Games")
    ctry_default = next((c.replace("country_", "") for c in clean.columns if c.startswith("country_") and s.get(c, 0) == 1), "US")
    has_video_default = int(s.get("has_video", 0))
    dow_default = int(s.get("launch_dow", 1))
    hour_default = int(s.get("launch_hour", 11))
    month_default = int(s.get("launch_month", 4))
    year_default = int(s.get("launch_year", 2024))
else:
    actual_label = None
    name_default = "My Awesome Project"
    blurb_default = "A 96-page hardback solo journaling RPG with 12 unique scenarios."
    goal_default = 8000.0
    duration_default = 30.0
    cat_default = "Games"
    ctry_default = "US"
    has_video_default = 1
    dow_default = 1
    hour_default = 11
    month_default = 4
    year_default = 2024

name = st.sidebar.text_input("Project name", name_default)
blurb = st.sidebar.text_area("Blurb", blurb_default, height=80)
col1, col2 = st.sidebar.columns(2)
with col1:
    goal_usd = st.number_input("Goal (USD)", min_value=100.0, value=goal_default, step=500.0)
    duration_days = st.number_input(
        "Duration (days)", min_value=1, max_value=90, value=int(duration_default)
    )
    category = st.selectbox(
        "Category",
        CATEGORIES,
        index=CATEGORIES.index(cat_default) if cat_default in CATEGORIES else 0,
    )
    country = st.selectbox(
        "Country",
        COUNTRIES,
        index=COUNTRIES.index(ctry_default) if ctry_default in COUNTRIES else 0,
    )
with col2:
    has_video = st.checkbox("Has video", value=bool(has_video_default))
    launched_dow = st.selectbox(
        "Launch day", list(range(7)), format_func=lambda i: DOW_NAMES[i], index=dow_default
    )
    launched_hour = st.slider("Launch hour", 0, 23, hour_default)
    launched_month = st.selectbox(
        "Launch month",
        list(range(1, 13)),
        format_func=lambda i: MONTH_NAMES[i],
        index=month_default - 1,
    )

predict_btn = st.sidebar.button("Predict", type="primary")

# ---- main panel -----------------------------------------------------------
if predict_btn:
    out = rec.predict(
        {
            "goal_usd": goal_usd,
            "duration_days": duration_days,
            "category": category,
            "country": country,
            "has_video": int(has_video),
            "blurb": blurb,
            "launched_dow": int(launched_dow),
            "launched_hour": int(launched_hour),
            "launched_month": int(launched_month),
            "launch_year": year_default,
        }
    )

    proba = out["probability"]
    color = "#0e9f6e" if proba >= 0.6 else ("#d97706" if proba >= 0.4 else "#dc2626")
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:10px;background:{color}22;border-left:6px solid {color};">
        <div style="font-size:14px;color:#444">Predicted probability of success</div>
        <div style="font-size:38px;font-weight:700;color:{color}">{proba:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if actual_label is not None:
        actual_str = "Successful" if actual_label == 1 else "Failed"
        st.caption(f"Actual outcome of this held-out campaign: **{actual_str}**")

    st.subheader("Recommendations")
    if not out["recommendations"]:
        st.success("No major weaknesses detected — your campaign profile looks strong.")
    else:
        for i, r in enumerate(out["recommendations"], 1):
            st.markdown(f"**{i}. {r['message']}**")
            st.caption(
                f"feature={r['feature']}  ·  SHAP={r['shap']:+.3f} (negative = hurting your prediction)"
            )

    st.subheader("Top SHAP contributors")
    shap_df = pd.DataFrame(out["shap_top"])
    shap_df = shap_df[["feature", "value", "shap"]].copy()
    shap_df["direction"] = np.where(shap_df["shap"] >= 0, "+ (helping)", "− (hurting)")
    st.dataframe(shap_df, hide_index=True, use_container_width=True)
    st.caption(
        "SHAP attribution = how much each feature shifted the prediction (in log-odds) "
        "from the base rate. Positive helps, negative hurts."
    )

    st.markdown(
        "*Recommendations are correlational, derived from patterns in 334k historical "
        "campaigns. They reflect what tends to be associated with success — they are "
        "not causal guarantees.*"
    )
else:
    st.info("Fill in your campaign details in the sidebar and press Predict.")
    st.markdown(
        "**About this tool**\n\n"
        "- Trains a tabular XGBoost on creator-controllable features only "
        "(`is_spotlight` and `is_staff_pick` are excluded — those are post- or "
        "during-campaign signals).\n"
        "- Recommendations come from SHAP attributions: only features pushing your "
        "prediction *down* are surfaced.\n"
        "- Test set is post-2022; validation AUROC ≈ 0.78 on tabular features alone."
    )
