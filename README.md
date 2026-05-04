# Kickstarter Pre-Launch Recommender

A pre-launch success-probability prediction system for Kickstarter campaigns,
with SHAP-explained creator recommendations.

- **Live demo:** [huggingface.co/spaces/OD2004/OD](https://huggingface.co/spaces/OD2004/OD) *(private — collaborator access)*
- **Presentation deck:** [`reports/kickstarter_presentation.pptx`](reports/kickstarter_presentation.pptx)

## Headline results (held-out test set, post-2022)

| model | features | test AUROC | test AP | brier | accuracy |
|---|---:|---:|---:|---:|---:|
| LR baseline (tabular) | 52 | 0.7729 | 0.8930 | 0.165 | 0.762 |
| XGBoost (tabular) | 52 | 0.8264 | 0.9178 | 0.140 | 0.802 |
| XGBoost full minus Trends | 596,112 | 0.8691 | 0.9381 | 0.123 | 0.827 |
| XGBoost full (tab + word + char TF-IDF) | 596,114 | 0.8681 | 0.9374 | 0.126 | 0.822 |

**Stepwise contribution:** non-linear ML over LR ≈ +5.4 AUROC pts; TF-IDF text
features ≈ +4.3 pts; **Google Trends adds ≈ 0** on test (val Δ ≈ +0.5 pts; not
preserved on transfer to post-2022 set — a notable null result for the paper).

![ROC curves on post-2022 test set](reports/roc_curves.png)

## Data leak we caught

The original prep included two columns that broke the pre-launch use-case:

- `is_spotlight` had correlation **1.0** with the label. Kickstarter only
  marks campaigns spotlighted *after* they succeed; including it gave 100% AUROC.
- `is_staff_pick` is a Kickstarter curatorial decision, not a creator-controllable
  pre-launch input. Dropped because the recommender targets creators *before* launch.

Both are dropped in `data/X_train_clean.parquet`, `X_test_clean.parquet`,
`X_train_full_clean.npz`, `X_test_full_clean.npz`.

## Project layout

```
kickstarter_project/
  data/        # cleaned matrices + vectorizers + trend lookup (1.5 GB)
  src/         # training + analysis scripts (see below)
  models/      # saved boosters + LR pipeline + proba arrays
  reports/     # ablation table, ROC + calibration figures, fairness CSVs
  app/         # Streamlit demo
  Dockerfile   # ships only the lightweight tabular model
  requirements.txt
```

## Setup

```bash
git clone https://github.com/YLA2ATI/kickstarter-recommender.git
cd kickstarter-recommender
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

The repo ships trained models and reports. Re-running the full pipeline below
requires the raw WebRobots Kickstarter dump (not bundled — see "Honest limitations"
for size). Without raw data you can still run the recommender (`src/recommend.py`)
and the Streamlit demo, both of which only need `models/xgb_tabular.json`.

## Pipeline (run order)

```bash
# from project root, inside venv
python src/sanity_check.py            # confirm data + GPU
python src/audit_features.py          # leak audit
python src/drop_leaks.py              # writes *_clean.parquet / *_clean.npz
python src/train_baseline_lr.py       # LR baseline
python src/train_xgb_full.py          # XGBoost on full sparse (GPU)
python src/train_xgb_ablations.py     # XGB tabular + XGB no-Trends
python src/build_ablation_report.py   # reports/ablation_table.* + plots
python src/fairness_report.py         # per-category + per-country metrics
python src/temporal_drift.py          # multi-window training + drift analysis
python src/threshold_tuning.py        # PR curve, best threshold per objective
python src/lebanon_case_studies.py    # diaspora subgroup case studies
python src/recommend.py               # smoke-test the recommender
streamlit run app/streamlit_app.py    # demo UI
```

## Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Two modes:
1. **Manual** — fill in your campaign details
2. **Real held-out campaign** — load a random post-2022 campaign, predict, then reveal the actual outcome

The recommender translates SHAP values into actionable text, e.g.:

> Your goal of $250,000 may be too high for this profile. Consider lowering it.
> Your campaign length of 60 days is hurting your odds. Most successful campaigns run 21-35 days.
> Add a video. Campaigns without a video underperform meaningfully in this category.

Recommendations only fire when (a) SHAP < -0.05 for that feature *and* (b) the
suggestion makes sense for the current value (e.g., "add a video" only fires if
`has_video=0`).

## Docker

```bash
docker build -t kickstarter-recommender .
docker run -p 8501:8501 kickstarter-recommender
# -> http://localhost:8501
```

The image only ships the tabular XGBoost (compact) and the trends lookup. The
1.5 GB sparse matrices and full-feature model stay out of the image.

## Temporal drift (proposal contribution #2)

Trained XGBoost-tabular on four launch-year windows; scored each on a fixed
forward test set (≥2022).

| Training window | n | Own holdout AUROC | Forward (≥2022) AUROC | Transfer gap |
|---|---:|---:|---:|---:|
| 2010-2014 | 48,657 | 0.8026 | 0.6425 | **+0.1601** |
| 2015-2017 | 86,272 | 0.8084 | 0.7611 | +0.0474 |
| 2018-2020 | 78,847 | 0.8550 | 0.8148 | +0.0402 |
| 2021+     | 120,090| 0.8651 | 0.8923 | -0.0272 |

A model trained on 2010-2014 data is **essentially broken on 2022+** (0.64
AUROC vs 0.86 on its own holdout). Notable feature drift:

- `has_video`: dominant feature 2010-2017 (gain 20-35k), drops to 10k by 2021+ — became table stakes.
- `trend_score`: 0 gain pre-2018 (data unavailable), grows to 14k by 2021+.
- `country_HK`: 0 gain pre-2018 → 12k by 2021+.

![Temporal drift heatmap](reports/temporal_drift_heatmap.png)

See `reports/temporal_drift_*.csv,png` and `reports/temporal_drift_summary.json`.

## Threshold tuning

| Objective | Threshold | Precision | Recall | F1 | Acc |
|---|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.82 | 0.93 | 0.87 | 0.80 |
| Max F1 | 0.44 | 0.81 | 0.95 | 0.87 | 0.80 |
| P ≥ 0.85, max R | 0.64 | 0.85 | 0.86 | 0.85 | 0.79 |

Threshold 0.64 is the balanced sweet-spot — both precision and recall above 0.85.
See `reports/pr_curve.png`, `threshold_sweep.png`, `threshold_best_picks.csv`.

## Lebanon diaspora subgroup (case study, n=49)

Subgroup base rate 65% (≈ overall 64%). Subgroup AUROC = 0.807 — comparable to
global tabular XGB AUROC of 0.826 (small N caveats apply).

See `reports/lebanon_case_studies.{csv,md}` for per-campaign predictions,
high-confidence misses, and correctly flagged risks.

## Honest limitations

- **All recommendations are correlational, not causal** — patterns from 334k
  historical campaigns. We cannot guarantee changing a feature would change the outcome.
- **Google Trends data starts 2019.** Pre-2019 campaigns have `has_trend=0` and
  `trend_score=50` (neutral fill). The `has_trend` flag lets the model treat
  missingness separately from a real Trend score of 50.
- **Lebanese diaspora subset is ~49 campaigns.** Use case studies, not subgroup metrics.
- **Test set has higher base rate** (72.5% positive) than train (61.2%). Post-2022
  campaigns succeed more often in this dataset; could be platform shift, could be
  WebRobots scraping bias toward already-resolved newer campaigns.
- **Per-group AUROC varies 0.69–0.90.** The model performs unevenly across
  category and country. See `reports/fairness_by_*.csv`.

## Future work

- Per-category calibration (per-group AUROC spread of 0.69–0.90 means a single global threshold is the wrong call for some segments).
- Re-test the Google Trends signal with a query-specific lookup rather than the generic one.
- Causal layer — replace SHAP-based recommendations with an uplift model on near-duplicate campaign pairs that differ in only one feature.
- Productionize a retraining cadence (≥ every 12 months given observed temporal drift).
