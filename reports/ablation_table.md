# Ablation results (test set, post-2022)

| model | n_features | auroc | ap | brier | logloss | accuracy | fit_time_s |
|---|---|---|---|---|---|---|---|
| LR baseline (52 tabular) | 52 | 0.7729 | 0.8930 | 0.1647 | 0.5004 | 0.7616 | 0.5641 |
| XGBoost (52 tabular) | 52 | 0.8264 | 0.9178 | 0.1395 | 0.4346 | 0.8025 | 2.9297 |
| XGBoost full minus Trends | 596112 | 0.8691 | 0.9381 | 0.1231 | 0.3901 | 0.8268 | 909.5175 |
| XGBoost full (596k features) | 596114 | 0.8681 | 0.9374 | 0.1255 | 0.3966 | 0.8222 | 454.7740 |
