# Phase 3: drift-type classification

Seeds 1001-1010 (tuning side), leave-one-seed-out. Look-ahead 10 rounds after the alarm; learned models do not see `n_prior`. Identifiable 4-way score excludes the first onset of recurring drifts.

## Alarm source: oracle

1890 identifiable episodes {'sudden': 840, 'gradual': 510, 'recurring': 270, 'feature_level': 270}; 270 recurring_first episodes.

| classifier | accuracy | macro-F1 | recall sudden | recall gradual | recall recurring | recall feature_level |
|---|---|---|---|---|---|---|
| majority | 0.444 | 0.154 | 1.00 | 0.00 | 0.00 | 0.00 |
| rule | 0.405 | 0.310 | 0.49 | 0.55 | 0.04 | 0.24 |
| logreg | 0.492 | 0.520 | 0.36 | 0.60 | 0.86 | 0.32 |
| gboost | 0.625 | 0.567 | 0.76 | 0.51 | 1.00 | 0.05 |

Accuracy by severity (low / med / high) and by mode (virtual / real):

| classifier | low | med | high | virtual | real |
|---|---|---|---|---|---|
| majority | 0.44 | 0.44 | 0.44 | 0.37 | 0.65 |
| rule | 0.39 | 0.39 | 0.43 | 0.39 | 0.44 |
| logreg | 0.39 | 0.48 | 0.60 | 0.48 | 0.52 |
| gboost | 0.58 | 0.62 | 0.68 | 0.57 | 0.77 |

Confusion matrix of gboost (rows = truth, columns = predicted, order sudden, gradual, recurring, feature_level):

    [641, 166, 0, 33]
    [235, 258, 0, 17]
    [0, 0, 270, 0]
    [194, 63, 0, 13]

recurring_first (not identifiable) predicted as: {'rule': {'feature_level': 45, 'sudden': 175, 'gradual': 50}, 'logreg': {'gradual': 99, 'sudden': 92, 'feature_level': 79}, 'gboost': {'sudden': 210, 'gradual': 51, 'feature_level': 9}}

Feature importance (gboost, descriptive): mem_returned 0.31, dF_pre 0.14, dF_ramp 0.08, mem_cos 0.07, dF_slope 0.06, ent_ramp 0.05

## Alarm source: detected

828 identifiable episodes {'sudden': 425, 'gradual': 218, 'recurring': 85, 'feature_level': 100}; 96 recurring_first episodes.

| classifier | accuracy | macro-F1 | recall sudden | recall gradual | recall recurring | recall feature_level |
|---|---|---|---|---|---|---|
| majority | 0.513 | 0.170 | 1.00 | 0.00 | 0.00 | 0.00 |
| rule | 0.414 | 0.250 | 0.64 | 0.26 | 0.00 | 0.17 |
| logreg | 0.329 | 0.295 | 0.32 | 0.36 | 0.41 | 0.23 |
| gboost | 0.476 | 0.336 | 0.70 | 0.33 | 0.25 | 0.06 |

Accuracy by severity (low / med / high) and by mode (virtual / real):

| classifier | low | med | high | virtual | real |
|---|---|---|---|---|---|
| majority | 0.53 | 0.53 | 0.50 | 0.41 | 0.68 |
| rule | 0.43 | 0.42 | 0.41 | 0.38 | 0.47 |
| logreg | 0.23 | 0.31 | 0.37 | 0.28 | 0.41 |
| gboost | 0.53 | 0.47 | 0.47 | 0.39 | 0.61 |

Confusion matrix of gboost (rows = truth, columns = predicted, order sudden, gradual, recurring, feature_level):

    [296, 78, 26, 25]
    [128, 71, 8, 11]
    [48, 9, 21, 7]
    [70, 20, 4, 6]

recurring_first (not identifiable) predicted as: {'rule': {'sudden': 68, 'feature_level': 13, 'gradual': 15}, 'logreg': {'recurring': 47, 'feature_level': 18, 'gradual': 12, 'sudden': 19}, 'gboost': {'feature_level': 15, 'sudden': 63, 'recurring': 7, 'gradual': 11}}

Feature importance (gboost, descriptive): ent_delta 0.11, dF_post 0.10, dF_slope 0.10, err_delta 0.09, dF_pre 0.09, dF_ramp 0.09
