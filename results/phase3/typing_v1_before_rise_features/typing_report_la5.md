# Phase 3: drift-type classification

Seeds 1001-1010 (tuning side), leave-one-seed-out. Look-ahead 5 rounds after the alarm; learned models do not see `n_prior`. Identifiable 4-way score excludes the first onset of recurring drifts.

## Alarm source: oracle

1890 identifiable episodes {'sudden': 840, 'gradual': 510, 'recurring': 270, 'feature_level': 270}; 270 recurring_first episodes.

| classifier | accuracy | macro-F1 | recall sudden | recall gradual | recall recurring | recall feature_level |
|---|---|---|---|---|---|---|
| majority | 0.444 | 0.154 | 1.00 | 0.00 | 0.00 | 0.00 |
| rule | 0.349 | 0.278 | 0.39 | 0.50 | 0.05 | 0.23 |
| logreg | 0.504 | 0.521 | 0.40 | 0.61 | 0.88 | 0.24 |
| gboost | 0.641 | 0.596 | 0.79 | 0.48 | 1.00 | 0.12 |

Accuracy by severity (low / med / high) and by mode (virtual / real):

| classifier | low | med | high | virtual | real |
|---|---|---|---|---|---|
| majority | 0.44 | 0.44 | 0.44 | 0.37 | 0.65 |
| rule | 0.32 | 0.35 | 0.38 | 0.36 | 0.32 |
| logreg | 0.44 | 0.49 | 0.58 | 0.48 | 0.56 |
| gboost | 0.60 | 0.65 | 0.67 | 0.58 | 0.80 |

Confusion matrix of gboost (rows = truth, columns = predicted, order sudden, gradual, recurring, feature_level):

    [664, 131, 0, 45]
    [248, 247, 0, 15]
    [1, 0, 269, 0]
    [176, 62, 0, 32]

recurring_first (not identifiable) predicted as: {'rule': {'feature_level': 58, 'gradual': 108, 'sudden': 104}, 'logreg': {'gradual': 105, 'sudden': 100, 'feature_level': 62, 'recurring': 3}, 'gboost': {'feature_level': 15, 'sudden': 220, 'gradual': 35}}

Feature importance (gboost, descriptive): mem_returned 0.33, dF_pre 0.12, err_delta 0.08, dF_ramp 0.07, dF_delta 0.07, mem_cos 0.05

## Alarm source: detected

831 identifiable episodes {'sudden': 425, 'gradual': 218, 'recurring': 88, 'feature_level': 100}; 96 recurring_first episodes.

| classifier | accuracy | macro-F1 | recall sudden | recall gradual | recall recurring | recall feature_level |
|---|---|---|---|---|---|---|
| majority | 0.511 | 0.169 | 1.00 | 0.00 | 0.00 | 0.00 |
| rule | 0.375 | 0.251 | 0.53 | 0.29 | 0.01 | 0.22 |
| logreg | 0.350 | 0.303 | 0.37 | 0.39 | 0.36 | 0.16 |
| gboost | 0.481 | 0.336 | 0.76 | 0.20 | 0.22 | 0.14 |

Accuracy by severity (low / med / high) and by mode (virtual / real):

| classifier | low | med | high | virtual | real |
|---|---|---|---|---|---|
| majority | 0.53 | 0.53 | 0.50 | 0.41 | 0.68 |
| rule | 0.38 | 0.39 | 0.37 | 0.35 | 0.41 |
| logreg | 0.29 | 0.33 | 0.38 | 0.28 | 0.47 |
| gboost | 0.47 | 0.49 | 0.48 | 0.39 | 0.63 |

Confusion matrix of gboost (rows = truth, columns = predicted, order sudden, gradual, recurring, feature_level):

    [323, 72, 17, 13]
    [154, 44, 10, 10]
    [44, 13, 19, 12]
    [53, 28, 5, 14]

recurring_first (not identifiable) predicted as: {'rule': {'sudden': 63, 'gradual': 21, 'feature_level': 12}, 'logreg': {'gradual': 30, 'feature_level': 18, 'recurring': 20, 'sudden': 28}, 'gboost': {'sudden': 64, 'feature_level': 7, 'gradual': 15, 'recurring': 10}}

Feature importance (gboost, descriptive): dF_delta 0.11, ent_delta 0.11, dF_ramp 0.10, dF_post 0.09, dF_slope 0.09, ent_ramp 0.09
