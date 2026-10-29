# Phase 3: detector tuning and validation

Tuned on seeds [1001, 1002, 1003] (S0 + all S1-S4 variants, `med` severity, fixed timing) under the bound <= 1.0 false alarm per 1,000 clean windows; validated on seeds [1004, 1005, 1006, 1007, 1008, 1009, 1010]. Horizon 20 rounds; a miss costs the full horizon in the delay objective.

## Tuned configurations

| detector | config | feasible on tuning | tuning delay (rounds) | tuning recall | tuning FA/1000 |
|---|---|---|---|---|---|
| D1_adwin_err | `{'delta': 0.3}` | True | 17.93 | 0.22 | 0.51 |
| D2_ph_err | `{'delta': 0.05, 'lam': 8}` | True | 15.86 | 0.44 | 0.69 |
| D2c_ph_err_class_cond | `{'delta': 0.05, 'lam': 8}` | True | 15.80 | 0.43 | 0.66 |
| D2s_ph_err_20pct_labels | `{'delta': 0.02, 'lam': 2}` | True | 17.69 | 0.24 | 0.54 |
| D3_ph_entropy | `{'delta': 1.0, 'lam': 160}` | False | 16.64 | 0.28 | 1.37 |
| D4_mmd | `{'q': 0.999, 'persist': 5}` | True | 19.10 | 0.13 | 0.91 |
| D5_two_level | `{'ent_lam': 80, 'q': 0.999, 'sent_lam': 4, 'alpha_b': 0.001}` | False | 17.48 | 0.26 | 1.00 |
| D5ns_two_level_no_sentinel | `{'ent_lam': 80, 'q': 0.999, 'alpha_b': 0.001}` | False | 17.48 | 0.26 | 1.00 |

## Validation (unseen seeds), `med` severity, all drifted variants

| detector | recall | mean delay (rounds, detected only) | objective delay | FA / 1000 clean windows | redundant alarms | labels / 1000 windows |
|---|---|---|---|---|---|---|
| D1_adwin_err | 0.20 | 11.2 | 18.2 | 0.25 (20/81156) | 104 | 1000 |
| D2_ph_err | 0.42 | 9.2 | 15.5 | 0.39 (32/81156) | 128 | 1000 |
| D2c_ph_err_class_cond | 0.41 | 9.2 | 15.6 | 0.39 (32/81156) | 127 | 1000 |
| D2s_ph_err_20pct_labels | 0.26 | 10.7 | 17.6 | 0.23 (19/81156) | 124 | 200 |
| D3_ph_entropy | 0.25 | 9.6 | 17.4 | 1.24 (101/81156) | 85 | 0 |
| D4_mmd | 0.13 | 11.8 | 19.0 | 1.53 (124/81156) | 59 | 0 |
| D5_two_level | 0.19 | 11.4 | 18.4 | 0.81 (66/81156) | 112 | 376 |
| D5ns_two_level_no_sentinel | 0.17 | 11.1 | 18.5 | 0.83 (67/81156) | 95 | 207 |

## Placebo recall (chance level): the same onsets applied to the CLEAN S0 stream of the same client

A detector with no information about the drift still scores this recall just by alarming at random times inside the 20-round horizon.

| detector | recall on drifted streams (med) | placebo recall | recall above chance |
|---|---|---|---|
| D1_adwin_err | 0.20 | 0.00 | +0.20 |
| D2_ph_err | 0.42 | 0.07 | +0.35 |
| D2c_ph_err_class_cond | 0.41 | 0.07 | +0.34 |
| D2s_ph_err_20pct_labels | 0.26 | 0.00 | +0.25 |
| D3_ph_entropy | 0.25 | 0.16 | +0.09 |
| D4_mmd | 0.13 | 0.03 | +0.10 |
| D5_two_level | 0.19 | 0.04 | +0.14 |
| D5ns_two_level_no_sentinel | 0.17 | 0.04 | +0.13 |

## Validation by drift mode (recall / delay rounds)

| detector | virtual | real |
|---|---|---|
| D1_adwin_err | 0.14 / 10.9 | 0.36 / 11.5 |
| D2_ph_err | 0.33 / 9.8 | 0.66 / 8.5 |
| D2c_ph_err_class_cond | 0.32 / 9.9 | 0.64 / 8.3 |
| D2s_ph_err_20pct_labels | 0.21 / 10.7 | 0.36 / 10.8 |
| D3_ph_entropy | 0.28 / 9.2 | 0.15 / 11.6 |
| D4_mmd | 0.16 / 12.0 | 0.04 / 10.0 |
| D5_two_level | 0.20 / 11.8 | 0.15 / 10.0 |
| D5ns_two_level_no_sentinel | 0.20 / 11.9 | 0.11 / 7.3 |

## Validation by ground-truth type (recall / delay rounds)

| detector | sudden | gradual | recurring | feature_level |
|---|---|---|---|---|
| D1_adwin_err | 0.26 / 11.0 | 0.08 / 16.2 | 0.23 / 9.9 | 0.17 / 11.0 |
| D2_ph_err | 0.51 / 9.2 | 0.37 / 12.6 | 0.36 / 5.8 | 0.35 / 9.8 |
| D2c_ph_err_class_cond | 0.51 / 9.1 | 0.35 / 12.5 | 0.33 / 5.6 | 0.37 / 10.0 |
| D2s_ph_err_20pct_labels | 0.31 / 11.3 | 0.27 / 14.3 | 0.22 / 5.4 | 0.14 / 10.7 |
| D3_ph_entropy | 0.24 / 9.8 | 0.26 / 10.2 | 0.21 / 9.0 | 0.29 / 9.0 |
| D4_mmd | 0.15 / 12.3 | 0.15 / 13.3 | 0.10 / 8.8 | 0.06 / 11.5 |
| D5_two_level | 0.19 / 12.5 | 0.18 / 13.2 | 0.19 / 8.5 | 0.16 / 10.6 |
| D5ns_two_level_no_sentinel | 0.17 / 11.9 | 0.18 / 13.0 | 0.18 / 8.7 | 0.16 / 10.6 |

## Validation severity sweep (recall / FA per 1000 on the same streams)

| detector | low | med | high |
|---|---|---|---|
| D1_adwin_err | 0.08 / 0.09 | 0.20 / 0.27 | 0.39 / 0.43 |
| D2_ph_err | 0.19 / 0.46 | 0.42 / 0.38 | 0.65 / 0.38 |
| D2c_ph_err_class_cond | 0.19 / 0.43 | 0.41 / 0.38 | 0.64 / 0.39 |
| D2s_ph_err_20pct_labels | 0.08 / 0.19 | 0.26 / 0.24 | 0.47 / 0.21 |
| D3_ph_entropy | 0.17 / 1.20 | 0.25 / 1.20 | 0.41 / 1.21 |
| D4_mmd | 0.04 / 1.49 | 0.13 / 1.54 | 0.35 / 1.69 |
| D5_two_level | 0.06 / 0.75 | 0.19 / 0.82 | 0.40 / 0.86 |
| D5ns_two_level_no_sentinel | 0.06 / 0.75 | 0.17 / 0.83 | 0.38 / 0.85 |