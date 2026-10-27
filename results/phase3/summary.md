# Phase 3 summary (detectors and drift typing)

All numbers: tuning-side seeds only (tuned on 1001-1003, validated on 1004-1010, typing leave-one-seed-out over 1001-1010),
frozen local models, v2 protocol, `med` severity unless stated. The 10 evaluation seeds were NOT used.
Details: `detector_report.md`, `typing_report_la5.md` / `typing_report_la10.md`; earlier iterations are kept in
`detector_report_run2_before_D2s.md` and `typing_v1_before_rise_features/`.

## Detection (validation seeds; false-alarm bound 1 per 1,000 clean windows)

| detector | labels / 1000 windows | recall | placebo recall | FA / 1000 | mean delay (rounds, detected) |
|---|---|---|---|---|---|
| D1 ADWIN on error | 1000 | 0.20 | 0.00 | 0.25 | 11.2 |
| **D2 Page-Hinkley on error** | 1000 | **0.42** | 0.07 | 0.39 | 9.2 |
| D2c class-conditioned error PH | 1000 | 0.41 | 0.07 | 0.39 | 9.2 |
| D2s PH on error, 20% of windows labelled | 200 | 0.26 | 0.00 | 0.23 | 10.7 |
| D3 PH on entropy (label-free) | 0 | 0.25 | 0.16 | 1.24 (bound violated) | 9.6 |
| D4 class-matched MMD (label-free) | 0 | 0.13 | 0.03 | 1.53 (bound violated) | 11.8 |
| D5 two-level (+ sentinel) | 376 | 0.19 | 0.04 | 0.81 | 11.4 |
| D5ns two-level, no sentinel | 207 | 0.17 | 0.04 | 0.83 | 11.1 |

Reading: supervised error is the only signal clearly above chance (+0.35 over placebo for D2). Label-free detectors are
barely above chance at this stream length and false-alarm bound. The two-level design does NOT beat a plain sparse-label
detector at a comparable label budget (D2s 0.26 at 200 labels vs D5 0.19 at 376). Recall rises with severity
(D2: 0.19 / 0.42 / 0.65 for low / med / high).

## Drift-type classification (4-way, identifiable episodes, leave-one-seed-out, look-ahead 5 rounds)

| alarm source | majority acc | rule acc / macro-F1 | logreg | gboost acc / macro-F1 |
|---|---|---|---|---|
| oracle (onset + 2 rounds) | 0.444 | 0.349 / 0.278 | 0.508 / 0.529 | 0.633 / 0.589 |
| detected (D2) | 0.511 | 0.375 / 0.251 | 0.345 / 0.300 | 0.462 / 0.323 |

Reading: typing is weak. With detected alarms accuracy does not beat the majority class (sudden); macro-F1 0.32 shows some
minority-class signal. The oracle numbers are inflated for `recurring` (recall 1.00) because the presence of an earlier
alarm identifies a re-occurrence in perfect-alarm streams; that shortcut disappears once false alarms enter. The a-priori
rule classifier is worse than the learned ones. A longer look-ahead (10 rounds) and retrospective rise-time features did
not help.
