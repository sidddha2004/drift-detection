# Phase 6 confirmatory results (10 evaluation seeds)

Seeds: [7, 11, 13, 17, 19, 23, 42, 123, 999, 2024]. Unit = seed; post-change macro-F1 on drifted clients averaged over the scenario's variants ({'S1': 4, 'S2': 2, 'S3': 2}). E8 = `typed_learned`. Wilcoxon two-sided, bootstrap 10,000, Holm over the 12 family tests.

## Confirmatory family

| H | scenario | comparator | E8 | comparator F1 | diff | 95% CI | p | Holm p | significant |
|---|---|---|---|---|---|---|---|---|---|
| H1 | S1 | frozen_E3b_personal | 0.756 | 0.677 | +0.079 | [+0.055, +0.105] | 0.0020 | 0.0234 | yes |
| H1 | S2 | frozen_E3b_personal | 0.691 | 0.612 | +0.080 | [+0.040, +0.120] | 0.0098 | 0.0781 | no |
| H1 | S3 | frozen_E3b_personal | 0.711 | 0.761 | -0.050 | [-0.074, -0.029] | 0.0020 | 0.0234 | yes |
| H2 | S1 | periodic_E3c_full_fed | 0.756 | 0.710 | +0.046 | [+0.008, +0.085] | 0.1055 | 0.6328 | no |
| H2 | S2 | periodic_E3c_full_fed | 0.691 | 0.635 | +0.057 | [+0.003, +0.109] | 0.0840 | 0.5879 | no |
| H2 | S3 | periodic_E3c_full_fed | 0.711 | 0.744 | -0.033 | [-0.070, +0.002] | 0.1934 | 0.9668 | no |
| H3 | S1 | ph_fedprox_E7a | 0.756 | 0.738 | +0.018 | [-0.008, +0.047] | 0.4922 | 1.0000 | no |
| H3 | S2 | ph_fedprox_E7a | 0.691 | 0.669 | +0.023 | [-0.021, +0.063] | 0.4316 | 1.0000 | no |
| H3 | S3 | ph_fedprox_E7a | 0.711 | 0.763 | -0.052 | [-0.084, -0.025] | 0.0039 | 0.0352 | yes |
| H5 | S1 | ph_head_long | 0.756 | 0.753 | +0.004 | [-0.019, +0.026] | 0.7695 | 1.0000 | no |
| H5 | S2 | ph_head_long | 0.691 | 0.707 | -0.016 | [-0.054, +0.019] | 0.6953 | 1.0000 | no |
| H5 | S3 | ph_head_long | 0.711 | 0.766 | -0.055 | [-0.082, -0.031] | 0.0020 | 0.0234 | yes |

## H4 non-inferiority: ph_full_fed_E5 minus ph_adapter_fed (margin +0.010) and uplink

| scenario | E5 F1 | adapter F1 | E5 - adapter | 95% CI | non-inferior | uplink E5 (kB) | uplink adapter (kB) | reduction |
|---|---|---|---|---|---|---|---|---|
| S1 | 0.704 | 0.710 | -0.005 | [-0.015, +0.005] | yes | 3387 | 127 | 96% |
| S2 | 0.647 | 0.649 | -0.002 | [-0.013, +0.008] | yes | 3888 | 142 | 96% |
| S3 | 0.735 | 0.746 | -0.011 | [-0.022, -0.000] | yes | 2929 | 112 | 96% |

## All systems, post-change macro-F1 (mean over seeds ± std over seeds), uplink per run

| system | S1 F1 | S2 F1 | S3 F1 | S4 F1 | uplink S1 (kB) |
|---|---|---|---|---|---|
| adwin_full_fed_E4 | 0.699 ± 0.021 | 0.642 ± 0.022 | 0.733 ± 0.021 | 0.726 ± 0.039 | 2269 |
| frozen_E3_global | 0.658 ± 0.023 | 0.590 ± 0.029 | 0.737 ± 0.016 | 0.635 ± 0.030 | 0 |
| frozen_E3b_personal | 0.677 ± 0.022 | 0.612 ± 0.031 | 0.761 ± 0.019 | 0.681 ± 0.034 | 0 |
| mode_typed_oracle | 0.779 ± 0.049 | 0.717 ± 0.035 | 0.731 ± 0.020 | 0.811 ± 0.061 | 0 |
| periodic_E3c_full_fed | 0.710 ± 0.026 | 0.635 ± 0.021 | 0.744 ± 0.020 | 0.726 ± 0.024 | 18748 |
| ph_adapter_fed | 0.710 ± 0.021 | 0.649 ± 0.023 | 0.746 ± 0.016 | 0.726 ± 0.022 | 127 |
| ph_align | 0.707 ± 0.027 | 0.616 ± 0.051 | 0.637 ± 0.021 | 0.820 ± 0.043 | 0 |
| ph_fedprox_E7a | 0.738 ± 0.014 | 0.669 ± 0.019 | 0.763 ± 0.010 | 0.761 ± 0.024 | 3174 |
| ph_full_fed_E5 | 0.704 ± 0.015 | 0.647 ± 0.027 | 0.735 ± 0.014 | 0.721 ± 0.027 | 3387 |
| ph_full_local | 0.704 ± 0.022 | 0.647 ± 0.029 | 0.725 ± 0.015 | 0.722 ± 0.035 | 0 |
| ph_head_long | 0.753 ± 0.022 | 0.707 ± 0.027 | 0.766 ± 0.014 | 0.753 ± 0.019 | 0 |
| ph_mode_oracle | 0.783 ± 0.028 | 0.744 ± 0.029 | 0.734 ± 0.019 | 0.819 ± 0.041 | 0 |
| typed_learned | 0.756 ± 0.050 | 0.691 ± 0.077 | 0.711 ± 0.051 | 0.810 ± 0.044 | 0 |
| typed_oracle | 0.779 ± 0.049 | 0.762 ± 0.031 | 0.797 ± 0.018 | 0.863 ± 0.043 | 0 |

## Typing ablations (descriptive)

| comparison | scenario | diff | 95% CI | p (uncorrected) |
|---|---|---|---|---|
| oracle 4-way - oracle 2-way | S1 | +0.000 | [+0.000, +0.000] | 1.0000 |
| oracle 4-way - oracle 2-way | S2 | +0.045 | [+0.026, +0.067] | 0.0020 |
| oracle 4-way - oracle 2-way | S3 | +0.066 | [+0.058, +0.074] | 0.0020 |
| oracle 4-way - oracle 2-way | S4 | +0.052 | [+0.030, +0.075] | 0.0020 |
| deployable E8 - E8 with true mode | S1 | -0.027 | [-0.043, -0.011] | 0.0156 |
| deployable E8 - E8 with true mode | S2 | -0.053 | [-0.092, -0.022] | 0.0156 |
| deployable E8 - E8 with true mode | S3 | -0.023 | [-0.048, -0.000] | 0.1289 |
| deployable E8 - E8 with true mode | S4 | -0.009 | [-0.019, +0.000] | 0.2500 |