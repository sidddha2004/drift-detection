# Phase 4 summary: streaming FL, adaptation responses and drift typing (design look)

Tuning-side seeds only (1001-1003), `med` severity, 8 clients, 60 rounds, deployed model = E3b personalised model, immediate labels. Means ± std over seeds; with 3 seeds nothing here is a confirmatory test.

### S1 real_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.622 ± 0.016 | 0.945 | 0.323 | 90.0 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.608 ± 0.014 | 0.934 | 0.346 | 90.0 | nan | 0.878 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.618 ± 0.030 | 0.945 | 0.414 | 90.0 | nan | 0.940 | 4261 | 6.7 |
| ph_full_fed_E5 | 0.626 ± 0.002 | 0.945 | 0.379 | 90.0 | nan | 0.936 | 5255 | 8.3 |
| ph_align | 0.418 ± 0.064 | 0.945 | 0.698 | 90.0 | nan | 0.929 | 0 | 9.0 |
| ph_head_long | 0.734 ± 0.010 | 0.945 | 0.335 | 83.7 | nan | 0.939 | 0 | 4.7 |
| typed_learned | 0.644 ± 0.096 | 0.945 | 0.464 | 84.1 | nan | 0.932 | 0 | 5.7 |
| ph_mode_oracle | 0.730 ± 0.009 | 0.945 | 0.339 | 84.1 | nan | 0.928 | 0 | 5.0 |
| oracle_align | 0.500 ± 0.080 | 0.945 | 0.470 | 90.0 | nan | 0.945 | 0 | 3.0 |
| typed_oracle | 0.714 ± 0.024 | 0.945 | 0.336 | 90.0 | nan | 0.945 | 0 | 3.0 |

### S1 virtual_gain_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.762 ± 0.067 | 0.945 | 0.183 | 80.0 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.755 ± 0.029 | 0.934 | 0.278 | 75.6 | nan | 0.878 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.745 ± 0.082 | 0.945 | 0.252 | 85.4 | nan | 0.943 | 2556 | 4.0 |
| ph_full_fed_E5 | 0.747 ± 0.045 | 0.945 | 0.316 | 90.0 | nan | 0.934 | 4474 | 7.0 |
| ph_align | 0.826 ± 0.040 | 0.945 | 0.232 | 58.0 | nan | 0.929 | 0 | 5.7 |
| ph_head_long | 0.797 ± 0.045 | 0.945 | 0.197 | 76.1 | nan | 0.939 | 0 | 3.7 |
| typed_learned | 0.813 ± 0.033 | 0.945 | 0.190 | 74.9 | nan | 0.932 | 0 | 4.0 |
| ph_mode_oracle | 0.813 ± 0.033 | 0.945 | 0.190 | 74.9 | nan | 0.928 | 0 | 4.3 |
| oracle_align | 0.811 ± 0.088 | 0.945 | 0.264 | 33.4 | nan | 0.945 | 0 | 3.0 |
| typed_oracle | 0.811 ± 0.088 | 0.945 | 0.264 | 33.4 | nan | 0.945 | 0 | 3.0 |

### S1 virtual_offset_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.892 ± 0.014 | 0.945 | 0.053 | 40.0 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.861 ± 0.011 | 0.934 | 0.116 | 58.9 | nan | 0.875 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.877 ± 0.031 | 0.945 | 0.111 | 50.0 | nan | 0.942 | 1278 | 2.0 |
| ph_full_fed_E5 | 0.877 ± 0.016 | 0.945 | 0.090 | 45.1 | nan | 0.933 | 2556 | 4.0 |
| ph_align | 0.883 ± 0.039 | 0.945 | 0.137 | 29.9 | nan | 0.929 | 0 | 4.3 |
| ph_head_long | 0.892 ± 0.019 | 0.945 | 0.078 | 45.0 | nan | 0.939 | 0 | 2.7 |
| typed_learned | 0.885 ± 0.034 | 0.945 | 0.076 | 35.6 | nan | 0.932 | 0 | 2.7 |
| ph_mode_oracle | 0.885 ± 0.034 | 0.945 | 0.076 | 35.6 | nan | 0.928 | 0 | 3.0 |
| oracle_align | 0.836 ± 0.103 | 0.945 | 0.164 | 43.0 | nan | 0.945 | 0 | 3.0 |
| typed_oracle | 0.836 ± 0.103 | 0.945 | 0.164 | 43.0 | nan | 0.945 | 0 | 3.0 |

### S1 virtual_rotation_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.682 ± 0.042 | 0.945 | 0.263 | 90.0 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.735 ± 0.040 | 0.934 | 0.311 | 82.2 | nan | 0.878 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.714 ± 0.031 | 0.945 | 0.291 | 90.0 | nan | 0.939 | 2343 | 3.7 |
| ph_full_fed_E5 | 0.692 ± 0.031 | 0.945 | 0.365 | 86.4 | nan | 0.933 | 5752 | 9.0 |
| ph_align | 0.816 ± 0.038 | 0.945 | 0.265 | 54.9 | nan | 0.929 | 0 | 6.0 |
| ph_head_long | 0.735 ± 0.043 | 0.945 | 0.276 | 88.8 | nan | 0.938 | 0 | 5.0 |
| typed_learned | 0.817 ± 0.032 | 0.945 | 0.281 | 54.7 | nan | 0.932 | 0 | 4.3 |
| ph_mode_oracle | 0.831 ± 0.038 | 0.945 | 0.281 | 46.7 | nan | 0.928 | 0 | 4.7 |
| oracle_align | 0.789 ± 0.088 | 0.945 | 0.333 | 70.7 | nan | 0.945 | 0 | 3.0 |
| typed_oracle | 0.789 ± 0.088 | 0.945 | 0.333 | 70.7 | nan | 0.945 | 0 | 3.0 |

### S2 real_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.632 ± 0.017 | 0.945 | 0.330 | 84.4 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.618 ± 0.009 | 0.932 | 0.353 | 84.4 | nan | 0.873 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.654 ± 0.031 | 0.945 | 0.369 | 84.4 | nan | 0.944 | 3196 | 5.0 |
| ph_full_fed_E5 | 0.626 ± 0.023 | 0.945 | 0.382 | 84.4 | nan | 0.929 | 5326 | 8.3 |
| ph_align | 0.485 ± 0.037 | 0.945 | 0.647 | 84.4 | nan | 0.929 | 0 | 8.3 |
| ph_head_long | 0.773 ± 0.018 | 0.945 | 0.331 | 72.4 | nan | 0.941 | 0 | 5.0 |
| typed_learned | 0.680 ± 0.100 | 0.945 | 0.432 | 78.8 | nan | 0.931 | 0 | 5.3 |
| ph_mode_oracle | 0.767 ± 0.023 | 0.945 | 0.335 | 74.3 | nan | 0.928 | 0 | 5.0 |
| oracle_align | 0.534 ± 0.058 | 0.945 | 0.442 | 84.4 | nan | 0.945 | 0 | 3.0 |
| typed_oracle | 0.768 ± 0.015 | 0.945 | 0.316 | 74.6 | nan | 0.945 | 0 | 3.0 |

### S2 virtual_rotation_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.703 ± 0.038 | 0.945 | 0.265 | 75.4 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.753 ± 0.045 | 0.932 | 0.281 | 82.2 | nan | 0.879 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.725 ± 0.033 | 0.945 | 0.340 | 75.4 | nan | 0.938 | 2983 | 4.7 |
| ph_full_fed_E5 | 0.701 ± 0.041 | 0.945 | 0.378 | 84.4 | nan | 0.935 | 5255 | 8.3 |
| ph_align | 0.838 ± 0.029 | 0.945 | 0.265 | 42.8 | nan | 0.929 | 0 | 6.3 |
| ph_head_long | 0.742 ± 0.047 | 0.945 | 0.283 | 75.4 | nan | 0.943 | 0 | 5.7 |
| typed_learned | 0.846 ± 0.030 | 0.945 | 0.270 | 35.2 | nan | 0.931 | 0 | 4.7 |
| ph_mode_oracle | 0.845 ± 0.029 | 0.945 | 0.270 | 36.9 | nan | 0.928 | 0 | 5.3 |
| oracle_align | 0.760 ± 0.055 | 0.945 | 0.216 | 84.4 | nan | 0.945 | 0 | 3.0 |
| typed_oracle | 0.907 ± 0.019 | 0.945 | 0.228 | 16.0 | nan | 0.945 | 0 | 3.0 |

### S3 real_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.784 ± 0.016 | 0.945 | 0.323 | 15.0 | 0.945 | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.765 ± 0.011 | 0.934 | 0.340 | 47.2 | 0.922 | 0.877 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.728 ± 0.013 | 0.945 | 0.369 | 58.8 | 0.852 | 0.941 | 3196 | 5.0 |
| ph_full_fed_E5 | 0.732 ± 0.015 | 0.945 | 0.381 | 65.2 | 0.832 | 0.935 | 5752 | 9.0 |
| ph_align | 0.544 ± 0.100 | 0.945 | 0.720 | 81.4 | 0.613 | 0.929 | 0 | 7.7 |
| ph_head_long | 0.754 ± 0.008 | 0.945 | 0.332 | 72.9 | 0.818 | 0.940 | 0 | 5.0 |
| typed_learned | 0.685 ± 0.097 | 0.945 | 0.441 | 48.3 | 0.744 | 0.932 | 0 | 5.7 |
| ph_mode_oracle | 0.751 ± 0.009 | 0.945 | 0.328 | 40.7 | 0.819 | 0.928 | 0 | 5.7 |
| oracle_align | 0.622 ± 0.042 | 0.945 | 0.645 | 80.0 | 0.718 | 0.945 | 0 | 9.0 |
| typed_oracle | 0.796 ± 0.011 | 0.945 | 0.336 | 23.3 | 0.926 | 0.945 | 0 | 18.0 |

### S3 virtual_rotation_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.814 ± 0.017 | 0.945 | 0.263 | 15.0 | 0.945 | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.806 ± 0.012 | 0.934 | 0.319 | 22.8 | 0.917 | 0.878 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.792 ± 0.024 | 0.945 | 0.280 | 41.7 | 0.895 | 0.940 | 2343 | 3.7 |
| ph_full_fed_E5 | 0.794 ± 0.014 | 0.945 | 0.336 | 48.3 | 0.891 | 0.936 | 3835 | 6.0 |
| ph_align | 0.755 ± 0.055 | 0.945 | 0.407 | 58.0 | 0.779 | 0.929 | 0 | 7.3 |
| ph_head_long | 0.815 ± 0.029 | 0.945 | 0.277 | 30.0 | 0.912 | 0.939 | 0 | 4.0 |
| typed_learned | 0.785 ± 0.041 | 0.945 | 0.390 | 47.0 | 0.828 | 0.932 | 0 | 5.3 |
| ph_mode_oracle | 0.777 ± 0.034 | 0.945 | 0.393 | 53.7 | 0.817 | 0.928 | 0 | 6.0 |
| oracle_align | 0.764 ± 0.016 | 0.945 | 0.394 | 66.6 | 0.714 | 0.945 | 0 | 9.0 |
| typed_oracle | 0.845 ± 0.039 | 0.945 | 0.356 | 13.9 | 0.914 | 0.945 | 0 | 18.0 |

### S4 ankle_gain_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.724 ± 0.050 | 0.945 | 0.221 | 90.0 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.706 ± 0.030 | 0.934 | 0.331 | 90.0 | nan | 0.872 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.752 ± 0.028 | 0.945 | 0.263 | 87.2 | nan | 0.941 | 2983 | 4.7 |
| ph_full_fed_E5 | 0.762 ± 0.042 | 0.945 | 0.258 | 80.7 | nan | 0.933 | 4900 | 7.7 |
| ph_align | 0.841 ± 0.035 | 0.945 | 0.221 | 47.7 | nan | 0.929 | 0 | 5.7 |
| ph_head_long | 0.765 ± 0.033 | 0.945 | 0.233 | 86.4 | nan | 0.939 | 0 | 4.3 |
| typed_learned | 0.826 ± 0.040 | 0.945 | 0.221 | 51.1 | nan | 0.932 | 0 | 4.0 |
| ph_mode_oracle | 0.826 ± 0.040 | 0.945 | 0.221 | 51.1 | nan | 0.928 | 0 | 4.3 |
| oracle_align | 0.830 ± 0.065 | 0.945 | 0.286 | 42.1 | nan | 0.945 | 0 | 3.0 |
| typed_oracle | 0.909 ± 0.036 | 0.945 | 0.222 | 31.6 | nan | 0.945 | 0 | 3.0 |

### S4 chest_rotation_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.733 ± 0.047 | 0.945 | 0.212 | 90.0 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.758 ± 0.009 | 0.934 | 0.279 | 86.7 | nan | 0.880 | 18748 | 88.0 |
| adwin_full_fed_E4 | 0.746 ± 0.048 | 0.945 | 0.241 | 85.4 | nan | 0.938 | 2130 | 3.3 |
| ph_full_fed_E5 | 0.742 ± 0.062 | 0.945 | 0.310 | 85.3 | nan | 0.936 | 4616 | 7.3 |
| ph_align | 0.844 ± 0.017 | 0.945 | 0.220 | 41.7 | nan | 0.929 | 0 | 6.7 |
| ph_head_long | 0.767 ± 0.064 | 0.945 | 0.242 | 78.8 | nan | 0.942 | 0 | 5.0 |
| typed_learned | 0.849 ± 0.026 | 0.945 | 0.228 | 40.8 | nan | 0.932 | 0 | 4.0 |
| ph_mode_oracle | 0.853 ± 0.027 | 0.945 | 0.228 | 36.9 | nan | 0.928 | 0 | 4.3 |
| oracle_align | 0.834 ± 0.077 | 0.945 | 0.281 | 51.6 | nan | 0.945 | 0 | 3.0 |
| typed_oracle | 0.899 ± 0.018 | 0.945 | 0.262 | 31.6 | nan | 0.945 | 0 | 3.0 |

## Typed (oracle) vs fixed responses, post-change macro-F1 per variant

| variant | typed_oracle | best fixed (name) | typed - best fixed | frozen E3b |
|---|---|---|---|---|
## Deployable (detector-triggered) systems, post-change macro-F1 (3 tuning seeds, immediate labels)

| variant | frozen_E3b_personal | ph_full_fed_E5 | ph_align | ph_head_long | typed_learned | typed_learned_conf80 | ph_mode_oracle |
|---|---|---|---|---|---|---|---|
| S1 real_partial | 0.622 | 0.626 | 0.418 | 0.734 | 0.644 | - | 0.730 |
| S1 virtual_gain_partial | 0.762 | 0.747 | 0.826 | 0.797 | 0.813 | - | 0.813 |
| S1 virtual_offset_partial | 0.892 | 0.877 | 0.883 | 0.892 | 0.885 | - | 0.885 |
| S1 virtual_rotation_partial | 0.682 | 0.692 | 0.816 | 0.735 | 0.817 | - | 0.831 |
| S2 real_partial | 0.632 | 0.626 | 0.485 | 0.773 | 0.680 | - | 0.767 |
| S2 virtual_rotation_partial | 0.703 | 0.701 | 0.838 | 0.742 | 0.846 | - | 0.845 |
| S3 real_partial | 0.784 | 0.732 | 0.544 | 0.754 | 0.685 | - | 0.751 |
| S3 virtual_rotation_partial | 0.814 | 0.794 | 0.755 | 0.815 | 0.785 | - | 0.777 |
| S4 ankle_gain_partial | 0.724 | 0.762 | 0.841 | 0.765 | 0.826 | - | 0.826 |
| S4 chest_rotation_partial | 0.733 | 0.742 | 0.844 | 0.767 | 0.849 | - | 0.853 |
| **mean (variants common to all)** | **0.735** | **0.730** | **0.725** | **0.777** | **0.783** | - | **0.808** |

10 variants common to all columns. typed_* use a mode classifier (virtual / real) after a 5-round typing delay; ph_mode_oracle uses the true mode; conf80 applies alignment only if the classifier is >= 80% sure the drift is virtual.
