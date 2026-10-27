# Phase 4 summary: streaming FL, adaptation responses and drift typing (design look)

Tuning-side seeds only (1001-1003), `med` severity, 8 clients, 60 rounds, deployed model = E3b personalised model, immediate labels. Means ± std over seeds; with 3 seeds nothing here is a confirmatory test.

### S1 real_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3_global | 0.609 ± 0.021 | 0.930 | 0.321 | 40.0 | nan | 0.896 | 0 | 0.0 |
| frozen_E3b_personal | 0.622 ± 0.016 | 0.945 | 0.323 | 40.0 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.610 ± 0.017 | 0.936 | 0.337 | 40.0 | nan | 0.882 | 8522 | 40.0 |
| oracle_full_fed | 0.590 ± 0.011 | 0.945 | 0.364 | 40.0 | nan | 0.945 | 1917 | 3.0 |
| oracle_head_long | 0.675 ± 0.008 | 0.945 | 0.333 | 36.9 | nan | 0.945 | 0 | 3.0 |
| oracle_adapter_long | 0.652 ± 0.013 | 0.945 | 0.394 | 37.6 | nan | 0.945 | 0 | 3.0 |
| oracle_full_long | 0.628 ± 0.017 | 0.945 | 0.407 | 38.0 | nan | 0.945 | 0 | 3.0 |
| ph_full_fed_E5 | 0.624 ± 0.008 | 0.945 | 0.364 | 40.0 | nan | 0.932 | 2983 | 4.7 |
| ph_head_long | 0.666 ± 0.007 | 0.945 | 0.329 | 37.4 | nan | 0.944 | 0 | 3.7 |
| typed_learned | 0.622 ± 0.065 | 0.945 | 0.396 | 38.0 | nan | 0.939 | 0 | 3.7 |
| ph_mode_oracle | 0.664 ± 0.007 | 0.945 | 0.337 | 37.3 | nan | 0.923 | 0 | 4.0 |
| typed_learned_conf80 | 0.623 ± 0.065 | 0.945 | 0.397 | 38.0 | nan | 0.944 | 0 | 3.7 |

### S1 virtual_gain_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3_global | 0.683 ± 0.041 | 0.930 | 0.247 | 40.0 | nan | 0.896 | 0 | 0.0 |
| frozen_E3b_personal | 0.762 ± 0.067 | 0.945 | 0.183 | 35.6 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.720 ± 0.051 | 0.936 | 0.248 | 35.6 | nan | 0.877 | 8522 | 40.0 |
| oracle_full_fed | 0.775 ± 0.060 | 0.945 | 0.253 | 31.7 | nan | 0.945 | 1917 | 3.0 |
| oracle_adapter_fed | 0.795 ± 0.067 | 0.945 | 0.187 | 27.4 | nan | 0.945 | 71 | 3.0 |
| oracle_align | 0.884 ± 0.035 | 0.945 | 0.197 | 16.1 | nan | 0.945 | 0 | 3.0 |
| ph_full_fed_E5 | 0.744 ± 0.089 | 0.945 | 0.218 | 35.6 | nan | 0.931 | 1704 | 2.7 |
| ph_align | 0.780 ± 0.049 | 0.945 | 0.232 | 32.8 | nan | 0.928 | 0 | 2.3 |
| ph_adapter_fed | 0.743 ± 0.091 | 0.945 | 0.219 | 35.6 | nan | 0.944 | 47 | 2.0 |
| typed_learned | 0.768 ± 0.060 | 0.945 | 0.183 | 35.6 | nan | 0.939 | 0 | 2.0 |
| ph_mode_oracle | 0.768 ± 0.060 | 0.945 | 0.183 | 35.6 | nan | 0.923 | 0 | 2.3 |
| ph_head_long | 0.762 ± 0.067 | 0.945 | 0.199 | 35.6 | nan | 0.944 | 0 | 2.0 |
| typed_learned_conf80 | 0.768 ± 0.060 | 0.945 | 0.183 | 35.6 | nan | 0.942 | 0 | 2.0 |

### S1 virtual_offset_full (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3_global | 0.807 ± 0.024 | 0.909 | 0.102 | 36.7 | nan | nan | 0 | 0.0 |
| frozen_E3b_personal | 0.873 ± 0.015 | 0.945 | 0.072 | 25.0 | nan | nan | 0 | 0.0 |
| periodic_E3c_full_fed | 0.843 ± 0.022 | 0.924 | 0.132 | 26.2 | nan | nan | 8522 | 40.0 |
| oracle_full_fed | 0.820 ± 0.034 | 0.945 | 0.165 | 30.6 | nan | nan | 5113 | 8.0 |
| oracle_adapter_fed | 0.871 ± 0.025 | 0.945 | 0.095 | 24.4 | nan | nan | 189 | 8.0 |
| oracle_align | 0.880 ± 0.019 | 0.945 | 0.144 | 19.2 | nan | nan | 0 | 8.0 |
| ph_full_fed_E5 | 0.850 ± 0.026 | 0.944 | 0.117 | 27.7 | nan | nan | 2343 | 3.7 |
| ph_align | 0.867 ± 0.015 | 0.944 | 0.118 | 22.5 | nan | nan | 0 | 3.3 |
| ph_adapter_fed | 0.863 ± 0.019 | 0.945 | 0.095 | 26.4 | nan | nan | 79 | 3.3 |
| typed_learned | 0.863 ± 0.008 | 0.945 | 0.113 | 23.4 | nan | nan | 0 | 3.0 |
| ph_mode_oracle | 0.863 ± 0.008 | 0.945 | 0.113 | 23.4 | nan | nan | 0 | 3.0 |
| ph_head_long | 0.873 ± 0.014 | 0.945 | 0.087 | 26.0 | nan | nan | 0 | 3.0 |
| typed_learned_conf80 | 0.863 ± 0.009 | 0.945 | 0.111 | 23.4 | nan | nan | 0 | 3.0 |

### S1 virtual_offset_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3_global | 0.841 ± 0.011 | 0.930 | 0.089 | 40.0 | nan | 0.896 | 0 | 0.0 |
| frozen_E3b_personal | 0.892 ± 0.014 | 0.945 | 0.053 | 17.8 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.844 ± 0.030 | 0.936 | 0.125 | 37.8 | nan | 0.878 | 8522 | 40.0 |
| oracle_full_fed | 0.840 ± 0.023 | 0.945 | 0.132 | 35.9 | nan | 0.945 | 1917 | 3.0 |
| oracle_adapter_fed | 0.885 ± 0.015 | 0.945 | 0.078 | 14.9 | nan | 0.945 | 71 | 3.0 |
| oracle_align | 0.909 ± 0.028 | 0.945 | 0.080 | 19.1 | nan | 0.945 | 0 | 3.0 |
| ph_full_fed_E5 | 0.869 ± 0.045 | 0.945 | 0.111 | 21.4 | nan | 0.931 | 1917 | 3.0 |
| ph_align | 0.879 ± 0.037 | 0.945 | 0.137 | 17.8 | nan | 0.928 | 0 | 2.7 |
| ph_adapter_fed | 0.879 ± 0.029 | 0.945 | 0.084 | 22.2 | nan | 0.945 | 55 | 2.3 |
| typed_learned | 0.887 ± 0.023 | 0.945 | 0.076 | 17.8 | nan | 0.939 | 0 | 2.3 |
| ph_mode_oracle | 0.887 ± 0.023 | 0.945 | 0.076 | 17.8 | nan | 0.923 | 0 | 2.7 |
| ph_head_long | 0.891 ± 0.018 | 0.945 | 0.073 | 17.8 | nan | 0.944 | 0 | 2.3 |
| typed_learned_conf80 | 0.886 ± 0.023 | 0.945 | 0.080 | 17.8 | nan | 0.943 | 0 | 2.3 |

### S1 virtual_rotation_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3_global | 0.630 ± 0.049 | 0.930 | 0.300 | 40.0 | nan | 0.896 | 0 | 0.0 |
| frozen_E3b_personal | 0.682 ± 0.042 | 0.945 | 0.263 | 40.0 | nan | 0.945 | 0 | 0.0 |
| periodic_E3c_full_fed | 0.688 ± 0.053 | 0.936 | 0.337 | 37.8 | nan | 0.878 | 8522 | 40.0 |
| oracle_full_fed | 0.688 ± 0.040 | 0.945 | 0.327 | 40.0 | nan | 0.945 | 1917 | 3.0 |
| oracle_adapter_fed | 0.692 ± 0.074 | 0.945 | 0.283 | 35.8 | nan | 0.945 | 71 | 3.0 |
| oracle_align | 0.845 ± 0.034 | 0.945 | 0.268 | 24.0 | nan | 0.945 | 0 | 3.0 |
| ph_full_fed_E5 | 0.672 ± 0.046 | 0.945 | 0.348 | 39.9 | nan | 0.930 | 2983 | 5.0 |
| ph_align | 0.752 ± 0.047 | 0.945 | 0.302 | 32.1 | nan | 0.928 | 0 | 4.7 |
| ph_adapter_fed | 0.691 ± 0.063 | 0.945 | 0.321 | 36.9 | nan | 0.941 | 115 | 5.3 |
| typed_learned | 0.735 ± 0.044 | 0.945 | 0.263 | 32.9 | nan | 0.939 | 0 | 3.7 |
| ph_mode_oracle | 0.743 ± 0.052 | 0.945 | 0.263 | 31.6 | nan | 0.923 | 0 | 4.0 |
| ph_head_long | 0.708 ± 0.048 | 0.945 | 0.279 | 36.9 | nan | 0.945 | 0 | 4.0 |
| typed_learned_conf80 | 0.735 ± 0.044 | 0.945 | 0.263 | 32.9 | nan | 0.943 | 0 | 3.7 |

### S2 real_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.656 ± 0.018 | 0.945 | 0.330 | 34.4 | nan | 0.945 | 0 | 0.0 |
| typed_oracle | 0.682 ± 0.008 | 0.945 | 0.332 | 34.4 | nan | 0.945 | 0 | 3.0 |
| fixed_align_oneshot | 0.598 ± 0.038 | 0.945 | 0.401 | 34.4 | nan | 0.945 | 0 | 3.0 |
| fixed_head_long | 0.682 ± 0.008 | 0.945 | 0.332 | 34.4 | nan | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.625 ± 0.017 | 0.945 | 0.364 | 34.4 | nan | 0.945 | 1917 | 3.0 |
| mode_typed_oracle | 0.682 ± 0.008 | 0.945 | 0.332 | 34.4 | nan | 0.945 | 0 | 3.0 |
| typed_learned | 0.654 ± 0.024 | 0.945 | 0.393 | 34.4 | nan | 0.938 | 0 | 4.0 |
| ph_mode_oracle | 0.672 ± 0.006 | 0.945 | 0.345 | 34.4 | nan | 0.920 | 0 | 4.3 |
| ph_align | 0.541 ± 0.036 | 0.945 | 0.606 | 34.4 | nan | 0.926 | 0 | 5.3 |
| ph_head_long | 0.677 ± 0.005 | 0.945 | 0.334 | 34.4 | nan | 0.943 | 0 | 4.0 |
| ph_full_fed_E5 | 0.664 ± 0.004 | 0.945 | 0.357 | 34.4 | nan | 0.932 | 3125 | 5.0 |
| typed_learned_conf80 | 0.655 ± 0.024 | 0.945 | 0.390 | 34.4 | nan | 0.943 | 0 | 4.0 |

### S2 virtual_offset_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.903 ± 0.009 | 0.945 | 0.053 | 19.9 | nan | 0.945 | 0 | 0.0 |
| typed_oracle | 0.906 ± 0.029 | 0.945 | 0.180 | 15.9 | nan | 0.945 | 0 | 3.0 |
| fixed_align_oneshot | 0.861 ± 0.068 | 0.945 | 0.123 | 17.8 | nan | 0.945 | 0 | 3.0 |
| fixed_align_track | 0.906 ± 0.029 | 0.945 | 0.180 | 15.9 | nan | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.848 ± 0.026 | 0.945 | 0.132 | 24.0 | nan | 0.945 | 1917 | 3.0 |
| fixed_adapter_fed | 0.869 ± 0.024 | 0.945 | 0.093 | 19.8 | nan | 0.945 | 71 | 3.0 |
| mode_typed_oracle | 0.861 ± 0.068 | 0.945 | 0.123 | 17.8 | nan | 0.945 | 0 | 3.0 |
| typed_learned | 0.894 ± 0.021 | 0.945 | 0.089 | 19.9 | nan | 0.938 | 0 | 2.0 |
| ph_mode_oracle | 0.894 ± 0.021 | 0.945 | 0.089 | 19.9 | nan | 0.920 | 0 | 2.3 |
| ph_align | 0.889 ± 0.029 | 0.945 | 0.138 | 19.9 | nan | 0.926 | 0 | 2.3 |
| ph_head_long | 0.900 ± 0.013 | 0.945 | 0.070 | 19.9 | nan | 0.943 | 0 | 2.0 |
| ph_full_fed_E5 | 0.902 ± 0.011 | 0.945 | 0.068 | 19.9 | nan | 0.931 | 1704 | 2.7 |
| typed_learned_conf80 | 0.895 ± 0.021 | 0.945 | 0.076 | 19.9 | nan | 0.942 | 0 | 2.0 |

### S2 virtual_rotation_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.734 ± 0.034 | 0.945 | 0.265 | 31.0 | nan | 0.945 | 0 | 0.0 |
| typed_oracle | 0.873 ± 0.033 | 0.945 | 0.199 | 21.1 | nan | 0.945 | 0 | 3.0 |
| fixed_align_oneshot | 0.833 ± 0.041 | 0.945 | 0.161 | 31.4 | nan | 0.945 | 0 | 3.0 |
| fixed_align_track | 0.873 ± 0.033 | 0.945 | 0.199 | 21.1 | nan | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.699 ± 0.069 | 0.945 | 0.302 | 30.9 | nan | 0.945 | 1917 | 3.0 |
| fixed_adapter_fed | 0.719 ± 0.035 | 0.945 | 0.276 | 34.4 | nan | 0.945 | 71 | 3.0 |
| mode_typed_oracle | 0.833 ± 0.041 | 0.945 | 0.161 | 31.4 | nan | 0.945 | 0 | 3.0 |
| typed_learned | 0.783 ± 0.046 | 0.945 | 0.265 | 27.8 | nan | 0.938 | 0 | 3.3 |
| ph_mode_oracle | 0.792 ± 0.049 | 0.945 | 0.265 | 26.4 | nan | 0.920 | 0 | 3.7 |
| ph_align | 0.786 ± 0.038 | 0.945 | 0.278 | 28.6 | nan | 0.926 | 0 | 4.0 |
| ph_head_long | 0.747 ± 0.030 | 0.945 | 0.279 | 31.0 | nan | 0.943 | 0 | 3.7 |
| ph_full_fed_E5 | 0.753 ± 0.045 | 0.945 | 0.303 | 31.0 | nan | 0.932 | 2912 | 4.7 |
| typed_learned_conf80 | 0.783 ± 0.046 | 0.945 | 0.265 | 27.8 | nan | 0.942 | 0 | 3.3 |

### S3 real_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.784 ± 0.016 | 0.945 | 0.323 | 10.0 | 0.945 | 0.945 | 0 | 0.0 |
| typed_oracle | 0.785 ± 0.009 | 0.945 | 0.334 | 18.0 | 0.926 | 0.945 | 0 | 12.0 |
| fixed_align_oneshot | 0.552 ± 0.074 | 0.945 | 0.671 | 38.9 | 0.616 | 0.945 | 0 | 6.0 |
| fixed_head_long | 0.758 ± 0.014 | 0.945 | 0.329 | 27.8 | 0.863 | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.735 ± 0.015 | 0.945 | 0.403 | 34.4 | 0.880 | 0.945 | 3835 | 6.0 |
| mode_typed_oracle | 0.758 ± 0.014 | 0.945 | 0.329 | 27.8 | 0.863 | 0.945 | 0 | 3.0 |
| typed_learned | 0.714 ± 0.061 | 0.945 | 0.385 | 16.7 | 0.815 | 0.939 | 0 | 3.7 |
| ph_mode_oracle | 0.757 ± 0.015 | 0.945 | 0.329 | 16.7 | 0.879 | 0.923 | 0 | 4.0 |
| ph_align | 0.570 ± 0.037 | 0.945 | 0.636 | 33.2 | 0.635 | 0.928 | 0 | 4.3 |
| ph_head_long | 0.758 ± 0.013 | 0.945 | 0.329 | 20.8 | 0.875 | 0.944 | 0 | 3.7 |
| ph_full_fed_E5 | 0.758 ± 0.011 | 0.945 | 0.348 | 30.0 | 0.893 | 0.931 | 2983 | 4.7 |
| typed_learned_conf80 | 0.715 ± 0.061 | 0.945 | 0.385 | 16.7 | 0.817 | 0.944 | 0 | 3.7 |

### S3 virtual_offset_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.919 ± 0.001 | 0.945 | 0.053 | 4.4 | 0.945 | 0.945 | 0 | 0.0 |
| typed_oracle | 0.917 ± 0.013 | 0.945 | 0.112 | 9.9 | 0.929 | 0.945 | 0 | 12.0 |
| fixed_align_oneshot | 0.874 ± 0.015 | 0.945 | 0.153 | 25.3 | 0.849 | 0.945 | 0 | 6.0 |
| fixed_align_track | 0.871 ± 0.012 | 0.945 | 0.233 | 23.2 | 0.847 | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.878 ± 0.016 | 0.945 | 0.178 | 25.2 | 0.905 | 0.945 | 3835 | 6.0 |
| fixed_adapter_fed | 0.914 ± 0.009 | 0.945 | 0.095 | 23.8 | 0.945 | 0.945 | 142 | 6.0 |
| mode_typed_oracle | 0.874 ± 0.015 | 0.945 | 0.153 | 25.3 | 0.849 | 0.945 | 0 | 6.0 |
| typed_learned | 0.916 ± 0.003 | 0.945 | 0.066 | 11.1 | 0.938 | 0.939 | 0 | 2.0 |
| ph_mode_oracle | 0.916 ± 0.003 | 0.945 | 0.066 | 11.1 | 0.938 | 0.923 | 0 | 2.3 |
| ph_align | 0.907 ± 0.016 | 0.945 | 0.073 | 7.8 | 0.927 | 0.928 | 0 | 2.3 |
| ph_head_long | 0.917 ± 0.001 | 0.945 | 0.057 | 6.7 | 0.942 | 0.945 | 0 | 2.0 |
| ph_full_fed_E5 | 0.911 ± 0.009 | 0.945 | 0.078 | 7.8 | 0.939 | 0.931 | 1704 | 2.7 |
| typed_learned_conf80 | 0.916 ± 0.003 | 0.945 | 0.066 | 11.1 | 0.938 | 0.942 | 0 | 2.0 |

### S3 virtual_rotation_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.814 ± 0.017 | 0.945 | 0.263 | 10.0 | 0.945 | 0.945 | 0 | 0.0 |
| typed_oracle | 0.856 ± 0.019 | 0.945 | 0.319 | 9.6 | 0.898 | 0.945 | 0 | 12.0 |
| fixed_align_oneshot | 0.761 ± 0.015 | 0.945 | 0.365 | 26.2 | 0.731 | 0.945 | 0 | 6.0 |
| fixed_align_track | 0.724 ± 0.045 | 0.945 | 0.458 | 34.7 | 0.666 | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.798 ± 0.012 | 0.945 | 0.342 | 25.8 | 0.893 | 0.945 | 3835 | 6.0 |
| fixed_adapter_fed | 0.809 ± 0.026 | 0.945 | 0.295 | 21.3 | 0.924 | 0.945 | 142 | 6.0 |
| mode_typed_oracle | 0.761 ± 0.015 | 0.945 | 0.365 | 26.2 | 0.731 | 0.945 | 0 | 6.0 |
| typed_learned | 0.793 ± 0.030 | 0.945 | 0.308 | 16.7 | 0.895 | 0.939 | 0 | 2.3 |
| ph_mode_oracle | 0.793 ± 0.030 | 0.945 | 0.308 | 16.7 | 0.895 | 0.923 | 0 | 2.7 |
| ph_align | 0.786 ± 0.038 | 0.945 | 0.301 | 16.7 | 0.878 | 0.928 | 0 | 3.0 |
| ph_head_long | 0.813 ± 0.014 | 0.945 | 0.279 | 17.6 | 0.938 | 0.944 | 0 | 2.7 |
| ph_full_fed_E5 | 0.812 ± 0.020 | 0.945 | 0.276 | 13.3 | 0.936 | 0.934 | 2130 | 3.3 |
| typed_learned_conf80 | 0.793 ± 0.030 | 0.945 | 0.308 | 16.7 | 0.895 | 0.942 | 0 | 2.3 |

### S4 ankle_gain_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.724 ± 0.050 | 0.945 | 0.221 | 40.0 | nan | 0.945 | 0 | 0.0 |
| typed_oracle | 0.913 ± 0.015 | 0.945 | 0.221 | 11.3 | nan | 0.945 | 0 | 3.0 |
| fixed_align_oneshot | 0.893 ± 0.012 | 0.945 | 0.221 | 15.4 | nan | 0.945 | 0 | 3.0 |
| fixed_align_track | 0.902 ± 0.018 | 0.945 | 0.270 | 7.3 | nan | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.705 ± 0.089 | 0.945 | 0.312 | 35.9 | nan | 0.945 | 1917 | 3.0 |
| fixed_adapter_fed | 0.765 ± 0.036 | 0.945 | 0.232 | 40.0 | nan | 0.945 | 71 | 3.0 |
| mode_typed_oracle | 0.893 ± 0.012 | 0.945 | 0.221 | 15.4 | nan | 0.945 | 0 | 3.0 |
| typed_learned | 0.759 ± 0.013 | 0.945 | 0.221 | 35.8 | nan | 0.939 | 0 | 2.7 |
| ph_mode_oracle | 0.759 ± 0.013 | 0.945 | 0.221 | 35.8 | nan | 0.923 | 0 | 3.0 |
| ph_align | 0.780 ± 0.016 | 0.945 | 0.221 | 31.3 | nan | 0.928 | 0 | 3.3 |
| ph_head_long | 0.739 ± 0.034 | 0.945 | 0.223 | 40.0 | nan | 0.944 | 0 | 2.7 |
| ph_full_fed_E5 | 0.725 ± 0.039 | 0.945 | 0.244 | 40.0 | nan | 0.926 | 2130 | 3.3 |
| typed_learned_conf80 | 0.759 ± 0.013 | 0.945 | 0.223 | 35.8 | nan | 0.942 | 0 | 2.7 |

### S4 chest_rotation_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.733 ± 0.047 | 0.945 | 0.212 | 40.0 | nan | 0.945 | 0 | 0.0 |
| typed_oracle | 0.893 ± 0.035 | 0.945 | 0.220 | 15.3 | nan | 0.945 | 0 | 3.0 |
| fixed_align_oneshot | 0.878 ± 0.040 | 0.945 | 0.218 | 15.4 | nan | 0.945 | 0 | 3.0 |
| fixed_align_track | 0.897 ± 0.025 | 0.945 | 0.255 | 9.6 | nan | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.725 ± 0.046 | 0.945 | 0.274 | 40.0 | nan | 0.945 | 1917 | 3.0 |
| fixed_adapter_fed | 0.788 ± 0.042 | 0.945 | 0.230 | 35.9 | nan | 0.945 | 71 | 3.0 |
| mode_typed_oracle | 0.878 ± 0.040 | 0.945 | 0.218 | 15.4 | nan | 0.945 | 0 | 3.0 |
| typed_learned | 0.795 ± 0.041 | 0.945 | 0.212 | 29.7 | nan | 0.939 | 0 | 2.7 |
| ph_mode_oracle | 0.795 ± 0.041 | 0.945 | 0.212 | 29.7 | nan | 0.923 | 0 | 3.0 |
| ph_align | 0.805 ± 0.043 | 0.945 | 0.214 | 31.4 | nan | 0.928 | 0 | 3.7 |
| ph_head_long | 0.747 ± 0.037 | 0.945 | 0.216 | 40.0 | nan | 0.944 | 0 | 3.3 |
| ph_full_fed_E5 | 0.733 ± 0.044 | 0.945 | 0.249 | 40.0 | nan | 0.934 | 2912 | 4.7 |
| typed_learned_conf80 | 0.783 ± 0.029 | 0.945 | 0.219 | 32.6 | nan | 0.942 | 0 | 2.7 |

### S4 hand_offset_partial (severity med, seeds [1001, 1002, 1003])

| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |
|---|---|---|---|---|---|---|---|---|
| frozen_E3b_personal | 0.934 ± 0.015 | 0.945 | 0.011 | 0.0 | nan | 0.945 | 0 | 0.0 |
| typed_oracle | 0.941 ± 0.013 | 0.945 | 0.030 | 6.3 | nan | 0.945 | 0 | 3.0 |
| fixed_align_oneshot | 0.926 ± 0.023 | 0.945 | 0.043 | 14.9 | nan | 0.945 | 0 | 3.0 |
| fixed_align_track | 0.921 ± 0.017 | 0.945 | 0.154 | 16.8 | nan | 0.945 | 0 | 3.0 |
| fixed_full_fed | 0.867 ± 0.049 | 0.945 | 0.089 | 23.8 | nan | 0.945 | 1917 | 3.0 |
| fixed_adapter_fed | 0.901 ± 0.027 | 0.945 | 0.057 | 23.3 | nan | 0.945 | 71 | 3.0 |
| mode_typed_oracle | 0.926 ± 0.023 | 0.945 | 0.043 | 14.9 | nan | 0.945 | 0 | 3.0 |
| typed_learned | 0.934 ± 0.015 | 0.945 | 0.019 | 8.8 | nan | 0.939 | 0 | 2.0 |
| ph_mode_oracle | 0.923 ± 0.013 | 0.945 | 0.064 | 8.9 | nan | 0.923 | 0 | 2.3 |
| ph_align | 0.920 ± 0.016 | 0.945 | 0.099 | 8.9 | nan | 0.928 | 0 | 2.3 |
| ph_head_long | 0.933 ± 0.014 | 0.945 | 0.020 | 8.3 | nan | 0.944 | 0 | 2.0 |
| ph_full_fed_E5 | 0.925 ± 0.012 | 0.945 | 0.034 | 8.9 | nan | 0.931 | 1704 | 2.7 |
| typed_learned_conf80 | 0.934 ± 0.015 | 0.945 | 0.019 | 8.4 | nan | 0.942 | 0 | 2.0 |

## Typed (oracle) vs fixed responses, post-change macro-F1 per variant

| variant | typed_oracle | best fixed (name) | typed - best fixed | frozen E3b |
|---|---|---|---|---|
| S2 real_partial | 0.682 | 0.682 (fixed_head_long) | +0.000 | 0.656 |
| S2 virtual_offset_partial | 0.906 | 0.906 (fixed_align_track) | +0.000 | 0.903 |
| S2 virtual_rotation_partial | 0.873 | 0.873 (fixed_align_track) | +0.000 | 0.734 |
| S3 real_partial | 0.785 | 0.758 (fixed_head_long) | +0.027 | 0.784 |
| S3 virtual_offset_partial | 0.917 | 0.914 (fixed_adapter_fed) | +0.003 | 0.919 |
| S3 virtual_rotation_partial | 0.856 | 0.809 (fixed_adapter_fed) | +0.047 | 0.814 |
| S4 ankle_gain_partial | 0.913 | 0.902 (fixed_align_track) | +0.012 | 0.724 |
| S4 chest_rotation_partial | 0.893 | 0.897 (fixed_align_track) | -0.004 | 0.733 |
| S4 hand_offset_partial | 0.941 | 0.926 (fixed_align_oneshot) | +0.014 | 0.934 |

Mean (typed - best fixed) over 9 variants: +0.011. The best fixed policy is picked per variant, i.e. the comparison favours the fixed side.

## Deployable (detector-triggered) systems, post-change macro-F1 (3 tuning seeds, immediate labels)

| variant | frozen_E3b_personal | ph_full_fed_E5 | ph_align | ph_head_long | typed_learned | typed_learned_conf80 | ph_mode_oracle |
|---|---|---|---|---|---|---|---|
| S1 real_partial | 0.622 | 0.624 | - | 0.666 | 0.622 | 0.623 | 0.664 |
| S1 virtual_gain_partial | 0.762 | 0.744 | 0.780 | 0.762 | 0.768 | 0.768 | 0.768 |
| S1 virtual_offset_full | 0.873 | 0.850 | 0.867 | 0.873 | 0.863 | 0.863 | 0.863 |
| S1 virtual_offset_partial | 0.892 | 0.869 | 0.879 | 0.891 | 0.887 | 0.886 | 0.887 |
| S1 virtual_rotation_partial | 0.682 | 0.672 | 0.752 | 0.708 | 0.735 | 0.735 | 0.743 |
| S2 real_partial | 0.656 | 0.664 | 0.541 | 0.677 | 0.654 | 0.655 | 0.672 |
| S2 virtual_offset_partial | 0.903 | 0.902 | 0.889 | 0.900 | 0.894 | 0.895 | 0.894 |
| S2 virtual_rotation_partial | 0.734 | 0.753 | 0.786 | 0.747 | 0.783 | 0.783 | 0.792 |
| S3 real_partial | 0.784 | 0.758 | 0.570 | 0.758 | 0.714 | 0.715 | 0.757 |
| S3 virtual_offset_partial | 0.919 | 0.911 | 0.907 | 0.917 | 0.916 | 0.916 | 0.916 |
| S3 virtual_rotation_partial | 0.814 | 0.812 | 0.786 | 0.813 | 0.793 | 0.793 | 0.793 |
| S4 ankle_gain_partial | 0.724 | 0.725 | 0.780 | 0.739 | 0.759 | 0.759 | 0.759 |
| S4 chest_rotation_partial | 0.733 | 0.733 | 0.805 | 0.747 | 0.795 | 0.783 | 0.795 |
| S4 hand_offset_partial | 0.934 | 0.925 | 0.920 | 0.933 | 0.934 | 0.934 | 0.923 |
| **mean (variants common to all)** | **0.801** | **0.794** | **0.789** | **0.805** | **0.807** | **0.806** | **0.812** |

13 variants common to all columns. typed_* use a mode classifier (virtual / real) after a 5-round typing delay; ph_mode_oracle uses the true mode; conf80 applies alignment only if the classifier is >= 80% sure the drift is virtual.

## Value of the 4-way type beyond the 2-way mode (both with oracle labels), post-change F1

| variant | 4-way typed_oracle | 2-way mode_typed_oracle | difference |
|---|---|---|---|
| S2 real_partial | 0.682 | 0.682 | +0.000 |
| S2 virtual_offset_partial | 0.906 | 0.861 | +0.045 |
| S2 virtual_rotation_partial | 0.873 | 0.833 | +0.041 |
| S3 real_partial | 0.785 | 0.758 | +0.027 |
| S3 virtual_offset_partial | 0.917 | 0.874 | +0.043 |
| S3 virtual_rotation_partial | 0.856 | 0.761 | +0.095 |
| S4 ankle_gain_partial | 0.913 | 0.893 | +0.020 |
| S4 chest_rotation_partial | 0.893 | 0.878 | +0.015 |
| S4 hand_offset_partial | 0.941 | 0.926 | +0.014 |

Mean 4-way minus 2-way over 9 variants: +0.033.
