# Phase 2 drift scenario catalogue

Reference seed 42, severity `med`, protocol v2 clients [101, 102, 103, 104, 105, 106, 107, 108].

| scenario | variant | type4 | clients (start round) | change points (first client) |
|---|---|---|---|---|
| S0 | none | - | - | - |
| S1 | virtual_offset_partial | sudden | 101(20), 103(20), 107(20) | onset@20 |
| S1 | virtual_offset_full | sudden | 101(20), 102(20), 103(20), 104(20), 105(20), 106(20), 107(20), 108(20) | onset@20 |
| S1 | virtual_gain_partial | sudden | 101(20), 103(20), 107(20) | onset@20 |
| S1 | virtual_rotation_partial | sudden | 101(20), 103(20), 107(20) | onset@20 |
| S1 | real_partial | sudden | 101(20), 103(20), 107(20) | onset@20 |
| S1 | real_full | sudden | 101(20), 102(20), 103(20), 104(20), 105(20), 106(20), 107(20), 108(20) | onset@20 |
| S2 | virtual_offset_partial | gradual | 101(22), 103(26), 107(29) | onset@22, full@31 |
| S2 | virtual_offset_full | gradual | 101(22), 102(23), 103(26), 104(21), 105(21), 106(30), 107(29), 108(27) | onset@22, full@31 |
| S2 | virtual_rotation_partial | gradual | 101(22), 103(26), 107(29) | onset@22, full@31 |
| S2 | real_partial | gradual | 101(22), 103(26), 107(29) | onset@22, full@31 |
| S3 | virtual_offset_partial | recurring | 101(20), 103(20), 107(20) | onset@20, offset@30, onset@40, offset@50 |
| S3 | virtual_rotation_partial | recurring | 101(20), 103(20), 107(20) | onset@20, offset@30, onset@40, offset@50 |
| S3 | real_partial | recurring | 101(20), 103(20), 107(20) | onset@20, offset@30, onset@40, offset@50 |
| S4 | hand_offset_partial | feature_level | 101(20), 103(20), 107(20) | onset@20 |
| S4 | ankle_gain_partial | feature_level | 101(20), 103(20), 107(20) | onset@20 |
| S4 | chest_rotation_partial | feature_level | 101(20), 103(20), 107(20) | onset@20 |