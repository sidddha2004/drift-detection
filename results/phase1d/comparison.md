# Phase 1d: v1 (frozen, untrimmed) vs v2 (5 s edge trim)

## blocked split (5 seeds)

| method | v1 macro-F1 | v2 macro-F1 | shift | v1 worst-client | v2 worst-client |
|---|---|---|---|---|---|
| E1 local-only | 0.8826 ± 0.0027 | 0.9386 ± 0.0056 | +0.0560 | 0.830 | 0.894 |
| E2 centralised | 0.8711 ± 0.0075 | 0.9298 ± 0.0102 | +0.0587 | 0.769 | 0.825 |
| E3 FedAvg | 0.8507 ± 0.0146 | 0.8926 ± 0.0074 | +0.0419 | 0.676 | 0.730 |
| E3b FedAvg+FT | 0.8879 ± 0.0089 | 0.9354 ± 0.0147 | +0.0475 | 0.814 | 0.827 |

| pair (A − B) | v1 diff [95% CI] | v2 diff [95% CI] | ranking change? |
|---|---|---|---|
| E1 local-only − E2 centralised | +0.0115 [+0.0032, +0.0197] | +0.0088 [+0.0030, +0.0192] | no |
| E1 local-only − E3 FedAvg | +0.0319 [+0.0160, +0.0444] | +0.0460 [+0.0359, +0.0561] | no |
| E1 local-only − E3b FedAvg+FT | -0.0053 [-0.0120, +0.0021] | +0.0032 [-0.0096, +0.0167] | no |
| E2 centralised − E3 FedAvg | +0.0204 [+0.0101, +0.0294] | +0.0372 [+0.0295, +0.0481] | no |
| E2 centralised − E3b FedAvg+FT | -0.0168 [-0.0294, -0.0093] | -0.0056 [-0.0290, +0.0140] | no |
| E3 FedAvg − E3b FedAvg+FT | -0.0372 [-0.0491, -0.0236] | -0.0427 [-0.0609, -0.0282] | no |

Order by mean, v1: E3b FedAvg+FT > E1 local-only > E2 centralised > E3 FedAvg
Order by mean, v2: E1 local-only > E3b FedAvg+FT > E2 centralised > E3 FedAvg

**Ranking changes under the pre-set rule: 0** []

## temporal split (5 seeds)

| method | v1 macro-F1 | v2 macro-F1 | shift | v1 worst-client | v2 worst-client |
|---|---|---|---|---|---|
| E1 local-only | 0.8147 ± 0.0051 | 0.8946 ± 0.0041 | +0.0799 | 0.736 | 0.835 |
| E2 centralised | 0.8102 ± 0.0041 | 0.8835 ± 0.0106 | +0.0733 | 0.726 | 0.793 |
| E3 FedAvg | 0.7735 ± 0.0130 | 0.8622 ± 0.0141 | +0.0886 | 0.608 | 0.748 |
| E3b FedAvg+FT | 0.8150 ± 0.0074 | 0.9031 ± 0.0088 | +0.0881 | 0.743 | 0.840 |

| pair (A − B) | v1 diff [95% CI] | v2 diff [95% CI] | ranking change? |
|---|---|---|---|
| E1 local-only − E2 centralised | +0.0045 [-0.0034, +0.0127] | +0.0111 [-0.0016, +0.0225] | no |
| E1 local-only − E3 FedAvg | +0.0412 [+0.0294, +0.0495] | +0.0324 [+0.0235, +0.0428] | no |
| E1 local-only − E3b FedAvg+FT | -0.0003 [-0.0045, +0.0047] | -0.0085 [-0.0151, -0.0031] | no |
| E2 centralised − E3 FedAvg | +0.0367 [+0.0220, +0.0494] | +0.0213 [+0.0011, +0.0446] | no |
| E2 centralised − E3b FedAvg+FT | -0.0048 [-0.0141, +0.0042] | -0.0196 [-0.0326, -0.0066] | no |
| E3 FedAvg − E3b FedAvg+FT | -0.0415 [-0.0474, -0.0322] | -0.0409 [-0.0534, -0.0298] | no |

Order by mean, v1: E3b FedAvg+FT > E1 local-only > E2 centralised > E3 FedAvg
Order by mean, v2: E3b FedAvg+FT > E1 local-only > E2 centralised > E3 FedAvg

**Ranking changes under the pre-set rule: 0** []
