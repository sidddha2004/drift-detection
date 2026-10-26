# Phase 1 results

## S0 stationary control: blocked split (train/val/test interleaved in time)

Run `5b817cf` on NVIDIA GeForce RTX 5050 Laptop GPU | seeds [7, 42, 123, 999, 2024] | 8 clients (subjects [101, 102, 103, 104, 105, 106, 107, 108]) | split `blocked` | test split only.

## Headline (mean over seeds ± std; CI = 95% bootstrap over seeds)

| method | test macro-F1 | 95% CI | val macro-F1 (selected ckpt) | worst-client macro-F1 | accuracy | client std |
|---|---|---|---|---|---|---|
| E1 Local-only | 0.8828 ± 0.0038 | [0.8798, 0.8863] | 0.9158 ± 0.0024 | 0.8177 ± 0.0160 | 0.8937 ± 0.0039 | 0.0310 ± 0.0051 |
| E2 Centralised (oracle) | 0.8680 ± 0.0093 | [0.8598, 0.8757] | 0.8995 ± 0.0061 | 0.7514 ± 0.0579 | 0.9086 ± 0.0045 | 0.0578 ± 0.0143 |
| E3 FedAvg | 0.8355 ± 0.0099 | [0.8256, 0.8417] | 0.8442 ± 0.0120 | 0.6864 ± 0.0649 | 0.8858 ± 0.0034 | 0.0704 ± 0.0185 |

## Paired tests across seeds (macro-F1, Wilcoxon, Holm-adjusted)

| comparison | Δ macro-F1 | p (raw) | p (Holm) |
|---|---|---|---|
| E3_fedavg vs E1_local_only | -0.0473 | 0.0625 | 0.1875 |
| E2_centralized vs E3_fedavg | +0.0325 | 0.0625 | 0.1875 |
| E2_centralized vs E1_local_only | -0.0148 | 0.0625 | 0.1875 |

With n=5 seeds the smallest attainable two-sided Wilcoxon p is 0.0625; Holm over 3 tests multiplies it. Treat these tests as indicative only.

## Per-client macro-F1 (mean over seeds)

| method | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 |
|---|---|---|---|---|---|---|---|---|
| E1 Local-only | 0.902 | 0.915 | 0.860 | 0.898 | 0.827 | 0.893 | 0.898 | 0.868 |
| E2 Centralised (oracle) | 0.912 | 0.930 | 0.751 | 0.904 | 0.855 | 0.838 | 0.874 | 0.879 |
| E3 FedAvg | 0.885 | 0.919 | 0.692 | 0.883 | 0.843 | 0.805 | 0.837 | 0.820 |

## Cost

| method | train time / run (s) | comm. total (MB) | comm. to best round (MB) |
|---|---|---|---|
| E1 Local-only | 32 | 0 | - |
| E2 Centralised (oracle) | 30 | n/a (data pooled) | - |
| E3 FedAvg | 30 | 102.3 | 93.4 |

FedAvg full-model payload: 213,040 B per client per direction (53,260 FP32 parameters, 12-class head).
FedAvg selected round (best val): [24, 25, 26, 28, 29] of 30.

## Natural-drift probe

Temporal split: train = first part of each activity bout, test = last part. No injected drift: any loss vs the blocked control is natural within-activity temporal shift.

Run `5b817cf` on NVIDIA GeForce RTX 5050 Laptop GPU | seeds [7, 42, 123, 999, 2024] | 8 clients (subjects [101, 102, 103, 104, 105, 106, 107, 108]) | split `temporal` | test split only.

## Headline (mean over seeds ± std; CI = 95% bootstrap over seeds)

| method | test macro-F1 | 95% CI | val macro-F1 (selected ckpt) | worst-client macro-F1 | accuracy | client std |
|---|---|---|---|---|---|---|
| E1 Local-only | 0.8082 ± 0.0057 | [0.8034, 0.8132] | 0.9710 ± 0.0060 | 0.7387 ± 0.0055 | 0.8256 ± 0.0061 | 0.0497 ± 0.0076 |
| E2 Centralised (oracle) | 0.8086 ± 0.0106 | [0.8002, 0.8178] | 0.9670 ± 0.0079 | 0.7320 ± 0.0344 | 0.8345 ± 0.0088 | 0.0585 ± 0.0084 |
| E3 FedAvg | 0.7721 ± 0.0085 | [0.7646, 0.7792] | 0.9147 ± 0.0041 | 0.6559 ± 0.0396 | 0.8092 ± 0.0060 | 0.0696 ± 0.0143 |

## Paired tests across seeds (macro-F1, Wilcoxon, Holm-adjusted)

| comparison | Δ macro-F1 | p (raw) | p (Holm) |
|---|---|---|---|
| E3_fedavg vs E1_local_only | -0.0361 | 0.0625 | 0.1875 |
| E2_centralized vs E3_fedavg | +0.0365 | 0.0625 | 0.1875 |
| E2_centralized vs E1_local_only | +0.0004 | 0.8125 | 0.8125 |

With n=5 seeds the smallest attainable two-sided Wilcoxon p is 0.0625; Holm over 3 tests multiplies it. Treat these tests as indicative only.

## Per-client macro-F1 (mean over seeds)

| method | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 |
|---|---|---|---|---|---|---|---|---|
| E1 Local-only | 0.739 | 0.810 | 0.830 | 0.878 | 0.764 | 0.764 | 0.870 | 0.811 |
| E2 Centralised (oracle) | 0.776 | 0.786 | 0.744 | 0.885 | 0.785 | 0.773 | 0.912 | 0.807 |
| E3 FedAvg | 0.769 | 0.739 | 0.679 | 0.847 | 0.781 | 0.716 | 0.882 | 0.763 |

## Cost

| method | train time / run (s) | comm. total (MB) | comm. to best round (MB) |
|---|---|---|---|
| E1 Local-only | 34 | 0 | - |
| E2 Centralised (oracle) | 31 | n/a (data pooled) | - |
| E3 FedAvg | 33 | 102.3 | 87.9 |

FedAvg full-model payload: 213,040 B per client per direction (53,260 FP32 parameters, 12-class head).
FedAvg selected round (best val): [21, 24, 25, 27, 27] of 30.

### Natural-shift gap (blocked − temporal, test macro-F1)

| method | blocked | temporal | gap |
|---|---|---|---|
| E1 Local-only | 0.8828 | 0.8082 | +0.0745 |
| E2 Centralised (oracle) | 0.8680 | 0.8086 | +0.0594 |
| E3 FedAvg | 0.8355 | 0.7721 | +0.0633 |

## Audit A14: effect of the wrong channel indices (E1, 30 epochs, 3 seeds)

| channels | macro-F1 | accuracy |
|---|---|---|
| correct ±16g accelerometer (9) | 0.8735 ± 0.0031 | 0.8849 ± 0.0010 |
| old code: temperature + 2 acc axes per IMU (9) | 0.9133 ± 0.0106 | 0.9268 ± 0.0147 |
| temperature sensors only (3) | 0.8356 ± 0.0148 | 0.8855 ± 0.0075 |