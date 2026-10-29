# Phase 1 results

## S0 stationary control: blocked split (train/val/test interleaved in time)

Run `4393dbd+dirty` on NVIDIA GeForce RTX 5050 Laptop GPU | seeds [7, 42, 123, 999, 2024] | 8 clients (subjects [101, 102, 103, 104, 105, 106, 107, 108]) | split `blocked` | test split only.

## Headline (mean over seeds ± std; CI = 95% bootstrap over seeds)

| method | test macro-F1 | 95% CI | val macro-F1 (selected ckpt) | worst-client macro-F1 | accuracy | client std |
|---|---|---|---|---|---|---|
| E1 Local-only | 0.8826 ± 0.0027 | [0.8800, 0.8847] | 0.9257 ± 0.0023 | 0.8300 ± 0.0139 | 0.8919 ± 0.0032 | 0.0266 ± 0.0045 |
| E2 Centralised (single global model) | 0.8711 ± 0.0075 | [0.8645, 0.8778] | 0.9088 ± 0.0044 | 0.7693 ± 0.0382 | 0.9139 ± 0.0019 | 0.0529 ± 0.0072 |
| E3 FedAvg | 0.8507 ± 0.0146 | [0.8388, 0.8644] | 0.8699 ± 0.0134 | 0.6761 ± 0.0838 | 0.8973 ± 0.0041 | 0.0780 ± 0.0242 |

## Paired tests across seeds (macro-F1, Wilcoxon, Holm-adjusted)

| comparison | Δ macro-F1 | p (raw) | p (Holm) |
|---|---|---|---|
| E3_fedavg vs E1_local_only | -0.0319 | 0.0625 | 0.1875 |
| E2_centralized vs E3_fedavg | +0.0204 | 0.0625 | 0.1875 |
| E2_centralized vs E1_local_only | -0.0115 | 0.1250 | 0.1875 |

With n=5 seeds the smallest attainable two-sided Wilcoxon p is 0.0625; Holm over 3 tests multiplies it. Treat these tests as indicative only.

## Per-client macro-F1 (mean over seeds)

| method | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 |
|---|---|---|---|---|---|---|---|---|
| E1 Local-only | 0.910 | 0.902 | 0.862 | 0.896 | 0.836 | 0.896 | 0.895 | 0.864 |
| E2 Centralised (single global model) | 0.916 | 0.935 | 0.769 | 0.889 | 0.876 | 0.831 | 0.876 | 0.877 |
| E3 FedAvg | 0.903 | 0.929 | 0.676 | 0.891 | 0.881 | 0.810 | 0.877 | 0.839 |

## Cost

| method | train time / run (s) | comm. total (MB) | comm. to best round (MB) |
|---|---|---|---|
| E1 Local-only | 95 | 0 | - |
| E2 Centralised (single global model) | 93 | n/a (data pooled) | - |
| E3 FedAvg | 90 | 204.5 | 145.2 |

FedAvg full-model payload: 213,040 B per client per direction (53,260 FP32 parameters, 12-class head).
FedAvg selected round (best val): [32, 38, 39, 42, 57] of 60.

## Natural-drift probe

Temporal split: train = first part of each activity bout, test = last part. No injected drift: any loss vs the blocked control is natural within-activity temporal shift.

Run `4393dbd+dirty` on NVIDIA GeForce RTX 5050 Laptop GPU | seeds [7, 42, 123, 999, 2024] | 8 clients (subjects [101, 102, 103, 104, 105, 106, 107, 108]) | split `temporal` | test split only.

## Headline (mean over seeds ± std; CI = 95% bootstrap over seeds)

| method | test macro-F1 | 95% CI | val macro-F1 (selected ckpt) | worst-client macro-F1 | accuracy | client std |
|---|---|---|---|---|---|---|
| E1 Local-only | 0.8147 ± 0.0051 | [0.8106, 0.8196] | 0.9768 ± 0.0038 | 0.7357 ± 0.0035 | 0.8328 ± 0.0055 | 0.0502 ± 0.0041 |
| E2 Centralised (single global model) | 0.8102 ± 0.0041 | [0.8070, 0.8142] | 0.9728 ± 0.0034 | 0.7263 ± 0.0074 | 0.8447 ± 0.0087 | 0.0574 ± 0.0059 |
| E3 FedAvg | 0.7735 ± 0.0130 | [0.7640, 0.7863] | 0.9392 ± 0.0058 | 0.6085 ± 0.0750 | 0.8168 ± 0.0076 | 0.0812 ± 0.0201 |

## Paired tests across seeds (macro-F1, Wilcoxon, Holm-adjusted)

| comparison | Δ macro-F1 | p (raw) | p (Holm) |
|---|---|---|---|
| E3_fedavg vs E1_local_only | -0.0412 | 0.0625 | 0.1875 |
| E2_centralized vs E3_fedavg | +0.0367 | 0.0625 | 0.1875 |
| E2_centralized vs E1_local_only | -0.0045 | 0.4375 | 0.4375 |

With n=5 seeds the smallest attainable two-sided Wilcoxon p is 0.0625; Holm over 3 tests multiplies it. Treat these tests as indicative only.

## Per-client macro-F1 (mean over seeds)

| method | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 |
|---|---|---|---|---|---|---|---|---|
| E1 Local-only | 0.737 | 0.808 | 0.838 | 0.873 | 0.776 | 0.766 | 0.881 | 0.840 |
| E2 Centralised (single global model) | 0.782 | 0.802 | 0.760 | 0.871 | 0.790 | 0.740 | 0.911 | 0.825 |
| E3 FedAvg | 0.773 | 0.772 | 0.609 | 0.820 | 0.790 | 0.719 | 0.886 | 0.819 |

## Cost

| method | train time / run (s) | comm. total (MB) | comm. to best round (MB) |
|---|---|---|---|
| E1 Local-only | 95 | 0 | - |
| E2 Centralised (single global model) | 93 | n/a (data pooled) | - |
| E3 FedAvg | 93 | 204.5 | 177.2 |

FedAvg full-model payload: 213,040 B per client per direction (53,260 FP32 parameters, 12-class head).
FedAvg selected round (best val): [43, 50, 51, 53, 58] of 60.

### Natural-shift gap (blocked − temporal, test macro-F1)

| method | blocked | temporal | gap |
|---|---|---|---|
| E1 Local-only | 0.8826 | 0.8147 | +0.0678 |
| E2 Centralised (single global model) | 0.8711 | 0.8102 | +0.0609 |
| E3 FedAvg | 0.8507 | 0.7735 | +0.0772 |

## Budget sensitivity (blocked split, test macro-F1, mean ± std over seeds)

Initial budget: 60 epochs / 30 rounds. Locked budget: 120 epochs / 60 rounds (x2).

| method | initial budget | locked budget |
|---|---|---|
| E1 Local-only | 0.8828 ± 0.0038 | 0.8826 ± 0.0027 |
| E2 Centralised (single global model) | 0.8680 ± 0.0093 | 0.8711 ± 0.0075 |
| E3 FedAvg | 0.8355 ± 0.0099 | 0.8507 ± 0.0146 |

FedAvg was still improving at the initial budget (best round 24-29 of 30); at the locked budget it converges well before the last round.

## E3b FedAvg + local fine-tuning (blocked split)

Global FedAvg model fine-tuned per client for <= 20 epochs (best val macro-F1). FedAvg part reproduces the frozen E3 results exactly: **True**.

| method | macro-F1 | worst-client macro-F1 | accuracy |
|---|---|---|---|
| E1 Local-only | 0.8826 ± 0.0027 | 0.8300 ± 0.0139 | 0.8919 ± 0.0032 |
| E3 FedAvg | 0.8507 ± 0.0146 | 0.6761 ± 0.0838 | 0.8973 ± 0.0041 |
| E3b FedAvg + FT | 0.8879 ± 0.0089 | 0.8144 ± 0.0289 | 0.9095 ± 0.0045 |

E3b - E1: +0.0053 (95% CI [-0.0021, +0.0120]).

## E3b FedAvg + local fine-tuning (temporal split)

Global FedAvg model fine-tuned per client for <= 20 epochs (best val macro-F1). FedAvg part reproduces the frozen E3 results exactly: **True**.

| method | macro-F1 | worst-client macro-F1 | accuracy |
|---|---|---|---|
| E1 Local-only | 0.8147 ± 0.0051 | 0.7357 ± 0.0035 | 0.8328 ± 0.0055 |
| E3 FedAvg | 0.7735 ± 0.0130 | 0.6085 ± 0.0750 | 0.8168 ± 0.0076 |
| E3b FedAvg + FT | 0.8150 ± 0.0074 | 0.7430 ± 0.0436 | 0.8379 ± 0.0050 |

E3b - E1: +0.0003 (95% CI [-0.0047, +0.0045]).

## Audit A14: effect of the wrong channel indices (E1, 30 epochs, 3 seeds)

| channels | macro-F1 | accuracy |
|---|---|---|
| correct ±16g accelerometer (9) | 0.8735 ± 0.0031 | 0.8849 ± 0.0010 |
| old code: temperature + 2 acc axes per IMU (9) | 0.9133 ± 0.0106 | 0.9268 ± 0.0147 |
| temperature sensors only (3) | 0.8356 ± 0.0148 | 0.8855 ± 0.0075 |