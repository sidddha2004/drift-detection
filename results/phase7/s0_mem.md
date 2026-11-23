# S0 control and memory on S3 (descriptive, 10 evaluation seeds)

## S0 (no drift)

| system | seeds | alarms / stream (8 clients) | F1 whole stream | diff vs Personalized [95% CI] | uplink kB |
|---|---|---|---|---|---|
| frozen_E3b_personal | 10 | 0.00 | 0.939 | +0.0000 [+0.0000, +0.0000] | 0 |
| periodic_E3c_full_fed | 10 | 88.00 | 0.899 | -0.0407 [-0.0475, -0.0345] | 18748 |
| ph_adapter_fed | 10 | 1.50 | 0.936 | -0.0032 [-0.0071, -0.0002] | 34 |
| ph_align | 10 | 1.40 | 0.934 | -0.0051 [-0.0107, -0.0011] | 0 |
| ph_full_fed_E5 | 10 | 1.20 | 0.936 | -0.0032 [-0.0054, -0.0012] | 724 |
| ph_head_long | 10 | 1.00 | 0.938 | -0.0009 [-0.0020, -0.0002] | 0 |
| typed_learned | 10 | 0.90 | 0.938 | -0.0010 [-0.0023, -0.0001] | 0 |

## Memory on S3 (paired with TLA = typed_learned from the confirmatory runs)

| system | seeds | switches | post F1 (TLA) | diff [CI] p | off-phase F1 (TLA) | diff [CI] p |
|---|---|---|---|---|---|---|
| typed_memory_K0_t02 | 10 | 8 | 0.709 (0.711) | -0.0023 [-0.0113, +0.0033] p=0.383 | 0.785 (0.784) | +0.0008 [-0.0122, +0.0114] p=0.383 |
| typed_memory_K4_t02 | 10 | 8 | 0.709 (0.711) | -0.0023 [-0.0113, +0.0033] p=0.383 | 0.785 (0.784) | +0.0008 [-0.0122, +0.0114] p=0.383 |
