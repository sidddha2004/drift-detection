# Phase 2: severity check (frozen local model per client, test pool, alpha = 1, seed 1001, 8 clients)

Clean (no drift) macro-F1: mean 0.940, worst client 0.906

| transform | scope | low | med | high |
|---|---|---|---|---|
| offset | all sensors | 0.913 (worst 0.87) | 0.797 (worst 0.66) | 0.462 (worst 0.26) |
| offset | hand only | 0.932 (worst 0.89) | 0.912 (worst 0.84) | 0.783 (worst 0.57) |
| gain | all sensors | 0.911 (worst 0.88) | 0.813 (worst 0.75) | 0.656 (worst 0.52) |
| gain | ankle only | 0.923 (worst 0.88) | 0.825 (worst 0.69) | 0.612 (worst 0.47) |
| rotation | all sensors | 0.880 (worst 0.83) | 0.678 (worst 0.55) | 0.421 (worst 0.32) |
| rotation | chest only | 0.892 (worst 0.80) | 0.746 (worst 0.64) | 0.563 (worst 0.42) |
| label_remap | real drift | 0.761 (worst 0.67) | 0.584 (worst 0.43) | 0.420 (worst 0.29) |

Monotone (low >= med >= high): offset|all sensors=True, offset|hand only=True, gain|all sensors=True, gain|ankle only=True, rotation|all sensors=True, rotation|chest only=True, label_remap|real drift=True