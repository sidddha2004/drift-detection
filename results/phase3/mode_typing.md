# Mode (virtual vs real) identification from detected alarms

927 episodes (568 virtual, 359 real), leave-one-seed-out over seeds 1001-1010.

| classifier | accuracy | recall virtual | recall real | acc low | acc med | acc high |
|---|---|---|---|---|---|---|
| majority | 0.613 | 1.00 | 0.00 | 0.43 | 0.58 | 0.68 |
| rule dF_delta>0.05 | 0.655 | 0.61 | 0.73 | 0.58 | 0.62 | 0.70 |
| logreg label-free | 0.731 | 0.71 | 0.76 | 0.64 | 0.69 | 0.78 |
| gboost label-free | 0.845 | 0.92 | 0.72 | 0.72 | 0.82 | 0.89 |
| gboost all features | 0.841 | 0.93 | 0.70 | 0.73 | 0.82 | 0.89 |