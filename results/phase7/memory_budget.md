# On-device memory budget (A6)

FP32. One window = 256 x 9 x 4 B = 9.0 kB; about 2 labelled windows arrive per round; median client train pool = 966 windows (range 694-1126).

| component | bytes | kB |
|---|---|---|
| model (FP32, deployed) | 213,040 | 208.0 |
| adapter (input map + bottleneck + head) | 7,864 | 7.7 |
| input map only (align fit) | 360 | 0.4 |
| Page-Hinkley detector state | 64 | 0.1 |
| labelled arrival buffer: align (lookback 8 + episode 8 rounds) | 294,912 | 288.0 |
| labelled arrival buffer: head_long (45 rounds) | 829,440 | 810.0 |
| labelled arrival buffer: federated retraining (15 rounds) | 276,480 | 270.0 |
| replay pool (client train windows, median client) | 8,902,656 | 8694.0 |
| memory cache, K = 1 adapter states | 7,864 | 7.7 |
| memory cache, K = 2 adapter states | 15,728 | 15.4 |
| memory cache, K = 4 adapter states | 31,456 | 30.7 |

| system (peak, per client) | kB |
|---|---|
| E3b frozen | 208 |
| E5 / E7a (PH + full fed retraining) | 9380 |
| PH + federated adapter | 9180 |
| E8 (typed: align / head_long) | 9720 |
| E8 + memory K = 0 | 9720 |
| E8 + memory K = 1 | 9727 |
| E8 + memory K = 2 | 9735 |
| E8 + memory K = 4 | 9751 |

The replay pool dominates every adapting system; the memory cache of E8+memory adds 7.7 kB per entry, under 2% of the total even at K = 4. The simulator keeps the whole arrival history; a device needs only the buffer listed above.
