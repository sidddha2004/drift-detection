# Phase 1d: validation of the 5 s bout-edge trim

| client | windows v1 | windows v2 | kept | classes v1 | classes v2 |
|---|---|---|---|---|---|
| 101 | 1856 | 1754 | 0.95 | 12 | 12 |
| 102 | 1963 | 1854 | 0.94 | 12 | 12 |
| 103 | 1299 | 1213 | 0.93 | 8 | 8 |
| 104 | 1723 | 1632 | 0.95 | 10 | 10 |
| 105 | 2033 | 1929 | 0.95 | 12 | 12 |
| 106 | 1691 | 1599 | 0.95 | 10 | 10 |
| 107 | 1740 | 1637 | 0.94 | 11 | 11 |
| 108 | 1953 | 1844 | 0.94 | 12 | 12 |

Coverage floor (fixed before looking): train>=20, val>=5, test>=5 windows per client x class.

- (client, class) pairs lost entirely by trimming: **0** []
- pairs below the floor after trimming: **5** [(105, 8, 'train', 19), (105, 11, 'test', 2), (107, 4, 'train', 12), (107, 4, 'val', 3), (107, 4, 'test', 3)]

## Are edge windows atypical of their class? (median distance to class train centroid, train-std units)

edge 2.335 vs interior 0.710; edge farther in 99% of 84 client x class pairs. (Supporting evidence only; the rule is justified by label alignment, not by this.)

## Is the covariate content of the retained windows unchanged? (RBF-MMD^2, v1 vs v2, per client x class)

mean 0.0384, max 0.5718 (reference: train-vs-test MMD^2 under the blocked split in Phase 1b, see natural_drift.md).