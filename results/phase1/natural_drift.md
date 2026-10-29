# Phase 1b: natural temporal shift in PAMAP2

Local-only LSTM, 60 epochs, seeds [42, 123, 999], 8 clients. The temporal split loses ~6-8 macro-F1 vs blocked; why?

## (a) Input-distribution shift

Unbiased RBF-MMD^2 between train and test window features (per client x activity; mean over pairs):

| split | MMD^2 | pairs |
|---|---|---|
| blocked | 0.0710 | 84 |
| temporal | 0.1508 | 86 |

Median distance of a window's features to its class's train centroid (train-std units) by position in bout (temporal split; train covers 0-0.6, val ~0.6-0.75, test 0.75-1):

| position decile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| median distance | 1.10 | 0.65 | 0.63 | 0.62 | 0.64 | 0.67 | 0.70 | 0.74 | 0.77 | 1.02 |

## (b) Are late windows intrinsically harder? Test error rate by position in the bout

| split | 0.00-0.25 | 0.25-0.50 | 0.50-0.75 | 0.75-1.00 |
|---|---|---|---|---|
| blocked | 0.168 | 0.044 | 0.055 | 0.137 |
| temporal | nan | nan | 0.000 | 0.175 |

If the blocked model is also worse late in the bout, late windows are intrinsically harder (e.g. label noise near activity ends). If it is flat while the temporal model is bad, the loss is extrapolation to shifted input.

## (c) Class frequencies

TV distance of train vs test label histograms: blocked 0.0866, temporal 0.0091.

## (d) Sensor warm-up

Mean temperature rise over a bout: -0.01 C. Spearman(bout temperature rise, temporal-split error) = -0.064 (p=0.521, n=104 bouts).