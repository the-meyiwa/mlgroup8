# Error distribution: is the 4 MPa a p50 or a p99 problem?

Both, and that is the point. The 4.04 -> 3.79 headline is RMSE, a single average
that hides the shape of the error. Out-of-fold absolute error, shipped folds,
seed 42, MPa:

| model | rmse | p50 | p90 | p95 | p99 | max | within 5 MPa |
|---|---|---|---|---|---|---|---|
| linear | 10.477 | 6.593 | 17.520 | 20.774 | 27.102 | 34.454 | 38.1% |
| random forest | 5.040 | 2.464 | 7.827 | 9.850 | 16.735 | 36.271 | 75.6% |
| GBR v1 (defaults) | 4.039 | 1.921 | 5.864 | 7.803 | 14.500 | 30.700 | 85.8% |
| GBR v2 (shipped) | 3.792 | 1.606 | 5.821 | 7.880 | 13.453 | 31.301 | 87.5% |

Half of all mixes are predicted within 1.6 MPa. Nine in ten within 5.8. The
tail is unchanged: p99 went 14.50 -> 13.45 MPa and the single worst error got
marginally worse, 30.7 -> 31.3.

## How much of the gain is real

Tuning was selected on the same folds it is scored on, so 3.792 is optimistic.
Repeating the whole 10-fold protocol on 10 independent partitions:

| model | mean RMSE | min | max |
|---|---|---|---|
| v1 | 4.041 | 3.909 | 4.194 |
| v2 | 3.885 | 3.787 | 4.020 |

The honest number for the shipped model is 3.89, not 3.79, and the improvement
over defaults is about 0.16 MPa on average, not 0.25. The partitions barely
overlap, so the gain is real but small.

## Where the tail comes from

Not from one place. The worst row (31.3 MPa error) is a 7-day, 362.6 kg/m3
cement, 22.9 MPa sample predicted at 54.2, an outlier the model has no basis
for. Repeating mixes are rare (19 mixes, 53 rows) but one mix at 362.6 cement /
189 slag has four samples spanning 22.90 to 55.90 MPa. Same inputs, 33 MPa
apart. No model can fit that; it is label noise or undocumented curing
differences.

## Consequence for the MVP

Report per-sample spread, not one number: the UI already returns a single
estimate, which reads as more precision than the model has. A conformal
interval from the out-of-fold residuals (roughly +/-1.6 MPa at 50%, +/-13 MPa
at 99%) would be cheap to add and more honest than a bare figure.
