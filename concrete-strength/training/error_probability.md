# Probability of a 4 MPa error

Out-of-fold predictions, shipped model, 10-fold CV seed 42, n=1030.

| threshold | share of predictions |
|---|---|
| error > 2 MPa | 41.4% |
| error > 3 MPa | 27.8% |
| error > 4 MPa | 18.1% |
| error > 5 MPa | 12.5% |
| error > 8 MPa | 5.0% |
| error > 10 MPa | 2.8% |

Within +/-4 MPa: 81.9%. Within +/-5 MPa: 87.5%.
Median absolute error 1.61 MPa. RMSE 3.79.
Mean signed error -0.00 MPa (positive = overpredicts). Overpredicts by >4 MPa 9.0%, underpredicts by >4 MPa 9.0%.
Top 5% of errors carry 53% of the squared error.

| rows | n | median abs err | %>4 MPa | %>8 MPa |
|---|---|---|---|---|
| unique mix rows | 1010 | 1.60 | 17.9% | 4.8% |
| duplicate mix rows | 20 | 2.09 | 25.0% | 15.0% |

| true strength | n | median abs err | %>4 MPa |
|---|---|---|---|
| 0-25 MPa | 295 | 1.42 | 15.3% |
| 25-40 MPa | 356 | 1.63 | 15.7% |
| 40-55 MPa | 232 | 1.89 | 23.7% |
| 55-90 MPa | 147 | 1.35 | 20.4% |
