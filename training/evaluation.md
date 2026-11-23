# Evaluation report - GradientBoostingRegressor

Artefact: `model.joblib`  
Model version: `361e8cbdf`  
Dataset: 1030 rows, sha256 `483dc12a5ad09f97...`  
Protocol: 10-fold cross-validation, identical folds for all candidates, seed 42

| Model | CV RMSE (MPa) | CV MAE (MPa) | CV R2 |
| --- | ---: | ---: | ---: |
| GradientBoostingRegressor | 4.039 | 2.723 | 0.9415 |
| RandomForestRegressor | 5.040 | 3.500 | 0.9089 |
| LinearRegression | 10.477 | 8.299 | 0.6063 |

Selected on lowest cross-validated RMSE: **GradientBoostingRegressor**.  
RMSE is the selection metric because a 5 MPa miss matters the same whether it is over or under.

## Per-split view

Cross-validated predictions pooled across all folds; the numbers above are computed on the pooled set.
