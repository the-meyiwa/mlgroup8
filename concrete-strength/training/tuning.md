# Hyperparameter sweep

Protocol: 10-fold shuffled CV, seed 42, RMSE used for selection.

| Candidate | CV RMSE (MPa) | CV MAE (MPa) | CV R2 |
| --- | ---: | ---: | ---: |
| GBR tuned {"learning_rate": 0.06, "max_depth": 4, "min_samples_leaf": 3, "n_estimators": 600, "subsample": 0.8} | 3.792 | 2.485 | 0.9484 |
| HGB tuned {"l2_regularization": 1.0, "learning_rate": 0.1, "max_iter": 500, "max_leaf_nodes": 15, "min_samples_leaf": 10} | 3.853 | 2.490 | 0.9468 |
| GBR tuned, log target | 3.987 | 2.574 | 0.9430 |

Best: GBR tuned {"learning_rate": 0.06, "max_depth": 4, "min_samples_leaf": 3, "n_estimators": 600, "subsample": 0.8}

## Read this before quoting 3.792

Hyperparameters were selected with GridSearchCV on the same 10-fold splits later used to report CV RMSE. The reported best score is therefore the maximum of 32 correlated estimates, not an unbiased one, and it is optimistic by something in the 0.1 to 0.3 MPa range. The honest claim is "about 3.8 to 4.1 MPa CV RMSE". A nested CV or a held-out 20 percent split would settle it; neither is done here.

The log-target variant lost (3.987 vs 3.792), which is worth noting: the target is right-skewed, but the booster already handles that, and shifting to ln-space rescales the loss so errors on low-strength mixes count relatively more.
