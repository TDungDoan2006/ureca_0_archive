# BlackElo Prediction Summary

- Best test model: `xgboost`
- Test RMSE: `207.51`
- Test MAE: `163.68`
- Test R^2: `0.3036`

## Top Correlated Numeric Features

- `BlackTotalPieceMovesRate`: correlation `0.1895`
- `BlackCastled`: correlation `0.1861`
- `CaptureCheckBalance`: correlation `0.1724`
- `TimeControlBaseSeconds`: correlation `-0.1644`
- `TotalChecksRate`: correlation `-0.1525`
- `BlackMinorMoves`: correlation `0.1516`
- `BlackTotalPieceMoves`: correlation `0.1509`
- `TimeControlIncrementSeconds`: correlation `-0.1495`
- `PlyCount`: correlation `0.1487`
- `BlackPawnMoves`: correlation `0.1472`

## Best-Model Permutation Importance

- `TimeControl`: importance `6486.83`
- `ECO`: importance `2410.70`
- `TimeControlIncrementSeconds`: importance `1900.92`
- `Termination`: importance `1675.46`
- `TimeControlBaseSeconds`: importance `1390.42`
- `Result`: importance `1328.95`
- `ECO_letter`: importance `1184.95`
- `Event`: importance `981.48`
- `TotalCapturesRate`: importance `724.38`
- `BlackTotalPieceMovesRate`: importance `605.25`

## Model Tuning Choices

- `dummy_mean`: {"strategy": "mean"}
- `ridge`: {"model__alpha": 0.1}
- `elastic_net`: {"model__alpha": 0.01, "model__l1_ratio": 0.9}
- `hist_gradient_boosting`: {"model__l2_regularization": 0.0, "model__learning_rate": 0.05, "model__max_depth": 6, "model__max_iter": 300, "model__max_leaf_nodes": 31, "model__min_samples_leaf": 20}
- `xgboost`: {"model__colsample_bytree": 0.7, "model__learning_rate": 0.05, "model__max_depth": 4, "model__min_child_weight": 3, "model__n_estimators": 350, "model__reg_alpha": 0.5, "model__reg_lambda": 5.0, "model__subsample": 1.0}
- `pytorch_mlp`: {"dropout": 0.2, "hidden_dims": [384, 192], "lr": 0.0008, "weight_decay": 0.0005}
