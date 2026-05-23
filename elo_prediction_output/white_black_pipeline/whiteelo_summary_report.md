# WhiteElo Prediction Summary

- Best test model: `xgboost`
- Test RMSE: `205.40`
- Test MAE: `162.69`
- Test R^2: `0.2454`

## Top Correlated Numeric Features

- `TimeControlBaseSeconds`: correlation `-0.1674`
- `WhiteCastled`: correlation `0.1558`
- `TotalChecksRate`: correlation `-0.1430`
- `CaptureCheckBalance`: correlation `0.1328`
- `WhiteMinorMoves`: correlation `0.1291`
- `WhiteEarlyQueenMove`: correlation `-0.1239`
- `TimeControlIncrementSeconds`: correlation `-0.1228`
- `WhitePawnMoves`: correlation `0.1214`
- `MoveCount`: correlation `0.1180`
- `WhiteTotalPieceMoves`: correlation `0.1180`

## Best-Model Permutation Importance

- `TimeControlBaseSeconds`: importance `4149.12`
- `TimeControlIncrementSeconds`: importance `3316.36`
- `TimeControl`: importance `2963.11`
- `ECO`: importance `1715.10`
- `Result`: importance `1389.15`
- `Termination`: importance `1277.07`
- `Event`: importance `805.84`
- `ECO_letter`: importance `748.43`
- `TotalCapturesRate`: importance `547.12`
- `TotalChecksRate`: importance `500.30`

## Model Tuning Choices

- `dummy_mean`: {"strategy": "mean"}
- `ridge`: {"model__alpha": 100.0}
- `elastic_net`: {"model__alpha": 0.1, "model__l1_ratio": 0.7}
- `hist_gradient_boosting`: {"model__l2_regularization": 0.5, "model__learning_rate": 0.03, "model__max_depth": 4, "model__max_iter": 200, "model__max_leaf_nodes": 31, "model__min_samples_leaf": 20}
- `xgboost`: {"model__colsample_bytree": 0.7, "model__learning_rate": 0.05, "model__max_depth": 4, "model__min_child_weight": 3, "model__n_estimators": 350, "model__reg_alpha": 0.5, "model__reg_lambda": 5.0, "model__subsample": 1.0}
- `pytorch_mlp`: {"dropout": 0.15, "hidden_dims": [256, 128], "lr": 0.001, "weight_decay": 0.0001}
