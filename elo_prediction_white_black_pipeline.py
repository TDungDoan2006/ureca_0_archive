from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


ROOT = Path("/Users/phamhoanghai/Ureca")
INPUT_PATH = ROOT / "data_manipulation_output" / "output_data_cleaned.csv"
OUTPUT_DIR = ROOT / "elo_prediction_output" / "white_black_pipeline"
PLOT_DIR = ROOT / "elo_prediction_plots" / "white_black_pipeline"

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)


@dataclass
class SearchResult:
    name: str
    estimator: object
    best_params: dict
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_predictions: np.ndarray
    test_predictions: np.ndarray


def make_one_hot_encoder(min_frequency: int) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frequency,
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_dataset(max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = df.sort_values("GameID").reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_STATE).sort_values("GameID").reset_index(drop=True)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    move_den = out["MoveCount"].replace(0, np.nan)

    out["ECO_letter"] = out["ECO"].fillna("Unknown").astype(str).str[0]

    out["WhiteMinorMoves"] = out["WhiteKnightMoves"] + out["WhiteBishopMoves"]
    out["BlackMinorMoves"] = out["BlackKnightMoves"] + out["BlackBishopMoves"]
    out["WhiteHeavyMoves"] = out["WhiteRookMoves"] + out["WhiteQueenMoves"]
    out["BlackHeavyMoves"] = out["BlackRookMoves"] + out["BlackQueenMoves"]
    out["WhiteTotalPieceMoves"] = (
        out["WhitePawnMoves"]
        + out["WhiteKnightMoves"]
        + out["WhiteBishopMoves"]
        + out["WhiteRookMoves"]
        + out["WhiteQueenMoves"]
        + out["WhiteKingMoves"]
    )
    out["BlackTotalPieceMoves"] = (
        out["BlackPawnMoves"]
        + out["BlackKnightMoves"]
        + out["BlackBishopMoves"]
        + out["BlackRookMoves"]
        + out["BlackQueenMoves"]
        + out["BlackKingMoves"]
    )

    ratio_specs = [
        "WhitePawnMoves",
        "BlackPawnMoves",
        "WhiteKnightMoves",
        "BlackKnightMoves",
        "WhiteBishopMoves",
        "BlackBishopMoves",
        "WhiteRookMoves",
        "BlackRookMoves",
        "WhiteQueenMoves",
        "BlackQueenMoves",
        "WhiteKingMoves",
        "BlackKingMoves",
        "WhiteMinorMoves",
        "BlackMinorMoves",
        "WhiteHeavyMoves",
        "BlackHeavyMoves",
        "WhiteTotalPieceMoves",
        "BlackTotalPieceMoves",
        "TotalCaptures",
        "TotalChecks",
        "TotalPromotions",
    ]
    for column in ratio_specs:
        out[f"{column}Rate"] = out[column] / move_den

    out["WhiteQueenPressureRatio"] = out["WhiteQueenMoves"] / (out["WhiteMinorMoves"] + 1)
    out["BlackQueenPressureRatio"] = out["BlackQueenMoves"] / (out["BlackMinorMoves"] + 1)
    out["OpeningCaptureRate12Ply"] = out["OpeningCaptureCount12Ply"] / 12.0
    out["OpeningCheckRate12Ply"] = out["OpeningCheckCount12Ply"] / 12.0
    out["CaptureCheckBalance"] = out["TotalCaptures"] - out["TotalChecks"]

    return out


def get_feature_target_frames(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].astype(float).copy()
    drop_cols = [
        "WhiteElo",
        "BlackElo",
        "AvgElo",
        "EloDiff",
        "WhiteRatingDiff",
        "BlackRatingDiff",
        "MovesSAN",
        "MovesUCI",
        "FinalFEN",
        "ExtraHeaders",
        "Date",
        "UTCDate",
        "UTCTime",
        "GameDate",
        "GameDateSource",
        "Site",
        "White",
        "Black",
        "Opening",
        "GameID",
    ]

    feature_df = df.drop(columns=[column for column in drop_cols if column in df.columns]).copy()
    X = feature_df.copy()
    return X, y


def split_frames(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    split_1 = int(len(X) * TRAIN_FRAC)
    split_2 = int(len(X) * (TRAIN_FRAC + VAL_FRAC))

    X_train = X.iloc[:split_1].reset_index(drop=True)
    X_val = X.iloc[split_1:split_2].reset_index(drop=True)
    X_test = X.iloc[split_2:].reset_index(drop=True)
    y_train = y.iloc[:split_1].to_numpy()
    y_val = y.iloc[split_1:split_2].to_numpy()
    y_test = y.iloc[split_2:].to_numpy()
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_target_correlations(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_name: str,
    top_k: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    corr_rows = []
    series_y = pd.Series(y_train)
    for column in numeric_cols:
        corr = X_train[column].corr(series_y)
        corr_rows.append(
            {
                "feature": column,
                "correlation": float(0.0 if pd.isna(corr) else corr),
                "abs_correlation": float(abs(0.0 if pd.isna(corr) else corr)),
            }
        )

    corr_df = pd.DataFrame(corr_rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    corr_df.to_csv(OUTPUT_DIR / f"{target_name.lower()}_target_correlations.csv", index=False)

    top_features = corr_df.head(top_k)["feature"].tolist()
    heatmap_features = top_features[: min(len(top_features), 12)]
    if heatmap_features:
        heatmap_df = X_train[heatmap_features].copy()
        heatmap_df[target_name] = y_train
        corr_matrix = heatmap_df.corr(numeric_only=True)
        corr_matrix.to_csv(OUTPUT_DIR / f"{target_name.lower()}_top_feature_correlation_matrix.csv")

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
        plt.title(f"{target_name}: Top Numeric Feature Correlations")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{target_name.lower()}_correlation_heatmap.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.barplot(data=corr_df.head(top_k), x="correlation", y="feature", color="#2f6c8f")
        plt.title(f"{target_name}: Top Feature Correlations With Target")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{target_name.lower()}_top_correlations.png", dpi=200)
        plt.close()

    return corr_df, numeric_cols


def select_numeric_features_by_correlation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: list[str],
    threshold: float,
    target_name: str,
) -> tuple[list[str], pd.DataFrame]:
    corr_to_target = {}
    y_series = pd.Series(y_train)
    for column in numeric_cols:
        corr = X_train[column].corr(y_series)
        corr_to_target[column] = abs(float(0.0 if pd.isna(corr) else corr))

    numeric_frame = X_train[numeric_cols].copy()
    abs_corr_matrix = numeric_frame.corr(numeric_only=True).abs()
    upper = abs_corr_matrix.where(np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool))

    dropped = set()
    decisions = []
    for column in upper.columns:
        if column in dropped:
            continue
        correlated = upper[column][upper[column] > threshold]
        for other_feature, pair_corr in correlated.items():
            if other_feature in dropped or column in dropped:
                continue
            if corr_to_target[column] >= corr_to_target[other_feature]:
                drop_feature = other_feature
                keep_feature = column
            else:
                drop_feature = column
                keep_feature = other_feature
            dropped.add(drop_feature)
            decisions.append(
                {
                    "keep_feature": keep_feature,
                    "drop_feature": drop_feature,
                    "pair_correlation": float(pair_corr),
                    "keep_abs_target_corr": corr_to_target[keep_feature],
                    "drop_abs_target_corr": corr_to_target[drop_feature],
                }
            )
            if column in dropped:
                break

    selected = [column for column in numeric_cols if column not in dropped]
    decision_df = pd.DataFrame(decisions).sort_values("pair_correlation", ascending=False).reset_index(drop=True)
    decision_df.to_csv(OUTPUT_DIR / f"{target_name.lower()}_correlation_filter_decisions.csv", index=False)
    return selected, decision_df


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    min_category_count: int,
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder(min_category_count)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0,
    )


def run_grid_search(
    name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv,
    n_jobs: int,
) -> SearchResult:
    total_candidates = int(np.prod([len(values) for values in param_grid.values()])) if param_grid else 0
    print(f"[{name}] starting grid search with {total_candidates} candidates")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"[{name}] best params: {search.best_params_}")

    best_estimator = search.best_estimator_
    val_pred = best_estimator.predict(X_val)
    test_pred = best_estimator.predict(X_test)
    return SearchResult(
        name=name,
        estimator=best_estimator,
        best_params=search.best_params_,
        val_metrics=regression_metrics(y_val, val_pred),
        test_metrics=regression_metrics(y_test, test_pred),
        val_predictions=val_pred,
        test_predictions=test_pred,
    )


def run_sampled_search(
    name: str,
    pipeline: Pipeline,
    param_distributions: dict,
    n_iter: int,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cv,
    n_jobs: int,
) -> SearchResult:
    candidates = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=RANDOM_STATE))
    print(f"[{name}] starting sampled search with {len(candidates)} candidates")
    best_score = float("inf")
    best_estimator = None
    best_params = None

    for params in candidates:
        fold_scores = []
        for train_idx, holdout_idx in cv.split(X_train):
            estimator = clone(pipeline)
            estimator.set_params(**params)
            estimator.fit(X_train.iloc[train_idx], y_train[train_idx])
            fold_pred = estimator.predict(X_train.iloc[holdout_idx])
            fold_rmse = float(np.sqrt(mean_squared_error(y_train[holdout_idx], fold_pred)))
            fold_scores.append(fold_rmse)

        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            best_estimator = clone(pipeline)
            best_estimator.set_params(**params)
            best_estimator.fit(X_train, y_train)
            print(f"[{name}] new best CV RMSE: {best_score:.2f}")

    if best_estimator is None or best_params is None:
        raise RuntimeError(f"Search for {name} failed to produce a model.")

    val_pred = best_estimator.predict(X_val)
    test_pred = best_estimator.predict(X_test)
    return SearchResult(
        name=name,
        estimator=best_estimator,
        best_params=best_params,
        val_metrics=regression_metrics(y_val, val_pred),
        test_metrics=regression_metrics(y_test, test_pred),
        val_predictions=val_pred,
        test_predictions=test_pred,
    )


def run_dummy_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> SearchResult:
    print("[dummy_mean] fitting baseline")
    model = DummyRegressor(strategy="mean")
    model.fit(np.zeros((len(X_train), 1)), y_train)
    val_pred = model.predict(np.zeros((len(X_val), 1)))
    test_pred = model.predict(np.zeros((len(X_test), 1)))
    return SearchResult(
        name="dummy_mean",
        estimator=model,
        best_params={"strategy": "mean"},
        val_metrics=regression_metrics(y_val, val_pred),
        test_metrics=regression_metrics(y_test, test_pred),
        val_predictions=val_pred,
        test_predictions=test_pred,
    )


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_torch_model(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target_name: str,
    epochs: int,
) -> SearchResult | None:
    if not TORCH_AVAILABLE:
        return None
    print("[pytorch_mlp] preparing dense tensors")

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    target_mean = float(y_train.mean())
    target_std = float(y_train.std()) if float(y_train.std()) > 0 else 1.0
    y_train_scaled = (y_train - target_mean) / target_std
    y_val_scaled = (y_val - target_mean) / target_std

    X_train_tensor = torch.tensor(X_train_t, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_t, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_t, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=1024, shuffle=True)

    candidate_configs = [
        {"hidden_dims": [256, 128], "dropout": 0.15, "lr": 1e-3, "weight_decay": 1e-4},
        {"hidden_dims": [384, 192], "dropout": 0.20, "lr": 8e-4, "weight_decay": 5e-4},
        {"hidden_dims": [256, 128, 64], "dropout": 0.10, "lr": 1e-3, "weight_decay": 1e-5},
    ]

    best_state = None
    best_config = None
    best_val_rmse = float("inf")
    history_rows = []

    for config_id, config in enumerate(candidate_configs, start=1):
        print(f"[pytorch_mlp] trying config {config_id}: {config}")
        model = TorchMLP(X_train_t.shape[1], config["hidden_dims"], config["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        loss_fn = nn.MSELoss()
        patience = 10
        patience_left = patience
        local_best_state = None
        local_best_val = float("inf")

        X_val_device = X_val_tensor.to(device)
        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred_scaled = model(X_val_device).cpu().numpy().ravel()
            val_pred = val_pred_scaled * target_std + target_mean
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            history_rows.append(
                {
                    "target": target_name,
                    "config_id": config_id,
                    "epoch": epoch,
                    "val_rmse": val_rmse,
                }
            )

            if val_rmse < local_best_val:
                local_best_val = val_rmse
                local_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1

            if patience_left == 0:
                break

        if local_best_state is not None and local_best_val < best_val_rmse:
            best_val_rmse = local_best_val
            best_state = local_best_state
            best_config = config
            print(f"[pytorch_mlp] new best validation RMSE: {best_val_rmse:.2f}")

    if best_state is None or best_config is None:
        return None

    final_model = TorchMLP(X_train_t.shape[1], best_config["hidden_dims"], best_config["dropout"]).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()

    with torch.no_grad():
        val_pred_scaled = final_model(X_val_tensor.to(device)).cpu().numpy().ravel()
        test_pred_scaled = final_model(X_test_tensor.to(device)).cpu().numpy().ravel()

    val_pred = val_pred_scaled * target_std + target_mean
    test_pred = test_pred_scaled * target_std + target_mean

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(OUTPUT_DIR / f"{target_name.lower()}_torch_history.csv", index=False)

    return SearchResult(
        name="pytorch_mlp",
        estimator=final_model,
        best_params=best_config,
        val_metrics=regression_metrics(y_val, val_pred),
        test_metrics=regression_metrics(y_test, test_pred),
        val_predictions=val_pred,
        test_predictions=test_pred,
    )


def extract_linear_coefficients(search_result: SearchResult, target_name: str) -> None:
    if not isinstance(search_result.estimator, Pipeline):
        return
    model = search_result.estimator.named_steps["model"]
    if not hasattr(model, "coef_"):
        return
    preprocessor = search_result.estimator.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coef_df = (
        pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    coef_df.to_csv(OUTPUT_DIR / f"{target_name.lower()}_{search_result.name}_coefficients.csv", index=False)


def save_permutation_importance(
    search_result: SearchResult,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target_name: str,
) -> pd.DataFrame:
    if not isinstance(search_result.estimator, Pipeline):
        return pd.DataFrame(columns=["feature", "importance"])

    perm_rows = min(len(X_val), 5000)
    perm = permutation_importance(
        search_result.estimator,
        X_val.iloc[:perm_rows],
        y_val[:perm_rows],
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_mean_squared_error",
    )
    importance_df = (
        pd.DataFrame({"feature": X_val.columns, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(OUTPUT_DIR / f"{target_name.lower()}_best_model_permutation_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df.head(20), x="importance", y="feature", color="#4c956c")
    plt.title(f"{target_name}: Best Model Permutation Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{target_name.lower()}_best_model_importance.png", dpi=200)
    plt.close()
    return importance_df


def save_prediction_plots(
    target_name: str,
    y_test: np.ndarray,
    predictions_df: pd.DataFrame,
    best_model_name: str,
) -> None:
    pred_col = f"pred_{best_model_name}"
    residuals = predictions_df[pred_col] - y_test

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(y_test, predictions_df[pred_col], alpha=0.18, s=10)
    lims = [
        min(float(y_test.min()), float(predictions_df[pred_col].min())),
        max(float(y_test.max()), float(predictions_df[pred_col].max())),
    ]
    axes[0].plot(lims, lims, "k--", linewidth=1)
    axes[0].set_title(f"{target_name}: Predicted vs Actual")
    axes[0].set_xlabel(f"Actual {target_name}")
    axes[0].set_ylabel(f"Predicted {target_name}")

    axes[1].scatter(y_test, residuals, alpha=0.18, s=10)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"{target_name}: Residuals")
    axes[1].set_xlabel(f"Actual {target_name}")
    axes[1].set_ylabel("Residual")

    sns.histplot(residuals, bins=40, color="#ad7c59", ax=axes[2])
    axes[2].set_title(f"{target_name}: Residual Distribution")
    axes[2].set_xlabel("Residual")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{target_name.lower()}_prediction_diagnostics.png", dpi=200)
    plt.close()


def save_model_comparison(metrics_df: pd.DataFrame, target_name: str) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").reset_index(drop=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=test_df, x="rmse", y="model", color="#2f6c8f")
    plt.title(f"{target_name}: Test RMSE Comparison")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{target_name.lower()}_model_rmse_comparison.png", dpi=200)
    plt.close()


def save_summary_report(
    target_name: str,
    metrics_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    search_results: list[SearchResult],
) -> None:
    best_test = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").iloc[0]
    lines = [
        f"# {target_name} Prediction Summary",
        "",
        f"- Best test model: `{best_test['model']}`",
        f"- Test RMSE: `{best_test['rmse']:.2f}`",
        f"- Test MAE: `{best_test['mae']:.2f}`",
        f"- Test R^2: `{best_test['r2']:.4f}`",
        "",
        "## Top Correlated Numeric Features",
        "",
    ]

    for row in correlation_df.head(10).itertuples(index=False):
        lines.append(f"- `{row.feature}`: correlation `{row.correlation:.4f}`")

    lines.extend(["", "## Best-Model Permutation Importance", ""])
    for row in importance_df.head(10).itertuples(index=False):
        lines.append(f"- `{row.feature}`: importance `{row.importance:.2f}`")

    lines.extend(["", "## Model Tuning Choices", ""])
    for result in search_results:
        lines.append(f"- `{result.name}`: {json.dumps(result.best_params, sort_keys=True)}")

    (OUTPUT_DIR / f"{target_name.lower()}_summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_target_pipeline(
    df: pd.DataFrame,
    target_name: str,
    min_category_count: int,
    correlation_threshold: float,
    search_iter: int,
    torch_epochs: int,
    n_jobs: int,
) -> pd.DataFrame:
    print(f"\nPreparing target pipeline for {target_name}")
    X, y = get_feature_target_frames(df, target_name)
    X_train, X_val, X_test, y_train, y_val, y_test = split_frames(X, y)

    correlation_df, numeric_cols = compute_target_correlations(X_train, y_train, target_name=target_name)
    print(f"{target_name}: computed numeric correlations for {len(numeric_cols)} features")
    selected_numeric_cols, filter_decisions = select_numeric_features_by_correlation(
        X_train=X_train,
        y_train=y_train,
        numeric_cols=numeric_cols,
        threshold=correlation_threshold,
        target_name=target_name,
    )
    print(f"{target_name}: kept {len(selected_numeric_cols)} numeric features after correlation filtering")
    categorical_cols = [column for column in X_train.columns if column not in numeric_cols]

    preprocessor = build_preprocessor(
        numeric_cols=selected_numeric_cols,
        categorical_cols=categorical_cols,
        min_category_count=min_category_count,
    )
    cv = TimeSeriesSplit(n_splits=3)

    search_results: list[SearchResult] = []
    search_results.append(run_dummy_model(X_train, y_train, X_val, y_val, X_test, y_test))

    ridge_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Ridge(random_state=RANDOM_STATE)),
        ]
    )
    search_results.append(
        run_grid_search(
            name="ridge",
            pipeline=ridge_pipeline,
            param_grid={"model__alpha": [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]},
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            n_jobs=n_jobs,
        )
    )

    elastic_net_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", ElasticNet(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )
    search_results.append(
        run_grid_search(
            name="elastic_net",
            pipeline=elastic_net_pipeline,
            param_grid={
                "model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            n_jobs=n_jobs,
        )
    )

    hgb_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    search_results.append(
        run_sampled_search(
            name="hist_gradient_boosting",
            pipeline=hgb_pipeline,
            param_distributions={
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__max_depth": [4, 6, 8, None],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [20, 40, 80],
                "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
                "model__max_iter": [200, 300],
            },
            n_iter=search_iter,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            n_jobs=n_jobs,
        )
    )

    xgb_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                xgb.XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=max(n_jobs, 1),
                ),
            ),
        ]
    )
    search_results.append(
        run_sampled_search(
            name="xgboost",
            pipeline=xgb_pipeline,
            param_distributions={
                "model__n_estimators": [200, 350, 500],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__max_depth": [4, 6, 8],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_alpha": [0.0, 0.1, 0.5],
                "model__reg_lambda": [1.0, 5.0, 10.0],
            },
            n_iter=search_iter,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            cv=cv,
            n_jobs=n_jobs,
        )
    )

    torch_result = train_torch_model(
        preprocessor=clone(preprocessor),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        target_name=target_name,
        epochs=torch_epochs,
    )
    if torch_result is not None:
        search_results.append(torch_result)

    metrics_rows = []
    prediction_df = pd.DataFrame({"actual": y_test})
    params_rows = []
    for result in search_results:
        metrics_rows.append({"target": target_name, "model": result.name, "split": "validation", **result.val_metrics})
        metrics_rows.append({"target": target_name, "model": result.name, "split": "test", **result.test_metrics})
        prediction_df[f"pred_{result.name}"] = result.test_predictions
        params_rows.append({"target": target_name, "model": result.name, "best_params": json.dumps(result.best_params, sort_keys=True)})
        extract_linear_coefficients(result, target_name)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["split", "rmse"]).reset_index(drop=True)
    params_df = pd.DataFrame(params_rows)

    metrics_df.to_csv(OUTPUT_DIR / f"{target_name.lower()}_model_metrics.csv", index=False)
    params_df.to_csv(OUTPUT_DIR / f"{target_name.lower()}_best_params.csv", index=False)
    prediction_df.to_csv(OUTPUT_DIR / f"{target_name.lower()}_test_predictions.csv", index=False)

    best_test_model_name = metrics_df[metrics_df["split"] == "test"].sort_values("rmse").iloc[0]["model"]
    best_result = next(result for result in search_results if result.name == best_test_model_name)
    importance_df = save_permutation_importance(best_result, X_val, y_val, target_name)
    save_prediction_plots(target_name, y_test, prediction_df, best_model_name=best_test_model_name)
    save_model_comparison(metrics_df, target_name)
    save_summary_report(target_name, metrics_df, correlation_df, importance_df, search_results)

    print(f"\n=== {target_name} ===")
    print(f"Rows: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
    print(f"Selected numeric features: {len(selected_numeric_cols)} / {len(numeric_cols)}")
    print(f"Dropped by correlation filter: {len(filter_decisions)}")
    print(metrics_df[metrics_df["split"] == "test"][["model", "rmse", "mae", "r2"]].sort_values("rmse").to_string(index=False))

    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate White/Black Elo prediction pipeline with tuning and deep learning.")
    parser.add_argument("--max-rows", type=int, default=200_000, help="Optional cap on games used from the cleaned dataset.")
    parser.add_argument("--search-iter", type=int, default=8, help="Sample count for non-linear hyperparameter search.")
    parser.add_argument("--torch-epochs", type=int, default=35, help="Max epochs for each PyTorch candidate model.")
    parser.add_argument("--min-category-count", type=int, default=200, help="Minimum category count before one-hot grouping.")
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.97,
        help="Absolute pairwise correlation threshold used to drop redundant numeric features.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel worker count for sklearn searches.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["WhiteElo", "BlackElo"],
        help="Target columns to run. Example: --targets WhiteElo",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.max_rows)
    df = add_engineered_features(df)

    all_metrics = []
    for target_name in args.targets:
        metrics_df = run_target_pipeline(
            df=df,
            target_name=target_name,
            min_category_count=args.min_category_count,
            correlation_threshold=args.correlation_threshold,
            search_iter=args.search_iter,
            torch_epochs=args.torch_epochs,
            n_jobs=args.n_jobs,
        )
        all_metrics.append(metrics_df)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(OUTPUT_DIR / "combined_white_black_metrics.csv", index=False)
    print(f"\nWrote combined metrics to {OUTPUT_DIR / 'combined_white_black_metrics.csv'}")


if __name__ == "__main__":
    main()
