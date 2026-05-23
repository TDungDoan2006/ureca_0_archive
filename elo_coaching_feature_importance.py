from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("/Users/phamhoanghai/Ureca")
INPUT_PATH = ROOT / "data_manipulation_output" / "output_data_cleaned.csv"
OUTPUT_DIR = ROOT / "elo_prediction_output"
PLOT_DIR = ROOT / "elo_prediction_plots"
RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_player_frame(df: pd.DataFrame) -> pd.DataFrame:
    shared_columns = [
        "GameID",
        "Event",
        "TimeControl",
        "TimeControlBucket",
        "Termination",
        "MoveCount",
        "GameLengthBucket",
        "OpeningCaptureCount12Ply",
        "OpeningCheckCount12Ply",
        "TotalCaptures",
        "TotalChecks",
        "TotalPromotions",
        "EnPassantCount",
    ]

    white = pd.DataFrame(
        {
            "color": "white",
            "player_elo": df["WhiteElo"],
            "castled": df["WhiteCastled"],
            "castle_side": df["WhiteCastleSide"],
            "early_queen_move": df["WhiteEarlyQueenMove"],
            "pawn_moves": df["WhitePawnMoves"],
            "knight_moves": df["WhiteKnightMoves"],
            "bishop_moves": df["WhiteBishopMoves"],
            "rook_moves": df["WhiteRookMoves"],
            "queen_moves": df["WhiteQueenMoves"],
            "king_moves": df["WhiteKingMoves"],
        }
    )
    for column in shared_columns:
        white[column] = df[column]

    black = pd.DataFrame(
        {
            "color": "black",
            "player_elo": df["BlackElo"],
            "castled": df["BlackCastled"],
            "castle_side": df["BlackCastleSide"],
            "early_queen_move": df["BlackEarlyQueenMove"],
            "pawn_moves": df["BlackPawnMoves"],
            "knight_moves": df["BlackKnightMoves"],
            "bishop_moves": df["BlackBishopMoves"],
            "rook_moves": df["BlackRookMoves"],
            "queen_moves": df["BlackQueenMoves"],
            "king_moves": df["BlackKingMoves"],
        }
    )
    for column in shared_columns:
        black[column] = df[column]

    player_df = pd.concat([white, black], ignore_index=True)
    move_denominator = player_df["MoveCount"].replace(0, np.nan)

    for piece in ["pawn", "knight", "bishop", "rook", "queen", "king"]:
        move_col = f"{piece}_moves"
        player_df[f"{move_col}_rate"] = player_df[move_col] / move_denominator

    player_df["minor_piece_moves"] = player_df["knight_moves"] + player_df["bishop_moves"]
    player_df["minor_piece_move_rate"] = player_df["minor_piece_moves"] / move_denominator
    player_df["heavy_piece_moves"] = player_df["rook_moves"] + player_df["queen_moves"]
    player_df["heavy_piece_move_rate"] = player_df["heavy_piece_moves"] / move_denominator
    player_df["opening_checks_per_ply12"] = player_df["OpeningCheckCount12Ply"] / 12.0
    player_df["opening_captures_per_ply12"] = player_df["OpeningCaptureCount12Ply"] / 12.0
    player_df["checks_per_move"] = player_df["TotalChecks"] / move_denominator
    player_df["captures_per_move"] = player_df["TotalCaptures"] / move_denominator
    player_df["promotions_per_move"] = player_df["TotalPromotions"] / move_denominator
    player_df["en_passant_per_move"] = player_df["EnPassantCount"] / move_denominator

    player_df = player_df.dropna(subset=["player_elo"]).reset_index(drop=True)
    return player_df


def build_feature_frame(player_df: pd.DataFrame, include_context: bool) -> pd.DataFrame:
    feature_columns = [
        "castled",
        "castle_side",
        "early_queen_move",
        "pawn_moves_rate",
        "knight_moves_rate",
        "bishop_moves_rate",
        "rook_moves_rate",
        "queen_moves_rate",
        "king_moves_rate",
        "minor_piece_move_rate",
        "heavy_piece_move_rate",
        "opening_checks_per_ply12",
        "opening_captures_per_ply12",
        "checks_per_move",
        "captures_per_move",
        "promotions_per_move",
        "en_passant_per_move",
        "MoveCount",
        "GameLengthBucket",
        "color",
    ]
    if include_context:
        feature_columns.extend(["Event", "TimeControl", "TimeControlBucket", "Termination"])
    return player_df[feature_columns].copy()


def summarise_feature_name(raw_name: str) -> str:
    name = raw_name
    if "__" in name:
        name = name.split("__", 1)[1]
    if name.startswith("castle_side_"):
        return f"castle_side={name.replace('castle_side_', '')}"
    if name.startswith("color_"):
        return f"color={name.replace('color_', '')}"
    if name.startswith("GameLengthBucket_"):
        return f"game_length={name.replace('GameLengthBucket_', '')}"
    return name


def recommendation_text(feature_name: str, effect: str) -> str:
    guidance = {
        "early_queen_move": {
            "higher": "Moving the queen early is more common in lower-rated play. Develop knights and bishops first, then bring the queen out when it wins something concrete.",
            "lower": "Avoid early queen attacks unless they clearly win material or force a tactical sequence.",
        },
        "castled": {
            "higher": "Castling is associated with stronger play. Prioritize king safety and connect your rooks earlier.",
            "lower": "If you often delay castling, make king safety one of your first opening checks each game.",
        },
        "castle_side=kingside": {
            "higher": "Kingside castling shows up as a stable, practical habit. It is usually the safest default for improving players.",
            "lower": "If kingside castling is rare in your games, review whether you are leaving your king in the center too long.",
        },
        "castle_side=none": {
            "higher": "Not castling tends to correlate with weaker results. Look for faster king safety in the opening.",
            "lower": "Reducing no-castle games is usually a clean improvement target for beginners.",
        },
        "queen_moves_rate": {
            "higher": "A high share of queen moves often means lost tempi or repeated queen chases. Try to solve positions with minor pieces first.",
            "lower": "Keeping queen moves efficient is a good sign. Avoid using the queen as your first attacking piece.",
        },
        "rook_moves_rate": {
            "higher": "More rook activity can reflect better development and open-file usage. Bring rooks into play after king safety and minor-piece development.",
            "lower": "If your rooks stay passive, focus on castling and placing them on open or half-open files.",
        },
        "knight_moves_rate": {
            "higher": "Extra knight moves can mean rerouting, but too many early knight moves may also lose time. Aim for purposeful development rather than repetition.",
            "lower": "Make sure both knights develop naturally in the opening before launching side attacks.",
        },
        "bishop_moves_rate": {
            "higher": "Active bishops often support stronger positions. Look for simple developing squares before starting direct attacks.",
            "lower": "If bishops stay blocked, fix your pawn structure and complete development sooner.",
        },
        "minor_piece_move_rate": {
            "higher": "Using knights and bishops actively is generally healthier than overusing the queen early.",
            "lower": "For improvement, finish minor-piece development before spending multiple moves on the same major piece.",
        },
        "heavy_piece_move_rate": {
            "higher": "Frequent queen and rook moves can be good later, but too much heavy-piece activity early may signal wasted tempi.",
            "lower": "This usually means you are not over-relying on queen and rook moves too early.",
        },
        "checks_per_move": {
            "higher": "More checks do not always mean better chess. Prefer checks that improve your position or win material, not checks for their own sake.",
            "lower": "A lower check rate is fine if your moves improve development, king safety, and material balance.",
        },
        "captures_per_move": {
            "higher": "Frequent captures can reflect tactical positions, but beginners should still check whether each capture improves the position.",
            "lower": "Do not force captures. Improve your worst piece first and capture only when it helps.",
        },
        "opening_checks_per_ply12": {
            "higher": "Early checks are often tempting but not always strong. Before giving check, ask whether you are improving development or just helping the opponent.",
            "lower": "Avoiding random early checks is usually healthy if you are developing smoothly.",
        },
        "opening_captures_per_ply12": {
            "higher": "Early captures can be good when they win the center or a pawn, but avoid grabbing material if it delays development badly.",
            "lower": "A lower early-capture rate is fine when you are prioritizing piece activity and king safety.",
        },
        "king_moves_rate": {
            "higher": "Frequent king moves usually mean king safety problems. Castle earlier and avoid creating unnecessary king walks.",
            "lower": "Fewer king moves usually means better king safety.",
        },
    }
    feature_guidance = guidance.get(feature_name)
    if feature_guidance:
        return feature_guidance[effect]
    if effect == "higher":
        return "Higher values of this feature are associated with stronger players in this model."
    return "Lower values of this feature are associated with stronger players in this model."


def rank_actionable_features(
    feature_names: np.ndarray,
    coefficients: np.ndarray,
    importances: np.ndarray,
) -> pd.DataFrame:
    feature_df = pd.DataFrame(
        {
            "raw_feature": feature_names,
            "feature": [summarise_feature_name(name) for name in feature_names],
            "coefficient": coefficients,
            "importance": importances,
        }
    )
    feature_df["abs_coefficient"] = feature_df["coefficient"].abs()
    feature_df["effect_direction"] = np.where(feature_df["coefficient"] >= 0, "higher", "lower")
    feature_df["recommendation"] = [
        recommendation_text(feature, effect)
        for feature, effect in zip(feature_df["feature"], feature_df["effect_direction"])
    ]
    return feature_df.sort_values(["importance", "abs_coefficient"], ascending=[False, False]).reset_index(drop=True)


def compute_group_examples(player_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    examples = {
        "early_queen_move": ("early_queen_move", 0, 1),
        "castled": ("castled", 0, 1),
    }
    for label, (column, baseline, comparison) in examples.items():
        grouped = (
            player_df.groupby(column)["player_elo"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_elo", "count": "games"})
        )
        baseline_row = grouped[grouped[column] == baseline]
        comparison_row = grouped[grouped[column] == comparison]
        if baseline_row.empty or comparison_row.empty:
            continue
        rows.append(
            {
                "feature": label,
                "baseline_value": baseline,
                "comparison_value": comparison,
                "baseline_avg_elo": float(baseline_row["avg_elo"].iloc[0]),
                "comparison_avg_elo": float(comparison_row["avg_elo"].iloc[0]),
                "elo_gap": float(comparison_row["avg_elo"].iloc[0] - baseline_row["avg_elo"].iloc[0]),
                "baseline_games": int(baseline_row["games"].iloc[0]),
                "comparison_games": int(comparison_row["games"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(
    metrics: dict[str, float],
    actionable_features: pd.DataFrame,
    group_examples: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Elo Coaching Feature Importance",
        "",
        "This report trains a coaching-focused Elo model using player-controlled behaviors rather than time-control metadata.",
        "",
        "## Validation",
        "",
        f"- Validation RMSE: {metrics['val_rmse']:.2f}",
        f"- Validation MAE: {metrics['val_mae']:.2f}",
        f"- Validation R^2: {metrics['val_r2']:.3f}",
        f"- Test RMSE: {metrics['test_rmse']:.2f}",
        f"- Test MAE: {metrics['test_mae']:.2f}",
        f"- Test R^2: {metrics['test_r2']:.3f}",
        "",
        "## Top Actionable Signals",
        "",
    ]

    for row in actionable_features.head(10).itertuples(index=False):
        direction = "Higher is better" if row.effect_direction == "higher" else "Lower is better"
        lines.append(
            f"- `{row.feature}`: importance `{row.importance:.2f}`, coefficient `{row.coefficient:.2f}`. {direction}. {row.recommendation}"
        )

    if not group_examples.empty:
        lines.extend(["", "## Simple Beginner Examples", ""])
        for row in group_examples.itertuples(index=False):
            lines.append(
                f"- `{row.feature}`: average Elo changes from `{row.baseline_avg_elo:.1f}` to `{row.comparison_avg_elo:.1f}` "
                f"(gap `{row.elo_gap:.1f}`) between values `{row.baseline_value}` and `{row.comparison_value}`."
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a coaching-oriented Elo feature importance model.")
    parser.add_argument("--max-rows", type=int, default=250_000, help="Optional sample cap before player expansion.")
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include event/time-control context columns. Default behavior excludes them to keep the output coachable.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = df.sort_values("GameID").reset_index(drop=True)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=RANDOM_STATE).sort_values("GameID").reset_index(drop=True)

    player_df = build_player_frame(df)
    X = build_feature_frame(player_df, include_context=args.include_context)
    y = player_df["player_elo"].astype(float).to_numpy()

    split_1 = int(len(player_df) * TRAIN_FRAC)
    split_2 = int(len(player_df) * (TRAIN_FRAC + VAL_FRAC))

    X_train = X.iloc[:split_1].copy()
    X_val = X.iloc[split_1:split_2].copy()
    X_test = X.iloc[split_2:].copy()
    y_train = y[:split_1]
    y_val = y[split_1:split_2]
    y_test = y[split_2:]

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [column for column in X_train.columns if column not in numeric_cols]

    preprocessor = ColumnTransformer(
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
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_t, y_train)

    val_pred = model.predict(X_val_t)
    test_pred = model.predict(X_test_t)

    metrics = {
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
        "val_mae": float(mean_absolute_error(y_val, val_pred)),
        "val_r2": float(r2_score(y_val, val_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
    }

    perm_rows = min(len(y_val), 10_000)
    perm = permutation_importance(
        model,
        X_val_t[:perm_rows],
        y_val[:perm_rows],
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="neg_mean_squared_error",
    )

    feature_rank_df = rank_actionable_features(
        feature_names=feature_names,
        coefficients=model.coef_,
        importances=perm.importances_mean,
    )
    metrics_df = pd.DataFrame([metrics])
    group_examples_df = compute_group_examples(player_df)

    suffix = "with_context" if args.include_context else "actionable_only"
    metrics_path = OUTPUT_DIR / f"elo_coaching_metrics_{suffix}.csv"
    ranking_path = OUTPUT_DIR / f"elo_coaching_feature_importance_{suffix}.csv"
    examples_path = OUTPUT_DIR / f"elo_coaching_feature_examples_{suffix}.csv"
    summary_path = OUTPUT_DIR / f"elo_coaching_report_{suffix}.md"

    metrics_df.to_csv(metrics_path, index=False)
    feature_rank_df.to_csv(ranking_path, index=False)
    group_examples_df.to_csv(examples_path, index=False)
    write_markdown_summary(metrics, feature_rank_df, group_examples_df, summary_path)

    print(f"Rows used: {len(df):,} games / {len(player_df):,} player-rows")
    print(f"Include context: {args.include_context}")
    print(f"Validation RMSE: {metrics['val_rmse']:.2f}")
    print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {ranking_path}")
    print(f"Wrote: {examples_path}")
    print(f"Wrote: {summary_path}")
    print("\nTop actionable features:")
    print(
        feature_rank_df[
            ["feature", "importance", "coefficient", "effect_direction"]
        ].head(12).to_string(index=False)
    )


if __name__ == "__main__":
    main()
