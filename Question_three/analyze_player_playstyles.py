from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("/Users/phamhoanghai/Ureca/dataset/final_data")
OUTPUT_DIR = Path("/Users/phamhoanghai/Ureca/Question_three/output")
PLOT_DIR = Path("/Users/phamhoanghai/Ureca/Question_three/plots")

MIN_GAMES_PER_PLAYER = 20
TOP_N_FAVORITE_OPENINGS = 3
RANDOM_STATE = 42
K_RANGE = range(3, 7)

FEATURE_COLUMNS = [
    "avg_elo",
    "opening_diversity",
    "favorite_opening_share",
    "favorite_openings_coverage",
    "family_closed_share",
    "family_flank_share",
    "family_indian_share",
    "family_open_share",
    "family_semi_open_share",
    "eco_a_share",
    "eco_b_share",
    "eco_c_share",
    "eco_d_share",
    "eco_e_share",
    "attackish_share",
    "defensive_share",
    "positional_share",
    "hypermodern_share",
]


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    usecols = [
        "GameID",
        "White",
        "Black",
        "WhiteEloNumeric",
        "BlackEloNumeric",
        "ECO",
        "Opening",
        "OpeningFamily",
        "NumericResult",
    ]
    df = pd.read_csv(DATA_PATH, usecols=usecols)
    return df.dropna(
        subset=["White", "Black", "WhiteEloNumeric", "BlackEloNumeric", "Opening", "OpeningFamily", "ECO"]
    ).copy()


def build_player_game_rows(df: pd.DataFrame) -> pd.DataFrame:
    white_rows = df.rename(
        columns={
            "White": "player",
            "WhiteEloNumeric": "player_elo",
        }
    )[
        ["GameID", "player", "player_elo", "Opening", "OpeningFamily", "ECO", "NumericResult"]
    ].copy()
    white_rows["perspective_score"] = white_rows["NumericResult"]

    black_rows = df.rename(
        columns={
            "Black": "player",
            "BlackEloNumeric": "player_elo",
        }
    )[
        ["GameID", "player", "player_elo", "Opening", "OpeningFamily", "ECO", "NumericResult"]
    ].copy()
    black_rows["perspective_score"] = 1 - black_rows["NumericResult"]

    combined = pd.concat([white_rows, black_rows], ignore_index=True)
    combined["ECOGroup"] = combined["ECO"].astype(str).str[0].str.upper()
    return combined


def weighted_keyword_share(opening_counts: pd.Series, keywords: tuple[str, ...]) -> float:
    total = opening_counts.sum()
    if total == 0:
        return 0.0
    mask = opening_counts.index.to_series().str.lower().apply(
        lambda name: any(keyword in name for keyword in keywords)
    )
    return float(opening_counts[mask.values].sum() / total)


def build_player_profiles(player_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = player_games.groupby("player").filter(lambda frame: len(frame) >= MIN_GAMES_PER_PLAYER).copy()

    records: list[dict] = []
    favorite_rows: list[dict] = []

    for player, frame in eligible.groupby("player", sort=False):
        opening_counts = frame["Opening"].value_counts()
        favorite_counts = opening_counts.head(TOP_N_FAVORITE_OPENINGS)
        favorite_openings = favorite_counts.index.tolist()
        favorite_frame = frame[frame["Opening"].isin(favorite_openings)].copy()

        total_games = len(frame)
        favorite_total_games = int(favorite_counts.sum())
        family_share = (
            favorite_frame["OpeningFamily"]
            .value_counts(normalize=True)
            .reindex(
                [
                    "Closed and Semi-Closed",
                    "Flank Openings",
                    "Indian Defenses",
                    "Open Games and Semi-Open",
                    "Semi-Open Defenses",
                ],
                fill_value=0.0,
            )
        )
        eco_share = (
            favorite_frame["ECOGroup"]
            .value_counts(normalize=True)
            .reindex(list("ABCDE"), fill_value=0.0)
        )

        top_opening = favorite_counts.index[0]
        top_opening_games = int(favorite_counts.iloc[0])

        for opening_name, count in favorite_counts.items():
            favorite_rows.append(
                {
                    "player": player,
                    "opening": opening_name,
                    "games": int(count),
                    "share_within_player": float(count / total_games),
                }
            )

        records.append(
            {
                "player": player,
                "games_played": total_games,
                "avg_elo": float(frame["player_elo"].mean()),
                "mean_score": float(frame["perspective_score"].mean()),
                "opening_diversity": int(frame["Opening"].nunique()),
                "favorite_opening": top_opening,
                "favorite_opening_games": top_opening_games,
                "favorite_opening_share": float(top_opening_games / total_games),
                "favorite_openings_coverage": float(favorite_total_games / total_games),
                "family_closed_share": float(family_share["Closed and Semi-Closed"]),
                "family_flank_share": float(family_share["Flank Openings"]),
                "family_indian_share": float(family_share["Indian Defenses"]),
                "family_open_share": float(family_share["Open Games and Semi-Open"]),
                "family_semi_open_share": float(family_share["Semi-Open Defenses"]),
                "eco_a_share": float(eco_share["A"]),
                "eco_b_share": float(eco_share["B"]),
                "eco_c_share": float(eco_share["C"]),
                "eco_d_share": float(eco_share["D"]),
                "eco_e_share": float(eco_share["E"]),
                "attackish_share": weighted_keyword_share(favorite_counts, ("attack", "gambit", "sacrifice", "storm")),
                "defensive_share": weighted_keyword_share(favorite_counts, ("defense", "defence", "declined")),
                "positional_share": weighted_keyword_share(
                    favorite_counts,
                    ("system", "london", "english", "reti", "catalan", "queen's gambit", "slav", "caro-kann"),
                ),
                "hypermodern_share": weighted_keyword_share(
                    favorite_counts,
                    ("indian", "modern", "pirc", "reti", "english", "bird", "owen", "larsen", "grunfeld", "benoni"),
                ),
            }
        )

    profiles = pd.DataFrame(records).sort_values(["games_played", "avg_elo"], ascending=[False, False]).reset_index(
        drop=True
    )
    favorite_detail = pd.DataFrame(favorite_rows)
    return profiles, favorite_detail


def select_best_k(X_scaled: np.ndarray) -> tuple[int, pd.DataFrame]:
    results = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        results.append({"k": k, "silhouette_score": float(score)})

    score_df = pd.DataFrame(results).sort_values("silhouette_score", ascending=False).reset_index(drop=True)
    return int(score_df.iloc[0]["k"]), score_df


def infer_cluster_label(center: pd.Series) -> str:
    family_scores = {
        "closed": center["family_closed_share"],
        "flank": center["family_flank_share"],
        "indian": center["family_indian_share"],
        "open": center["family_open_share"],
        "semi_open": center["family_semi_open_share"],
    }
    dominant_family = max(family_scores, key=family_scores.get)

    if dominant_family == "flank":
        return "Hypermodern Strategists"
    if dominant_family == "indian":
        return "Indian Defense Specialists"
    if dominant_family == "closed":
        return "Positional Defenders"
    if dominant_family == "semi_open":
        return "Counterpunching Defenders"
    if dominant_family == "open":
        if center["attackish_share"] >= 0.18:
            return "Aggressive Attackers"
        return "Open-Game Technicians"
    return "Flexible Universalists"


def cluster_players(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profiles[FEATURE_COLUMNS])

    best_k, silhouette_df = select_best_k(X_scaled)
    model = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE)
    profiles = profiles.copy()
    profiles["cluster_id"] = model.fit_predict(X_scaled)

    centers_scaled = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLUMNS)
    centers_original = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURE_COLUMNS)
    centers_original["cluster_id"] = range(best_k)
    centers_original["archetype"] = centers_original.apply(infer_cluster_label, axis=1)

    label_map = centers_original.set_index("cluster_id")["archetype"].to_dict()
    profiles["archetype"] = profiles["cluster_id"].map(label_map)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    profiles["pca_1"] = coords[:, 0]
    profiles["pca_2"] = coords[:, 1]

    cluster_summary = (
        profiles.groupby(["cluster_id", "archetype"], as_index=False)
        .agg(
            players=("player", "count"),
            mean_games=("games_played", "mean"),
            mean_elo=("avg_elo", "mean"),
            mean_opening_diversity=("opening_diversity", "mean"),
            mean_favorite_share=("favorite_opening_share", "mean"),
            mean_top3_coverage=("favorite_openings_coverage", "mean"),
            mean_score=("mean_score", "mean"),
        )
        .sort_values(["players", "mean_elo"], ascending=[False, False])
        .reset_index(drop=True)
    )

    return profiles, centers_original, cluster_summary, silhouette_df


def build_top_openings_by_cluster(player_profiles: pd.DataFrame, favorite_detail: pd.DataFrame) -> pd.DataFrame:
    merged = favorite_detail.merge(
        player_profiles[["player", "cluster_id", "archetype"]],
        on="player",
        how="left",
    )
    summary = (
        merged.groupby(["cluster_id", "archetype", "opening"], as_index=False)["games"]
        .sum()
        .sort_values(["cluster_id", "games"], ascending=[True, False])
    )
    top_by_cluster = summary.groupby(["cluster_id", "archetype"], group_keys=False).head(10).reset_index(drop=True)
    return top_by_cluster


def save_outputs(
    profiles: pd.DataFrame,
    centers_original: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    silhouette_df: pd.DataFrame,
    top_openings_by_cluster: pd.DataFrame,
) -> None:
    profiles.sort_values(["cluster_id", "games_played"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "player_archetypes.csv",
        index=False,
    )
    centers_original.sort_values("cluster_id").to_csv(OUTPUT_DIR / "cluster_feature_centers.csv", index=False)
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    silhouette_df.to_csv(OUTPUT_DIR / "silhouette_scores.csv", index=False)
    top_openings_by_cluster.to_csv(OUTPUT_DIR / "top_openings_by_cluster.csv", index=False)


def make_plots(profiles: pd.DataFrame, centers_original: pd.DataFrame, silhouette_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=silhouette_df, x="k", y="silhouette_score", color="#2f6c8f")
    plt.title("Silhouette Score By Cluster Count")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "silhouette_scores.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=profiles,
        x="pca_1",
        y="pca_2",
        hue="archetype",
        size="games_played",
        sizes=(20, 160),
        alpha=0.75,
        palette="Set2",
    )
    plt.title("Player Playstyle Archetypes From Favorite Openings")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "player_archetypes_pca.png", dpi=200)
    plt.close()

    heatmap_cols = [
        "family_closed_share",
        "family_flank_share",
        "family_indian_share",
        "family_open_share",
        "family_semi_open_share",
        "favorite_opening_share",
        "favorite_openings_coverage",
        "attackish_share",
        "defensive_share",
        "positional_share",
        "hypermodern_share",
    ]
    heatmap_df = centers_original.sort_values("cluster_id").set_index("archetype")[heatmap_cols]

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Cluster Centers: Favorite Opening Characteristics")
    plt.xlabel("Feature")
    plt.ylabel("Archetype")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_feature_heatmap.png", dpi=200)
    plt.close()


def create_readme_summary(
    cluster_summary: pd.DataFrame,
    silhouette_df: pd.DataFrame,
    top_openings_by_cluster: pd.DataFrame,
) -> None:
    best_k_row = silhouette_df.sort_values("silhouette_score", ascending=False).iloc[0]

    lines = [
        "# Question 3: Player Playstyle Archetypes",
        "",
        "This project clusters chess players into playstyle archetypes using the defining characteristics of their favorite openings.",
        "",
        "## Method",
        f"- Dataset: `dataset/final_data`",
        f"- Player filter: at least {MIN_GAMES_PER_PLAYER} games",
        f"- Favorite openings per player: top {TOP_N_FAVORITE_OPENINGS}",
        "- Features: opening-family mix, ECO-group mix, opening concentration, opening diversity, and keyword-based style signals",
        f"- Clustering algorithm: KMeans with the best `k` selected from {list(K_RANGE)} using silhouette score",
        f"- Best cluster count: {int(best_k_row['k'])} with silhouette score {best_k_row['silhouette_score']:.4f}",
        "",
        "## Cluster Summary",
    ]

    for row in cluster_summary.itertuples(index=False):
        top_openings = top_openings_by_cluster[top_openings_by_cluster["cluster_id"] == row.cluster_id]["opening"].head(5)
        lines.append(
            f"- {row.archetype}: {int(row.players)} players, mean Elo {row.mean_elo:.1f}, "
            f"mean favorite-opening share {row.mean_favorite_share:.2f}, sample openings: {', '.join(top_openings)}"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "- `output/player_archetypes.csv`: one row per player with cluster and archetype",
            "- `output/cluster_summary.csv`: high-level statistics per cluster",
            "- `output/cluster_feature_centers.csv`: cluster centers in the original feature space",
            "- `output/top_openings_by_cluster.csv`: most common favorite openings inside each cluster",
            "- `plots/silhouette_scores.png`: model selection chart",
            "- `plots/player_archetypes_pca.png`: 2D PCA view of clustered players",
            "- `plots/cluster_feature_heatmap.png`: heatmap of cluster-defining features",
        ]
    )

    (Path("/Users/phamhoanghai/Ureca/Question_three/README.md")).write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_directories()
    raw_df = load_dataset()
    player_games = build_player_game_rows(raw_df)
    player_profiles, favorite_detail = build_player_profiles(player_games)
    profiles, centers_original, cluster_summary, silhouette_df = cluster_players(player_profiles)
    top_openings_by_cluster = build_top_openings_by_cluster(profiles, favorite_detail)
    save_outputs(profiles, centers_original, cluster_summary, silhouette_df, top_openings_by_cluster)
    make_plots(profiles, centers_original, silhouette_df)
    create_readme_summary(cluster_summary, silhouette_df, top_openings_by_cluster)

    print(f"Players clustered: {len(profiles):,}")
    print(f"Archetypes: {profiles['archetype'].nunique()}")
    print("Saved outputs to:")
    print(f"  - {OUTPUT_DIR}")
    print(f"  - {PLOT_DIR}")


if __name__ == "__main__":
    main()
