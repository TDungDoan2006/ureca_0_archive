# Question 3: Player Playstyle Archetypes

This project clusters chess players into playstyle archetypes using the defining characteristics of their favorite openings.

## Method
- Dataset: `dataset/final_data`
- Player filter: at least 20 games
- Favorite openings per player: top 3
- Features: opening-family mix, ECO-group mix, opening concentration, opening diversity, and keyword-based style signals
- Clustering algorithm: KMeans with the best `k` selected from [3, 4, 5, 6] using silhouette score
- Best cluster count: 5 with silhouette score 0.2403

## Cluster Summary
- Aggressive Attackers: 4486 players, mean Elo 1516.1, mean favorite-opening share 0.11, sample openings: French Defense: Knight Variation, Philidor Defense #3, Philidor Defense #2, Bishop's Opening, King's Pawn Game: Leonardis Variation
- Counterpunching Defenders: 3784 players, mean Elo 1642.4, mean favorite-opening share 0.11, sample openings: Scandinavian Defense: Mieses-Kotroc Variation, Caro-Kann Defense, Sicilian Defense, Sicilian Defense: Bowdler Attack, Owen Defense
- Hypermodern Strategists: 2375 players, mean Elo 1623.6, mean favorite-opening share 0.19, sample openings: Van't Kruijs Opening, Hungarian Opening, Modern Defense, Owen Defense, Mieses Opening
- Positional Defenders: 2019 players, mean Elo 1578.2, mean favorite-opening share 0.13, sample openings: Queen's Pawn Game #2, Horwitz Defense, Queen's Pawn Game, Queen's Pawn Game: Mason Attack, Queen's Pawn Game: Chigorin Variation
- Indian Defense Specialists: 224 players, mean Elo 1740.4, mean favorite-opening share 0.09, sample openings: Caro-Kann Defense, Queen's Gambit Declined, Nimzo-Indian Defense #2, Sicilian Defense, Amar Opening

## Outputs
- `output/player_archetypes.csv`: one row per player with cluster and archetype
- `output/cluster_summary.csv`: high-level statistics per cluster
- `output/cluster_feature_centers.csv`: cluster centers in the original feature space
- `output/top_openings_by_cluster.csv`: most common favorite openings inside each cluster
- `plots/silhouette_scores.png`: model selection chart
- `plots/player_archetypes_pca.png`: 2D PCA view of clustered players
- `plots/cluster_feature_heatmap.png`: heatmap of cluster-defining features
