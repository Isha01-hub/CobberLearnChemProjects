"""Explore Iris data and compare raw and normalized K-means clustering."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def save_scatter(features, groups, title, filename, species_names=None):
    """Plot petal measurements using cluster or species assignments."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for group in sorted(set(groups)):
        selected = groups == group
        label = (
            species_names[int(group)]
            if species_names is not None
            else f"Cluster {group}"
        )

        ax.scatter(
            features.loc[selected, "petal length (cm)"],
            features.loc[selected, "petal width (cm)"],
            label=label,
            alpha=0.75,
            s=45
        )

    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def analyze_clusters(features, species, species_ids, training_data, k, method):
    """Fit without species labels, then compare clusters with species."""
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Only numerical measurements enter the clustering algorithm.
    clusters = model.fit_predict(training_data)

    # Species labels enter the analysis only after model fitting.
    comparison = pd.DataFrame({
        "Cluster": clusters,
        "Species": species
    })

    counts = pd.crosstab(
        comparison["Cluster"],
        comparison["Species"]
    )

    # Each row shows the species composition of one cluster.
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    prefix = f"k{k}_{method}"

    counts.to_csv(OUTPUT_DIR / f"{prefix}_species_counts.csv")
    percentages.round(2).to_csv(
        OUTPUT_DIR / f"{prefix}_species_percentages.csv"
    )

    assignments = features.copy()
    assignments["Cluster"] = clusters
    assignments["Species"] = species
    assignments.to_csv(
        OUTPUT_DIR / f"{prefix}_assignments.csv",
        index=False
    )

    # ARI compares groupings without assuming cluster IDs equal species IDs.
    ari = adjusted_rand_score(species_ids, clusters)

    print(f"\n{'=' * 60}")
    print(f"K = {k} | {method.upper()} FEATURES")
    print("=" * 60)
    print("\nSpecies counts within each cluster:")
    print(counts.to_string())
    print("\nSpecies percentages within each cluster:")
    print(percentages.round(2).to_string())
    print(f"\nAdjusted Rand Index: {ari:.4f}")

    save_scatter(
        features,
        clusters,
        f"Iris K-means: k={k}, {method} features",
        f"{prefix}_clusters.png"
    )

    return {
        "Clusters": k,
        "Features": method,
        "Adjusted Rand Index": ari
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the built-in dataset; no download is required.
    iris = load_iris(as_frame=True)

    # This DataFrame contains only the four measurement columns.
    features = iris.data.copy()

    # Keep species labels separate from the clustering inputs.
    species_ids = iris.target
    species = species_ids.map(
        dict(enumerate(iris.target_names))
    )

    print("IRIS DATA EXPLORATION")
    print("=" * 60)
    print(f"Number of flowers: {len(features)}")
    print(f"Number of features: {features.shape[1]}")

    print("\nFeature names:")
    for name in features.columns:
        print(f"- {name}")

    print("\nFirst five rows:")
    print(features.head().to_string())

    print("\nSummary statistics:")
    print(features.describe().round(3).to_string())

    print("\nMissing values:")
    print(features.isna().sum().to_string())

    print("\nActual species counts, used only for comparison:")
    print(species.value_counts().to_string())

    features.to_csv(OUTPUT_DIR / "iris_features.csv", index=False)
    features.describe().to_csv(OUTPUT_DIR / "feature_summary.csv")

    # Enhancement: rescale each feature to the range 0 to 1.
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Required baseline: three clusters, with and without normalization.
    results = []

    for method, training_data in [
        ("raw", features),
        ("normalized", normalized_features)
    ]:
        results.append(
            analyze_clusters(
                features,
                species,
                species_ids,
                training_data,
                k=3,
                method=method
            )
        )

    # Use original measurements for the species comparison plot.
    save_scatter(
        features,
        species_ids,
        "Iris flowers colored by actual species",
        "actual_species.png",
        species_names=iris.target_names
    )

    # Optional enhancement: choose another k without editing the code.
    while True:
        answer = input(
            "\nEnter another cluster count from 2 to 10 "
            "(for example, 7), or press Enter to finish: "
        ).strip()

        if answer == "":
            break

        try:
            k = int(answer)
            if not 2 <= k <= 10:
                raise ValueError
        except ValueError:
            print("Please enter a whole number from 2 to 10.")
            continue

        for method, training_data in [
            ("raw", features),
            ("normalized", normalized_features)
        ]:
            results.append(
                analyze_clusters(
                    features,
                    species,
                    species_ids,
                    training_data,
                    k=k,
                    method=method
                )
            )

    results_table = (
        pd.DataFrame(results)
        .drop_duplicates(subset=["Clusters", "Features"], keep="last")
        .sort_values(["Clusters", "Features"])
    )

    results_table.to_csv(
        OUTPUT_DIR / "clustering_comparison.csv",
        index=False
    )

    print("\nFINAL COMPARISON")
    print(results_table.round(4).to_string(index=False))
    print("\nARI = 1 means perfect agreement with the species grouping.")
    print("ARI near 0 means agreement close to chance.")
    print("ARI is not a percentage accuracy.")
    print(f"\nAll plots and tables saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()