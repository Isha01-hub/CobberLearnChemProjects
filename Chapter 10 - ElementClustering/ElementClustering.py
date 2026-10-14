from pathlib import Path
import pandas as pd

# Radius definition: RSC non-bonded atomic radius.
# Converted from angstroms to picometres: 1 Å = 100 pm.
project_folder = Path(__file__).resolve().parent
csv_path = project_folder / "group1_elements.csv"

data = pd.read_csv(csv_path)

print("CHAPTER 10: ALKALI METALS")
print("Atomic radius: non-bonded, in pm")
print("First ionization energy: in kJ/mol\n")

print(data.drop(columns="Source_URL").to_string(index=False))
print("\nData sources:")
print(data[["Symbol", "Source_URL"]].to_string(index=False))
print("\nMissing values:")
print(data.isna().sum().to_string())

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

points = ax.scatter(
    data["Atomic_Radius_pm"],
    data["First_Ionization_Energy_kJ_mol"],
    c=data["Atomic_Number"],
    cmap="viridis",
    s=100,
    edgecolors="black",
)

# Label each point with its element symbol.
for _, row in data.iterrows():
    ax.annotate(
        row["Symbol"],
        (
            row["Atomic_Radius_pm"],
            row["First_Ionization_Energy_kJ_mol"],
        ),
        xytext=(7, 7),
        textcoords="offset points",
    )

ax.set_xlabel("Non-bonded atomic radius (pm)")
ax.set_ylabel("First ionization energy (kJ/mol)")
ax.set_title("Group 1: Atomic Radius and Ionization Energy")
ax.grid(alpha=0.25)
ax.margins(0.15)

fig.colorbar(points, ax=ax, label="Atomic number")
fig.tight_layout()

plot_path = project_folder / "group1_periodic_trend.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)

print(f"\nTrend plot saved to: {plot_path}")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

feature_columns = [
    "Atomic_Radius_pm",
    "First_Ionization_Energy_kJ_mol",
]

X = data[feature_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Convert centers back to the original units for plotting.
centers = scaler.inverse_transform(kmeans.cluster_centers_)

clustered_data = data.copy()
clustered_data["Cluster"] = labels

print("\nGROUP 1: TWO CLUSTERS")
print(
    clustered_data[
        ["Element", "Symbol"] + feature_columns + ["Cluster"]
    ].to_string(index=False)
)

fig, ax = plt.subplots(figsize=(8, 6))

for cluster in range(2):
    members = clustered_data[clustered_data["Cluster"] == cluster]
    ax.scatter(
        members["Atomic_Radius_pm"],
        members["First_Ionization_Energy_kJ_mol"],
        s=100,
        label=f"Cluster {cluster + 1}",
        edgecolors="black",
    )

for _, row in clustered_data.iterrows():
    ax.annotate(
        row["Symbol"],
        (
            row["Atomic_Radius_pm"],
            row["First_Ionization_Energy_kJ_mol"],
        ),
        xytext=(7, 7),
        textcoords="offset points",
    )

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    color="black",
    marker="X",
    s=250,
    label="Cluster centers",
)

ax.set_xlabel("Non-bonded atomic radius (pm)")
ax.set_ylabel("First ionization energy (kJ/mol)")
ax.set_title("Group 1: KMeans Clustering (k = 2)")
ax.legend()
ax.grid(alpha=0.25)
ax.margins(0.15)
fig.tight_layout()
fig.savefig(project_folder / "group1_clusters_k2.png", dpi=200)
plt.close(fig)

clustered_data.to_csv(
    project_folder / "group1_cluster_assignments_k2.csv",
    index=False,
)

print("\nSaved the cluster plot and assignment CSV.")

# Try different cluster counts until the user chooses to stop.
while True:
    choice = input(
        f"\nEnter a cluster count (1–{len(data)}), or q to quit: "
    ).strip().lower()

    if choice == "q":
        print("Finished exploring cluster counts.")
        break

    try:
        k = int(choice)
    except ValueError:
        print("Please enter a whole number or q.")
        continue

    if not 1 <= k <= len(data):
        print(f"Choose a number between 1 and {len(data)}.")
        continue

    trial_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )
    trial_labels = trial_model.fit_predict(X_scaled)
    trial_centers = scaler.inverse_transform(
        trial_model.cluster_centers_
    )

    trial_data = data.copy()
    trial_data["Cluster"] = trial_labels

    print(f"\nGROUP 1: k = {k}")
    print(
        trial_data[["Element", "Symbol", "Cluster"]]
        .to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    for cluster in range(k):
        members = trial_data[trial_data["Cluster"] == cluster]
        ax.scatter(
            members["Atomic_Radius_pm"],
            members["First_Ionization_Energy_kJ_mol"],
            s=100,
            edgecolors="black",
            label=f"Cluster {cluster + 1}",
        )

    for _, row in trial_data.iterrows():
        ax.annotate(
            row["Symbol"],
            (
                row["Atomic_Radius_pm"],
                row["First_Ionization_Energy_kJ_mol"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
        )

    ax.scatter(
        trial_centers[:, 0],
        trial_centers[:, 1],
        color="black",
        marker="X",
        s=250,
        label="Cluster centers",
    )

    ax.set_xlabel("Non-bonded atomic radius (pm)")
    ax.set_ylabel("First ionization energy (kJ/mol)")
    ax.set_title(f"Group 1: KMeans Clustering (k = {k})")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.margins(0.15)
    fig.tight_layout()

    fig.savefig(
        project_folder / f"group1_clusters_k{k}.png",
        dpi=200,
    )
    plt.close(fig)

    trial_data.to_csv(
        project_folder / f"group1_cluster_assignments_k{k}.csv",
        index=False,
    )
    print(f"Saved the plot and assignments for k = {k}.")

# Group 2 extension: this section must be outside the while loop.
group2_data = pd.read_csv(project_folder / "group2_elements.csv")

print("\nGROUP 2: ALKALINE EARTH METALS")
print(group2_data.drop(columns="Source_URL").to_string(index=False))

group2_scaler = StandardScaler()
group2_scaled = group2_scaler.fit_transform(
    group2_data[feature_columns]
)

group2_model = KMeans(n_clusters=2, random_state=42, n_init=10)
group2_data["Cluster"] = group2_model.fit_predict(group2_scaled)
group2_centers = group2_scaler.inverse_transform(
    group2_model.cluster_centers_
)

print("\nGROUP 2: TWO CLUSTERS")
print(group2_data[["Element", "Symbol", "Cluster"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))

for cluster in range(2):
    members = group2_data[group2_data["Cluster"] == cluster]
    ax.scatter(
        members["Atomic_Radius_pm"],
        members["First_Ionization_Energy_kJ_mol"],
        s=100,
        edgecolors="black",
        label=f"Cluster {cluster + 1}",
    )

for _, row in group2_data.iterrows():
    ax.annotate(
        row["Symbol"],
        (
            row["Atomic_Radius_pm"],
            row["First_Ionization_Energy_kJ_mol"],
        ),
        xytext=(7, 7),
        textcoords="offset points",
    )

ax.scatter(
    group2_centers[:, 0],
    group2_centers[:, 1],
    color="black",
    marker="X",
    s=250,
    label="Cluster centers",
)
ax.set_xlabel("Non-bonded atomic radius (pm)")
ax.set_ylabel("First ionization energy (kJ/mol)")
ax.set_title("Group 2: KMeans Clustering (k = 2)")
ax.legend()
ax.grid(alpha=0.25)
ax.margins(0.15)
fig.tight_layout()
fig.savefig(project_folder / "group2_clusters_k2.png", dpi=200)
plt.close(fig)

group2_data.to_csv(
    project_folder / "group2_cluster_assignments_k2.csv",
    index=False,
)
print("\nSaved the Group 2 plot and cluster assignments.")
