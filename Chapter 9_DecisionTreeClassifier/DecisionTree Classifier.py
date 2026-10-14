from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Data supplied in the Chapter 9 workbook.
data = pd.DataFrame({
    "Molecule": [f"Molecule {i}" for i in range(1, 13)],
    "Molecular Weight": [
        180, 250, 80, 300, 150, 400,
        90, 200, 130, 275, 135, 220
    ],
    "Hydrogen Bond Donors": [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    "Hydrogen Bond Acceptors": [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    "Water Solubility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
})

feature_names = [
    "Molecular Weight",
    "Hydrogen Bond Donors",
    "Hydrogen Bond Acceptors",
]

# Separate input features from the classification target.
X = data[feature_names]
y = data["Water Solubility"]

# Train the original decision tree.
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Improvement 1: limit a second tree to a maximum depth of 2.
limited_model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42,
)
limited_model.fit(X, y)

# Display clearly labeled predictions.
results = data.copy()
results["Original Prediction"] = model.predict(X)
results["Depth-2 Prediction"] = limited_model.predict(X)

print("CHAPTER 9: SOLUBILITY CLASSIFICATION")
print("0 = Not soluble; 1 = Soluble\n")
print(results.to_string(index=False))

print("\nTREE COMPARISON")
print(f"{'Model':<15} {'Depth':>6} {'Leaves':>8} {'Training accuracy':>20}")
for name, tree_model in [
    ("Original", model),
    ("Maximum depth 2", limited_model),
]:
    print(
        f"{name:<15} "
        f"{tree_model.get_depth():>6} "
        f"{tree_model.get_n_leaves():>8} "
        f"{tree_model.score(X, y):>19.1%}"
    )

print("\nThese accuracies use training data, not new molecules.")

# Save images beside this Python file.
output_folder = Path(__file__).resolve().parent


def save_tree(tree_model, filename, title):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Improvement 2: colored nodes and readable text.
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["Not soluble", "Soluble"],
        filled=True,
        rounded=True,
        fontsize=11,
        ax=ax,
    )

    ax.set_title(title, fontsize=16)
    fig.tight_layout()

    image_path = output_folder / filename
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {image_path}")


save_tree(
    model,
    "solubility_tree.png",
    "Solubility Decision Tree",
)

save_tree(
    limited_model,
    "solubility_tree_depth2.png",
    "Solubility Decision Tree — Maximum Depth 2",
)