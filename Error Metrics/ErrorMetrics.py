import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
# Values from the Chapter 8 workbook.
actual = np.array([2, 4, 5, 4, 5, 7, 9], dtype=float)
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8], dtype=float)
residuals = predicted - actual

# Calculate the metrics ourselves using NumPy.
mae = np.mean(np.abs(residuals))
mse = np.mean(residuals ** 2)
r_squared = 1 - (
    np.sum(residuals ** 2)
    / np.sum((actual - actual.mean()) ** 2)
)

# Improvement 1: display the values in readable columns.
print(f"{'Actual':>10} {'Predicted':>10} {'Residual':>10}")
for a, p, r in zip(actual, predicted, residuals):
    print(f"{a:10.2f} {p:10.2f} {r:10.2f}")

# Compare our calculations with scikit-learn.
manual = np.array([mae, mse, r_squared])
library = np.array([
    mean_absolute_error(actual, predicted),
    mean_squared_error(actual, predicted),
    r2_score(actual, predicted),
])
print(f"\n{'Metric':<8} {'NumPy':>12} {'scikit-learn':>14}")
for name, ours, theirs in zip(["MAE", "MSE", "R²"], manual, library):
    print(f"{name:<8} {ours:12.6f} {theirs:14.6f}")

print("\nDo the calculations match?", np.allclose(manual, library))

from pathlib import Path
import matplotlib.pyplot as plt

output_folder = Path(__file__).resolve().parent

# Improvement 2: highlight all predictions tied for largest error.
worst = np.isclose(np.abs(residuals), np.max(np.abs(residuals)))

# Plot 1: predicted versus actual values.
fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    actual[~worst], predicted[~worst],
    color="steelblue", s=70, label="Other predictions"
)
ax.scatter(
    actual[worst], predicted[worst],
    color="red", s=90, label="Largest absolute errors"
)

low = min(actual.min(), predicted.min()) - 0.5
high = max(actual.max(), predicted.max()) + 0.5
ax.plot([low, high], [low, high], "k--", label="Ideal: predicted = actual")
ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_xlabel("Actual value")
ax.set_ylabel("Predicted value")
ax.set_title("Chapter 8: Predicted vs. Actual")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(output_folder / "predicted_vs_actual.png", dpi=200)
plt.close(fig)

# Plot 2: residuals versus actual values.
fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    actual[~worst], residuals[~worst],
    color="steelblue", s=70, label="Other predictions"
)
ax.scatter(
    actual[worst], residuals[worst],
    color="red", s=90, label="Largest absolute errors"
)

ax.axhline(0, color="black", linestyle="--", label="Zero error")
ax.set_xlabel("Actual value")
ax.set_ylabel("Residual (predicted − actual)")
ax.set_title("Chapter 8: Residual Plot")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(output_folder / "residual_plot.png", dpi=200)
plt.close(fig)

print(f"\nBoth plots saved in: {output_folder}")

import pandas as pd

dataset_folder = output_folder / "Datasets"
lab_plot_folder = output_folder / "lab_plots"
lab_plot_folder.mkdir(exist_ok=True)

dataset_names = [
    "dataset_low_scatter",
    "dataset_high_scatter",
    "dataset_positive_deviation",
    "dataset_negative_deviation",
]

summary = []

for name in dataset_names:
    frame = pd.read_csv(dataset_folder / f"{name} (2).csv")
    lab_actual = frame["Actual"].to_numpy()
    lab_predicted = frame["Predicted"].to_numpy()
    lab_residuals = lab_predicted - lab_actual

    summary.append({
        "Dataset": name,
        "MAE": mean_absolute_error(lab_actual, lab_predicted),
        "MSE": mean_squared_error(lab_actual, lab_predicted),
        "R_squared": r2_score(lab_actual, lab_predicted),
        "Mean_residual": lab_residuals.mean(),
    })

    # Predicted versus actual plot.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lab_actual, lab_predicted, s=12, alpha=0.5)

    low = min(lab_actual.min(), lab_predicted.min())
    high = max(lab_actual.max(), lab_predicted.max())
    ax.plot([low, high], [low, high], "r--", label="Ideal fit")
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Predicted value")
    ax.set_title(name.replace("_", " ").title())
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        lab_plot_folder / f"{name}_predicted_vs_actual.png",
        dpi=200
    )
    plt.close(fig)

    # Residual plot.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lab_actual, lab_residuals, s=12, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--", label="Zero error")
    ax.set_xlabel("Actual value")
    ax.set_ylabel("Residual (predicted − actual)")
    ax.set_title(f"{name.replace('_', ' ').title()}: Residuals")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        lab_plot_folder / f"{name}_residuals.png",
        dpi=200
    )
    plt.close(fig)

metrics_table = pd.DataFrame(summary)
metrics_table.to_csv(output_folder / "lab_metrics.csv", index=False)

print("\nCHAPTER 8 DATASET COMPARISON")
print(metrics_table.to_string(index=False, float_format="%.6f"))
print(f"\nEight laboratory plots saved in: {lab_plot_folder}")