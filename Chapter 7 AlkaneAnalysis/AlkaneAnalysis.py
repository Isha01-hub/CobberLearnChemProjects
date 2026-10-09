from pathlib import Path
import pandas as pd

# Locate the dataset relative to this script.
project_folder = Path(__file__).resolve().parent.parent
csv_path = project_folder / "alkane_dataset.csv"
alkane_data = pd.read_csv(csv_path)

# Display the first 10 rows
print(alkane_data.head(10))
# Show the column names
print("\nColumn Names:")
print(alkane_data.columns)

# Count missing values in each column
print("\nMissing Values:")
print(alkane_data.isna().sum())

# Show the total number of rows
print("\nTotal Number of Rows:")
print(len(alkane_data))
# Remove rows that contain any missing values
complete_data = alkane_data.dropna()

# Count how many rows remain
remaining_rows = len(complete_data)
removed_rows = len(alkane_data) - remaining_rows

# Calculate percentage of dataset lost
percent_lost = (removed_rows / len(alkane_data)) * 100

print("\nRows Remaining After List-Wise Deletion:")
print(remaining_rows)

print("\nRows Removed:")
print(removed_rows)

print("\nPercentage of Dataset Lost:")
print(percent_lost)
# Compare molecule size before and after list-wise deletion
print("\nAverage Number of Carbons - Original Dataset:")
print(alkane_data["carbons"].mean())


print("\nAverage Number of Carbons - After List-Wise Deletion:")
print(complete_data["carbons"].mean())
# Look specifically at the rows removed by list-wise deletion
removed_data = alkane_data[alkane_data.isna().any(axis=1)]

print("\nAverage Number of Carbons - Removed Rows:")
print(removed_data["carbons"].mean())

print("\nAverage Number of Carbons - Kept Rows:")
print(complete_data["carbons"].mean())
print("\n=== MISSING VALUES SUMMARY ===")
print(alkane_data.isna().sum().to_string())
# Mean imputation for missing numerical values
mean_imputed_data = alkane_data.copy()

numeric_columns = mean_imputed_data.select_dtypes(include="number").columns

for column in numeric_columns:
    mean_imputed_data[column] = mean_imputed_data[column].fillna(
        mean_imputed_data[column].mean()
    )

print("\n=== AFTER MEAN IMPUTATION ===")
print(mean_imputed_data.isna().sum().to_string())

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

csv_path = Path(__file__).parent.parent / "alkane_dataset.csv"
data = pd.read_csv(csv_path)
features = ["carbons", "branch number"]

for property_name in [
    "viscosity",
    "heat capacity",
    "thermal conductivity"
]:
    # Use rows where this particular property is known.
    available = data.dropna(subset=[property_name])
    train, test = train_test_split(
        available, test_size=0.2, random_state=42
    )

    x_train = train[features]
    x_test = test[features]
    y_train = train[property_name]
    y_test = test[property_name]

    # Scaling matters when KNN measures distances between molecules.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    log_model = LinearRegression()
    log_model.fit(x_train, np.log(y_train))
    predictions = {
        "Log-linear": np.exp(log_model.predict(x_test))
    }

    for k in [3, 5, 10]:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(x_train_scaled, y_train)
        predictions[f"KNN k={k}"] = knn.predict(x_test_scaled)

    for depth in [5, 15]:
        forest = RandomForestRegressor(
            n_estimators=200,
            max_depth=depth,
            random_state=42
        )
        forest.fit(x_train, y_train)
        predictions[f"Forest depth={depth}"] = forest.predict(x_test)

    predictions["Ensemble"] = (
        predictions["Log-linear"]
        + predictions["KNN k=5"]
        + predictions["Forest depth=5"]
    ) / 3

    print(f"\nPROPERTY: {property_name}")
    print(f"Known rows: {len(available)}; test rows: {len(test)}")
    print("Model                 MAE          Errors over ±100%")

    for name, predicted in predictions.items():
        mae = mean_absolute_error(y_test, predicted)
        relative_error = (
            (predicted - y_test.to_numpy())
            / y_test.to_numpy()
            * 100
        )
        catastrophic = np.sum(np.abs(relative_error) > 100)
        print(f"{name:<21} {mae:<12.6f} {catastrophic}")
# Fill missing properties using models compared above.
model_imputed_data = data.copy()
imputation_flags = pd.DataFrame({"id": data["id"]})

chosen_models = {
    "viscosity": LinearRegression(),
    "heat capacity": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    ),
    "thermal conductivity": LinearRegression()
}

for property_name, model in chosen_models.items():
    known = data[property_name].notna()
    missing = data[property_name].isna()

    # Record which values are estimates.
    imputation_flags[f"{property_name}_imputed"] = missing

    x_known = data.loc[known, features]
    y_known = data.loc[known, property_name]
    x_missing = data.loc[missing, features]

    if property_name in ["viscosity", "thermal conductivity"]:
        model.fit(x_known, np.log(y_known))
        filled_values = np.exp(model.predict(x_missing))
    else:
        model.fit(x_known, y_known)
        filled_values = model.predict(x_missing)

    model_imputed_data.loc[missing, property_name] = filled_values
    print(f"Filled {missing.sum()} missing {property_name} values.")

output_folder = Path(__file__).resolve().parent

model_imputed_data.to_csv(
    output_folder / "alkane_model_imputed.csv",
    index=False
)
imputation_flags.to_csv(
    output_folder / "imputation_flags.csv",
    index=False
)

print("\nRemaining missing values:")
print(model_imputed_data.isna().sum().to_string())
print(f"\nResults saved in: {output_folder}")
import matplotlib.pyplot as plt
import seaborn as sns

# Remaining gaps are excluded pairwise when calculating correlations.
correlation = model_imputed_data.select_dtypes(
    include="number"
).corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    ax=ax
)
ax.set_title("Alkane Correlations After Model Imputation")
fig.tight_layout()

plot_path = output_folder / "alkane_correlation_heatmap.png"
fig.savefig(plot_path, dpi=200)
plt.close(fig)

correlation.to_csv(output_folder / "alkane_correlations.csv")
print(f"Correlation heatmap saved to: {plot_path}")