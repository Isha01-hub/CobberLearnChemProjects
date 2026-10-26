from sklearn.datasets import fetch_california_housing

# Load the dataset as a pandas DataFrame.
housing = fetch_california_housing(as_frame=True)
data = housing.frame

print("\nCALIFORNIA HOUSING DATASET")
print("=" * 50)

# Explain the dataset and its features.
print("\nDataset description:")
print(housing.DESCR)

# Preview the first five rows.
print("\nFirst five rows:")
print(data.head().to_string())

# Check the number of rows and columns.
print(f"\nRows: {data.shape[0]:,}")
print(f"Columns: {data.shape[1]}")

# Summarize the numerical values in each column.
print("\nSummary statistics:")
print(data.describe().to_string())

# Check for missing values.
print("\nMissing values:")
print(data.isna().sum())
import pandas as pd
from time import perf_counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Separate the eight input features from the target.
X = housing.data
y = housing.target

# Use the same training and testing data for every model.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Scale the features for SVR because it is sensitive to feature scales.
# The pipeline learns the scaling from the training data only.
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Support Vector Regressor": make_pipeline(
        StandardScaler(),
        SVR()
    )
}

results = []
trained_models = {}

print("\nMODEL COMPARISON")
print("=" * 50)

for name, model in models.items():
    print(f"\nTraining {name}...", flush=True)

    start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start

    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Training Time (s)": training_time,
        "R²": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions)
    })

    trained_models[name] = model
    print(f"Finished {name}.")

# Display the models from highest to lowest test R².
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="R²", ascending=False
).reset_index(drop=True)

print("\nResults:")
print(results_df.to_string(index=False, float_format="%.4f"))

best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]

print(f"\nHighest test R²: {best_model_name}")

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Save outputs beside this Python script.
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Save the model comparison table.
results_df.to_csv(output_dir / "model_comparison.csv", index=False)

# 1. Show the distribution of each input feature.
X.hist(bins=30, figsize=(14, 10), edgecolor="black")
plt.suptitle("California Housing Feature Distributions", fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig(output_dir / "feature_histograms.png", dpi=200)
plt.close()

# 2. Show correlations between the features and the target.
plt.figure(figsize=(12, 9))
sns.heatmap(
    data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("California Housing Correlation Heatmap")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
plt.close()

# 3. Compare the best model's predictions with actual values.
best_predictions = best_model.predict(X_test)

# Convert the target values to dollars.
actual_dollars = y_test * 100_000
predicted_dollars = best_predictions * 100_000

plt.figure(figsize=(8, 6))
plt.scatter(
    actual_dollars,
    predicted_dollars,
    alpha=0.25,
    s=12
)

low = min(actual_dollars.min(), predicted_dollars.min())
high = max(actual_dollars.max(), predicted_dollars.max())
plt.plot([low, high], [low, high], "r--", label="Ideal predictions")

plt.xlabel("Actual Median House Value ($)")
plt.ylabel("Predicted Median House Value ($)")
plt.title(f"{best_model_name}: Actual vs. Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "actual_vs_predicted.png", dpi=200)
plt.close()

# Print each feature's correlation with the target.
target_correlations = (
    data.corr()["MedHouseVal"]
    .drop("MedHouseVal")
    .sort_values(ascending=False)
)

print("\nFeature correlations with median house value:")
print(target_correlations.round(3).to_string())

print(f"\nGraphs and results saved to: {output_dir}")

import joblib

# Create three fictional census block groups.
# Change income while keeping the other features the same.
example_data = pd.DataFrame(
    [
        [3.0, 25, 5.0, 1.0, 1500, 3.0, 34.05, -118.25],
        [5.0, 25, 5.0, 1.0, 1500, 3.0, 34.05, -118.25],
        [7.0, 25, 5.0, 1.0, 1500, 3.0, 34.05, -118.25]
    ],
    columns=X.columns
)

# Predict median house values and convert them to dollars.
example_predictions = best_model.predict(example_data) * 100_000

example_results = example_data.copy()
example_results["Predicted Median Value ($)"] = example_predictions

print("\nEXAMPLE PREDICTIONS")
print("=" * 50)
print(
    example_results.to_string(
        index=False,
        float_format="%.2f"
    )
)

example_results.to_csv(
    output_dir / "example_predictions.csv",
    index=False
)

# Save the trained model and information needed by the GUI.
model_bundle = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_names": list(X.columns),
    "default_values": X_train.median().to_dict()
}

model_path = output_dir / "housing_model.joblib"
joblib.dump(model_bundle, model_path)

print(f"\nSaved model to: {model_path}")
import math

# Allow multiple predictions without retraining the model.
while True:
    answer = input("\nMake a prediction? (y/n): ").strip().lower()

    if answer == "n":
        print("Finished. Thank you!")
        break

    if answer != "y":
        print("Please enter y or n.")
        continue

    values = {}

    print("\nPress Enter to use each default value.")
    print("MedInc uses $10,000 units: 5 means approximately $50,000.")

    for feature in model_bundle["feature_names"]:
        default = model_bundle["default_values"][feature]

        while True:
            text = input(f"{feature} [{default:.4f}]: ").strip()

            try:
                value = default if text == "" else float(text)

                if not math.isfinite(value):
                    raise ValueError("Enter a finite number.")

                if feature == "Latitude":
                    if not 32 <= value <= 42:
                        raise ValueError("Use a latitude from 32 to 42.")

                elif feature == "Longitude":
                    if not -125 <= value <= -114:
                        raise ValueError("Use a longitude from -125 to -114.")

                elif value <= 0:
                    raise ValueError("Enter a value greater than zero.")

                values[feature] = value
                break

            except ValueError as error:
                print(f"Invalid input: {error}")

    inputs = pd.DataFrame(
        [values],
        columns=model_bundle["feature_names"]
    )

    price = float(best_model.predict(inputs)[0]) * 100_000
    print(f"\nPredicted median house value: ${price:,.0f}")