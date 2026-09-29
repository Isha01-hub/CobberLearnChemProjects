from typing import cast

import pandas as pd
import seaborn as sns

titanic =  sns.load_dataset("titanic")

print(titanic.head(10))
# Now we need to check how many values are missing in the Age column
print("\nMissing Age Values:")
print(titanic["age"].isna().sum())
# Now we need ot calculate the mean age. According to the book the simplest imputation strategy is to fill the missing
# age with the mean of the known ages. Pandas automatically ignores the Nan values when calculating mean()
mean_age = titanic["age"].mean()

print("\nMean Age:")
print(mean_age)
#Next step is to fill the missing ages
#Fill the missing Age values with the mean age
titanic["age"] = titanic["age"].fillna(mean_age)

print("\nMissing Age values after mean imputation:")
print(titanic["age"].isna().sum())
#Reload the original Titanic dataset for correlation analysis
titanic_original = sns.load_dataset("titanic")

#Select only numerical columns and calculate correlations
correlation_matrix = titanic_original.select_dtypes(include="number").corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

print("\nCorrelation with Age:")
print(correlation_matrix["age"].sort_values(ascending=False))
# Next we will do Correlation Plot
import matplotlib.pyplot as plt
# Create a correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()
# Next is KNN Imputation, and for that we need "scikit-learn".
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error
# Prepare numerical data for KNN imputation
knn_data = titanic_original[
    ["age", "pclass", "sibsp", "parch", "fare", "survived"]
    ].copy()
# Keep only rows where Age is known so we can test the model
known_ages = knn_data.dropna(subset=["age"]).copy()

print("\nNumber of passengers with known ages:")
print(len(known_ages))

from sklearn.model_selection import train_test_split

train, test = train_test_split(
    known_ages,
    test_size=0.2,
    random_state=42
)

print("Training passengers:", len(train))
print("Test passengers:", len(test))

features = ["pclass", "sibsp", "parch", "fare", "survived"]

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
feature_imputer = SimpleImputer(strategy="median")
train_features = feature_imputer.fit_transform(train[features])
test_features = feature_imputer.transform(test[features])

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

print("Training feature shape:", train_features.shape)
print("Test feature shape:", test_features.shape)

import numpy as np

# Put known training ages beside their prepared features.
train_for_knn = np.column_stack((train["age"].to_numpy(), train_features))

# Hide test ages; keep their features available.
test_for_knn = np.column_stack((
    np.full(len(test), np.nan),
    test_features
))

knn_imputer = KNNImputer(n_neighbors=5)
knn_imputer.fit(train_for_knn)

predicted_ages = knn_imputer.transform(test_for_knn)[:, 0]
mae = mean_absolute_error(test["age"], predicted_ages)

print(f"KNN mean absolute error: {mae:.2f} years")

from pathlib import Path

plt.figure(figsize=(7, 6))
plt.scatter(test["age"], predicted_ages, alpha=0.6)

maximum_age = max(test["age"].max(), predicted_ages.max())
plt.plot([0, maximum_age], [0, maximum_age], "r--",
         label="Perfect prediction")

plt.xlabel("Actual age (years)")
plt.ylabel("KNN-predicted age (years)")
plt.title("KNN age predictions on test passengers")
plt.legend()
plt.tight_layout()

plot_path = Path(__file__).parent / "knn_actual_vs_predicted.png"
plt.savefig(plot_path, dpi=150)
plt.show()

print("Saved plot to:", plot_path)

# Find passengers whose ages were originally missing.
missing_ages = knn_data[knn_data["age"].isna()].copy()

# Prepare their features in the same way as the known-age passengers.
all_known_features = feature_imputer.fit_transform(
    known_ages[features]
)
missing_features = feature_imputer.transform(
    missing_ages[features]
)

all_known_features = scaler.fit_transform(all_known_features)
missing_features = scaler.transform(missing_features)

# KNN needs age as the first column. Missing ages are marked with NaN.
all_known_for_knn = np.column_stack((
    known_ages["age"].to_numpy(),
    all_known_features
))
missing_for_knn = np.column_stack((
    np.full(len(missing_ages), np.nan),
    missing_features
))

# Learn from all known ages, then estimate the missing ones.
final_knn = KNNImputer(n_neighbors=5)
final_knn.fit(all_known_for_knn)
filled_ages = final_knn.transform(missing_for_knn)[:, 0]

# Fill only the missing ages in a copy of the original dataset.
titanic_knn = titanic_original.copy()
titanic_knn.loc[
    titanic_knn["age"].isna(), "age"
] = filled_ages

print(f"Mean age before KNN: {titanic_original['age'].mean():.2f}")
print(f"Mean age after KNN: {titanic_knn['age'].mean():.2f}")
print(
    "Missing ages after KNN:",
    titanic_knn["age"].isna().sum()
)
# The book's remaining coding task is to try a second model. Adding it to compare a random forest).
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=3,
    random_state=42
)

forest.fit(train_features, train["age"])
forest_predictions = forest.predict(test_features)
forest_mae = mean_absolute_error(test["age"], forest_predictions)

print(f"Random forest mean absolute error: {forest_mae:.2f} years")
print(f"KNN mean absolute error: {mae:.2f} years")
# Based on our comparison, our random forest MAE is 9.70 years, which is slightly lower than KNN's 9.96 years on the dame test passengers. That is a modest improvement for this split, but it is not a proof that every forest prediction is better.
titanic_knn["age_was_imputed"] = titanic_original["age"].isna()

csv_path = Path(__file__).parent / "titanic_age_imputed.csv"
titanic_knn.to_csv(csv_path, index=False)

print("Saved completed dataset to:", csv_path)
print("Estimated ages marked:", titanic_knn["age_was_imputed"].sum())