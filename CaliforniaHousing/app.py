import pandas as pd
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("California Housing Predictor")
st.write(
    "Change the neighborhood details below to see how "
    "the predicted median house value changes."
)
st.caption(
    "Educational estimates based on 1990 census data. "
    "These are not current market prices or individual home appraisals."
)


@st.cache_resource
def prepare_model():
    """Train the model once and reuse it when inputs change."""
    housing = fetch_california_housing(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        housing.data,
        housing.target,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "model": model,
        "features": list(X_train.columns),
        "defaults": X_train.median().to_dict(),
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions) * 100_000
    }


try:
    with st.spinner("Preparing the model..."):
        bundle = prepare_model()
except Exception as error:
    st.error("Unable to prepare the model. Check your internet connection.")
    st.code(str(error))
    st.stop()


labels = {
    "MedInc": "Median income ($10,000 units)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Block group population",
    "AveOccup": "Average people per household",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}

# Set initial values from the training data.
for feature in bundle["features"]:
    if feature not in st.session_state:
        st.session_state[feature] = float(bundle["defaults"][feature])


def reset_inputs():
    """Restore the original input values."""
    for feature in bundle["features"]:
        st.session_state[feature] = float(bundle["defaults"][feature])


st.button("Reset to defaults", on_click=reset_inputs)

st.subheader("Neighborhood details")
st.caption(
    "Income example: 5.0 represents approximately $50,000. "
    "Press Enter after typing a value to update the prediction."
)

columns = st.columns(2)
values = {}

for index, feature in enumerate(bundle["features"]):
    with columns[index % 2]:
        if feature == "Latitude":
            values[feature] = st.number_input(
                labels[feature],
                min_value=32.0,
                max_value=42.0,
                step=0.01,
                format="%.4f",
                key=feature
            )

        elif feature == "Longitude":
            values[feature] = st.number_input(
                labels[feature],
                min_value=-125.0,
                max_value=-114.0,
                step=0.01,
                format="%.4f",
                key=feature
            )

        else:
            values[feature] = st.number_input(
                labels[feature],
                min_value=0.0001,
                step=1.0 if feature == "Population" else 0.1,
                format="%.4f",
                key=feature
            )


inputs = pd.DataFrame([values], columns=bundle["features"])
prediction = float(bundle["model"].predict(inputs)[0]) * 100_000

st.divider()
st.metric("Predicted median house value", f"${prediction:,.0f}")

with st.expander("Model performance and limitations"):
    st.write("Model: Random Forest")
    st.write(f"Test R²: {bundle['r2']:.4f}")
    st.write(f"Test mean absolute error: ${bundle['mae']:,.0f}")
    st.write(
        "The model uses an 80% training and 20% testing split. "
        "It estimates block-group median house values. "
        "The dataset's highest target values are capped near $500,000. "
        "Changing an input explores the model's predictions "
        "and does not establish cause and effect."
    )