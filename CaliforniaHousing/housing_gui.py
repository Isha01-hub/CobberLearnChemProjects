from pathlib import Path
import math
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd


def main():
    """Open the housing prediction interface."""
    root = tk.Tk()
    root.title("California Housing Predictor")
    root.resizable(False, False)

    # Find the saved model inside this assignment's outputs folder.
    model_path = (
        Path(__file__).resolve().parent
        / "outputs"
        / "housing_model.joblib"
    )

    try:
        saved_data = joblib.load(model_path)
    except Exception as error:
        messagebox.showerror(
            "Model could not be loaded",
            "Run housing_regression.py to save the model first.\n\n"
            f"Details: {error}",
            parent=root
        )
        root.destroy()
        return

    model = saved_data["model"]
    feature_names = saved_data["feature_names"]
    defaults = saved_data["default_values"]

    labels = {
        "MedInc": "Median income ($10,000 units)",
        "HouseAge": "Median house age (years)",
        "AveRooms": "Average rooms per household",
        "AveBedrms": "Average bedrooms per household",
        "Population": "Block group population",
        "AveOccup": "Average people per household",
        "Latitude": "Latitude (32 to 42)",
        "Longitude": "Longitude (-125 to -114)"
    }

    frame = ttk.Frame(root, padding=20)
    frame.grid(row=0, column=0)

    ttk.Label(
        frame,
        text="California Housing Predictor",
        font=("Arial", 16, "bold")
    ).grid(row=0, column=0, columnspan=2, pady=(0, 10))

    ttk.Label(
        frame,
        text=f"Model: {saved_data['model_name']}"
    ).grid(row=1, column=0, columnspan=2, pady=(0, 15))

    entries = {}

    # Create an input field for every feature used during training.
    for row, feature in enumerate(feature_names, start=2):
        ttk.Label(
            frame,
            text=labels[feature]
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 15),
            pady=5
        )

        entry = ttk.Entry(frame, width=22)
        entry.grid(row=row, column=1, pady=5)
        entries[feature] = entry

    result = tk.StringVar()

    def reset():
        """Restore the training-data median values."""
        for feature, entry in entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, f"{defaults[feature]:.4f}")

        result.set("Enter values and click Predict Price.")

    def predict():
        """Validate the inputs and display the predicted value."""
        result.set("Enter values and click Predict Price.")
        values = {}

        for feature in feature_names:
            try:
                value = float(entries[feature].get())

                if not math.isfinite(value):
                    raise ValueError("Enter a finite number.")

                if feature == "Latitude":
                    if not 32 <= value <= 42:
                        raise ValueError(
                            "Latitude must be between 32 and 42."
                        )

                elif feature == "Longitude":
                    if not -125 <= value <= -114:
                        raise ValueError(
                            "Longitude must be between -125 and -114."
                        )

                elif value <= 0:
                    raise ValueError(
                        "The value must be greater than zero."
                    )

                values[feature] = value

            except ValueError as error:
                messagebox.showerror(
                    "Invalid input",
                    f"Check {labels[feature]}.\n\n{error}",
                    parent=root
                )
                entries[feature].focus_set()
                return

        # Preserve the same feature order used to train the model.
        inputs = pd.DataFrame([values], columns=feature_names)

        try:
            # The dataset's target is measured in $100,000 units.
            price = float(model.predict(inputs)[0]) * 100_000
            result.set(
                f"Predicted median house value: ${price:,.0f}"
            )
        except Exception as error:
            messagebox.showerror(
                "Prediction failed",
                str(error),
                parent=root
            )

    button_row = len(feature_names) + 2

    ttk.Button(
        frame,
        text="Predict Price",
        command=predict
    ).grid(row=button_row, column=0, pady=15)

    ttk.Button(
        frame,
        text="Reset",
        command=reset
    ).grid(row=button_row, column=1, pady=15)

    ttk.Label(
        frame,
        textvariable=result,
        font=("Arial", 11, "bold"),
        wraplength=460,
        justify="center"
    ).grid(
        row=button_row + 1,
        column=0,
        columnspan=2,
        pady=10
    )

    ttk.Label(
        frame,
        text=(
            "Educational estimate of a block group's median house value.\n"
            "Based on 1990 census data, not current market prices.\n"
            "Income example: 5.0 represents approximately $50,000."
        ),
        justify="center",
        wraplength=460
    ).grid(
        row=button_row + 2,
        column=0,
        columnspan=2,
        pady=(5, 0)
    )

    reset()
    root.mainloop()


if __name__ == "__main__":
    main()