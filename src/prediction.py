"""
BudgetPlanner - Prediction

Loads the trained ML model and predicts savings
for a new user's financial information.

Run:
    python src/prediction.py
"""

from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "budget_model.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run model_training.py first."
        )

    model = joblib.load(MODEL_PATH)

    if "pipeline" not in model:
        raise ValueError(
            "Invalid model file. "
            "Trained pipeline was not found."
        )

    return model


# ============================================================
# GET FEATURES
# ============================================================

def get_features(model):
    """Get input features used during training."""

    pipeline = model["pipeline"]

    preprocessor = pipeline.named_steps["preprocessor"]

    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    raise ValueError(
        "Could not find feature names in the trained model."
    )


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

def show_model_info(model):
    """Display basic model information."""

    print("\n" + "=" * 60)
    print("BUDGETPLANNER ML MODEL")
    print("=" * 60)

    print(
        "Target:",
        model.get("target_column", "Unknown")
    )

    print(
        "Random State:",
        model.get("random_state", "Unknown")
    )

    print("\nModel Results:")

    results = model.get("model_results")

    if results is None:
        print("No evaluation results available.")

    elif isinstance(results, pd.DataFrame):
        print(results.to_string(index=False))

    elif isinstance(results, list):

        for result in results:

            if isinstance(result, dict):

                print(
                    f"\nModel : "
                    f"{result.get('Model', result.get('model', 'Unknown'))}"
                )

                print(
                    f"MAE   : "
                    f"{result.get('MAE', result.get('mae', 'N/A'))}"
                )

                print(
                    f"RMSE  : "
                    f"{result.get('RMSE', result.get('rmse', 'N/A'))}"
                )

                print(
                    f"R2    : "
                    f"{result.get('R2', result.get('r2', 'N/A'))}"
                )

            else:
                print(result)

    else:
        print(results)

    print("=" * 60)


# ============================================================
# GET USER INPUT
# ============================================================

def get_user_input(features):
    """Collect financial information from the user."""

    print("\n" + "=" * 60)
    print("ENTER FINANCIAL DETAILS")
    print("=" * 60)

    data = {}

    for feature in features:

        name = feature.replace("_", " ").title()

        while True:

            value = input(f"{name}: ").strip()

            if not value:
                print("Please enter a value.")
                continue

            try:
                numeric_value = float(value)

                if numeric_value.is_integer():
                    numeric_value = int(numeric_value)

                data[feature] = numeric_value

            except ValueError:
                data[feature] = value

            break

    return data


# ============================================================
# PREDICT
# ============================================================

def predict(model, user_data):
    """Generate prediction."""

    pipeline = model["pipeline"]

    target = model["target_column"]

    input_data = pd.DataFrame([user_data])

    features = get_features(model)

    input_data = input_data[features]

    prediction = pipeline.predict(input_data)

    return target, float(prediction[0])


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # Load model
        model = load_model()

        print("\nModel loaded successfully.")

        # Display model information
        show_model_info(model)

        # Get features
        features = get_features(model)

        print("\nFeatures used by the model:")

        for feature in features:
            print(f"- {feature}")

        # Get user input
        user_data = get_user_input(features)

        # Display input
        print("\n" + "=" * 60)
        print("USER INPUT")
        print("=" * 60)

        print(
            pd.DataFrame([user_data]).to_string(
                index=False
            )
        )

        # Prediction
        target, prediction = predict(
            model,
            user_data
        )

        # Display result
        print("\n" + "=" * 60)
        print("PREDICTION")
        print("=" * 60)

        print(
            f"\nPredicted {target.replace('_', ' ').title()}: "
            f"{prediction:.2f}"
        )

        print(
            "\nPrediction completed successfully."
        )

        print("=" * 60)

    except FileNotFoundError as error:

        print(f"\nERROR: {error}")

    except ValueError as error:

        print(f"\nERROR: {error}")

    except KeyError as error:

        print(
            f"\nERROR: Required model information is missing: "
            f"{error}"
        )

    except Exception as error:

        print(
            f"\nUnexpected error occurred:\n"
            f"{type(error).__name__}: {error}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()