"""
06_deviance.py

Evaluate the Poisson frequency model using deviance.

Calculates:
    - Null deviance
    - Residual deviance
    - Mean Poisson deviance
    - Deviance explained
    - Individual deviance residuals

Uses the train, validation, and test predictions generated
by 02_poisson.py.

Poisson deviance measures how much information is lost when
using the model's predicted claim frequencies instead of the
observed claim frequencies.

Lower deviance indicates a better-fitting model.
"""

import numpy as np
import pandas as pd

from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_DIR = Path("data/processed")
PREDICTIONS_DIR = Path("predictions")

OUTPUT_DIR = Path("predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Poisson deviance
# ---------------------------------------------------------

def poisson_deviance(y, prediction):
    """
    Calculate total Poisson deviance.

    D = 2 * sum[
        y * log(y / prediction)
        - (y - prediction)
    ]

    When y = 0, the logarithmic term is defined as zero.

    Lower deviance indicates a better-fitting model.
    """

    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if np.any(prediction <= 0):
        raise ValueError("Predictions must be greater than zero.")

    if np.any(y < 0):
        raise ValueError("Observed values cannot be negative.")

    # -----------------------------------------------------
    # Poisson deviance
    # -----------------------------------------------------

    #
    # D = 2 * sum[
    #       y * log(y / prediction)
    #       - (y - prediction)
    #     ]
    #
    # When y = 0:
    #
    # The logarithmic term is zero because:
    #
    # lim(y * log(y)) = 0 as y approaches zero.
    #
    # Therefore:
    #
    # D = 2 * prediction
    #
    # -----------------------------------------------------

    deviance = np.where(
        y > 0,
        y * np.log(np.maximum(y, 1e-300) / prediction)
        - (y - prediction),
        prediction
    )

    return 2 * np.sum(deviance)


# ---------------------------------------------------------
# Deviance residuals
# ---------------------------------------------------------

def deviance_residuals(y, prediction):
    """
    Calculate individual Poisson deviance residuals.

    Residuals preserve the direction of the prediction error.

    Positive residual:
        Observed claims exceed predicted claims.

    Negative residual:
        Predicted claims exceed observed claims.
    """

    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if np.any(prediction <= 0):
        raise ValueError("Predictions must be greater than zero.")

    if np.any(y < 0):
        raise ValueError("Observed values cannot be negative.")

    deviance = np.where(
        y > 0,
        2 * (
            y * np.log(np.maximum(y, 1e-300) / prediction)
            - (y - prediction)
        ),
        2 * prediction
    )

    return np.sign(y - prediction) * np.sqrt(
        np.maximum(deviance, 0)
    )


# ---------------------------------------------------------
# Load predictions
# ---------------------------------------------------------

def load_predictions(split):
    """
    Load actual and predicted claim counts.

    Expected prediction files:
        predictions/poisson_train_predictions.csv
        predictions/poisson_validation_predictions.csv
        predictions/poisson_test_predictions.csv

    Expected columns:
        ClaimNb
        Prediction
    """

    path = PREDICTIONS_DIR / f"poisson_{split}_predictions.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path)

    # Identify prediction column.
    prediction_columns = [
        "Prediction",
        "prediction",
        "Predicted",
        "predicted",
        "PredictedClaimNb",
        "predicted_claims"
    ]

    prediction_column = next(
        (col for col in prediction_columns if col in df.columns),
        None
    )

    if prediction_column is None:
        raise ValueError(
            f"No prediction column found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    if "ClaimNb" not in df.columns:
        raise ValueError(f"ClaimNb column missing from {path}.")

    return df["ClaimNb"].to_numpy(), df[prediction_column].to_numpy()


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_deviance(y, prediction, null_prediction):
    """
    Calculate deviance metrics.

    Null deviance:
        Deviance from a model containing only an intercept
        and the exposure offset.

    Residual deviance:
        Deviance from the fitted Poisson model.

    Deviance explained:
        Proportion of null deviance eliminated by the model.
    """

    null_deviance = poisson_deviance(y, null_prediction)

    residual_deviance = poisson_deviance(y, prediction)

    mean_deviance = residual_deviance / len(y)

    deviance_explained = (
        1 - residual_deviance / null_deviance
        if null_deviance > 0
        else np.nan
    )

    return {
        "NullDeviance": null_deviance,
        "ResidualDeviance": residual_deviance,
        "MeanDeviance": mean_deviance,
        "DevianceExplained": deviance_explained
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Load training data
    # -----------------------------------------------------

    train = pd.read_csv(DATA_DIR / "train.csv")

    # -----------------------------------------------------
    # Calculate baseline claim frequency
    # -----------------------------------------------------

    #
    # The null model contains no predictor variables.
    #
    # However, it still includes the exposure offset:
    #
    # log(E[ClaimNb]) = log(Exposure) + intercept
    #
    # The maximum likelihood estimate of the baseline
    # frequency is:
    #
    #     Total Claims / Total Exposure
    #
    # Therefore:
    #
    #     Null Prediction = Exposure * Baseline Frequency
    #
    # -----------------------------------------------------

    baseline_frequency = (
        train["ClaimNb"].sum() / train["Exposure"].sum()
    )

    # -----------------------------------------------------
    # Evaluate each dataset
    # -----------------------------------------------------

    results = []

    for split in ["train", "validation", "test"]:

        y, prediction = load_predictions(split)

        if split == "train":
            df = train
        else:
            df = pd.read_csv(DATA_DIR / f"{split}.csv")

        exposure = df["Exposure"].to_numpy()

        null_prediction = exposure * baseline_frequency

        metrics = evaluate_deviance(
            y,
            prediction,
            null_prediction
        )

        metrics["Dataset"] = split

        results.append(metrics)

        # -------------------------------------------------
        # Save individual deviance residuals
        # -------------------------------------------------

        residuals = deviance_residuals(y, prediction)

        residual_df = pd.DataFrame({
            "ClaimNb": y,
            "Prediction": prediction,
            "DevianceResidual": residuals
        })

        residual_df.to_csv(
            OUTPUT_DIR / f"poisson_{split}_deviance_residuals.csv",
            index=False
        )

    # -----------------------------------------------------
    # Save summary metrics
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Dataset",
            "NullDeviance",
            "ResidualDeviance",
            "MeanDeviance",
            "DevianceExplained"
        ]
    ]

    results_df.to_csv(
        OUTPUT_DIR / "poisson_deviance_summary.csv",
        index=False
    )

    print("\nPoisson Deviance Summary")
    print("------------------------")

    print(results_df.to_string(index=False))

    print("\nFiles saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()