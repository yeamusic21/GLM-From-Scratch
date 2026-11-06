"""
05_dispersion.py

Explore dispersion in Generalized Linear Models (GLMs).

We will use the freMTPL2 motor insurance dataset and our Poisson
frequency model to investigate whether the observed variance in
claim counts is consistent with the Poisson distribution.

Topics:
    1. What is dispersion?
    2. The Poisson variance assumption.
    3. Pearson chi-square dispersion.
    4. Deviance dispersion.
    5. Overdispersion and underdispersion.
    6. Implications for insurance frequency modeling.

The Poisson distribution assumes:

    E[Y]   = mu
    Var(Y) = mu

In other words, the expected value and variance are equal.

If the observed variance is substantially greater than the expected
variance, we have overdispersion.

If the observed variance is substantially smaller than the expected
variance, we have underdispersion.

We will investigate this using two statistics:

    Pearson dispersion = Pearson chi-square / residual degrees of freedom

    Deviance dispersion = Poisson deviance / residual degrees of freedom
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_DIR = Path("data/processed")

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = DATA_DIR / "dispersion"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "ClaimNb"
EXPOSURE = "Exposure"

MAX_ITERATIONS = 100
TOLERANCE = 1e-8


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("Loading datasets...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
test = pd.read_csv(TEST_PATH)


# ---------------------------------------------------------
# Identify independent variables
# ---------------------------------------------------------
#
# The target is ClaimNb.
#
# Exposure is not an independent variable in the traditional
# sense. It is incorporated into the model through an offset.
#
# IDpol is an identifier and should not be used as a predictor.
#
# TotalClaimAmount is the severity / pure premium target and
# should not be used in a frequency model.
#
# All remaining columns are assumed to have already been
# encoded during data preparation.
# ---------------------------------------------------------

EXCLUDED_COLUMNS = {
    TARGET,
    EXPOSURE,
    "IDpol",
    "TotalClaimAmount",
}

FEATURES = [
    column
    for column in train.columns
    if column not in EXCLUDED_COLUMNS
]

print(f"Number of predictors: {len(FEATURES)}")


# ---------------------------------------------------------
# Prepare model matrices
# ---------------------------------------------------------
#
# X represents the independent variables.
#
# We add an intercept column of ones.
#
# The model is:
#
# log(mu) = X @ beta + log(Exposure)
#
# Where:
#
# mu       = expected claim count
# X        = independent variables
# beta     = model coefficients
# Exposure = policy exposure
#
# The log(Exposure) term is an offset.
# ---------------------------------------------------------

def prepare_data(df):

    X = df[FEATURES].to_numpy(dtype=float)

    # Add intercept.
    X = np.column_stack([
        np.ones(len(X)),
        X
    ])

    y = df[TARGET].to_numpy(dtype=float)

    exposure = df[EXPOSURE].to_numpy(dtype=float)

    offset = np.log(exposure)

    return X, y, exposure, offset


X_train, y_train, exposure_train, offset_train = prepare_data(train)

X_validation, y_validation, exposure_validation, offset_validation = prepare_data(validation)

X_test, y_test, exposure_test, offset_test = prepare_data(test)


# ---------------------------------------------------------
# Poisson negative log-likelihood
# ---------------------------------------------------------
#
# We will minimize the Poisson negative log-likelihood.
#
# Ignoring the constant log(y!) term:
#
# L(beta) = sum(mu - y * log(mu))
#
# Where:
#
# mu = exp(X @ beta + offset)
#
# Minimizing this is equivalent to maximizing the
# Poisson log-likelihood.
# ---------------------------------------------------------

def poisson_loss(X, y, offset, beta):

    eta = X @ beta + offset

    # Protect against numerical overflow.
    eta = np.clip(eta, -30, 30)

    mu = np.exp(eta)

    return np.sum(mu - y * eta)


# ---------------------------------------------------------
# Fit Poisson regression using Newton-Raphson
# ---------------------------------------------------------
#
# We derive the coefficients using the gradient and Hessian.
#
# Gradient:
#
#     gradient = X.T @ (y - mu)
#
# Hessian:
#
#     Hessian = X.T @ diag(mu) @ X
#
# Newton-Raphson update:
#
#     beta_new = beta + inverse(Hessian) @ gradient
#
# We use np.linalg.solve rather than explicitly calculating
# the matrix inverse.
# ---------------------------------------------------------

def fit_poisson(X, y, offset):

    n_samples, n_features = X.shape

    beta = np.zeros(n_features)

    print("\nFitting Poisson model...")

    for iteration in range(MAX_ITERATIONS):

        eta = X @ beta + offset

        eta = np.clip(eta, -30, 30)

        mu = np.exp(eta)

        # Gradient of the log-likelihood.
        gradient = X.T @ (y - mu)

        # Weighted Hessian.
        hessian = X.T @ (mu[:, None] * X)

        # Solve Hessian * update = gradient.
        #
        # Least squares provides a fallback if the Hessian
        # is singular or poorly conditioned.
        try:
            update = np.linalg.solve(hessian, gradient)

        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(
                hessian,
                gradient,
                rcond=None
            )[0]

        # Current loss.
        current_loss = poisson_loss(X, y, offset, beta)

        # Step-halving to avoid increasing the loss.
        step_size = 1.0

        while step_size > 1e-8:

            candidate = beta + step_size * update

            candidate_loss = poisson_loss(
                X,
                y,
                offset,
                candidate
            )

            if candidate_loss <= current_loss:
                break

            step_size *= 0.5

        beta_new = candidate

        difference = np.max(np.abs(beta_new - beta))

        beta = beta_new

        if difference < TOLERANCE:

            print(
                f"Converged after {iteration + 1} iterations."
            )

            break

    else:
        print("Warning: Maximum iterations reached.")

    return beta


# ---------------------------------------------------------
# Fit model on training data
# ---------------------------------------------------------

beta = fit_poisson(
    X_train,
    y_train,
    offset_train
)


# ---------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------
#
# Remember:
#
# log(mu) = X @ beta + log(Exposure)
#
# Therefore:
#
# mu = exp(X @ beta + log(Exposure))
#
# These are expected claim counts, not claim frequencies.
#
# Expected frequency can be calculated as:
#
# frequency = mu / Exposure
# ---------------------------------------------------------

def predict_poisson(X, offset, beta):

    eta = X @ beta + offset

    eta = np.clip(eta, -30, 30)

    return np.exp(eta)


mu_train = predict_poisson(
    X_train,
    offset_train,
    beta
)

mu_validation = predict_poisson(
    X_validation,
    offset_validation,
    beta
)

mu_test = predict_poisson(
    X_test,
    offset_test,
    beta
)


# ---------------------------------------------------------
# Calculate Poisson deviance
# ---------------------------------------------------------
#
# Poisson deviance measures how well our model predictions
# explain the observed claim counts.
#
# D = 2 * sum(
#       y * log(y / mu) - (y - mu)
#     )
#
# When y = 0, the logarithmic term is defined as zero.
#
# Lower deviance indicates a better fit.
# ---------------------------------------------------------

def poisson_deviance(y, mu):

    y = np.asarray(y)
    mu = np.asarray(mu)

    # Avoid division by zero.
    mu = np.maximum(mu, 1e-12)

    # Initialize logarithmic contribution.
    log_component = np.zeros_like(y, dtype=float)

    # Only calculate logarithms where y > 0.
    mask = y > 0

    log_component[mask] = (
        y[mask] * np.log(y[mask] / mu[mask])
    )

    deviance = 2 * np.sum(
        log_component - (y - mu)
    )

    return deviance


# ---------------------------------------------------------
# Calculate Pearson chi-square
# ---------------------------------------------------------
#
# The Pearson residual is:
#
#     r_i = (y_i - mu_i) / sqrt(mu_i)
#
# Squaring the residual:
#
#     r_i^2 = (y_i - mu_i)^2 / mu_i
#
# Summing squared Pearson residuals:
#
#     X^2 = sum((y_i - mu_i)^2 / mu_i)
#
# This measures the difference between observed and
# expected claim counts relative to the expected variance.
# ---------------------------------------------------------

def pearson_chi_square(y, mu):

    mu = np.maximum(mu, 1e-12)

    return np.sum(
        (y - mu) ** 2 / mu
    )


# ---------------------------------------------------------
# Calculate dispersion statistics
# ---------------------------------------------------------
#
# Degrees of freedom:
#
#     df = number of observations - number of parameters
#
# We estimated p parameters using the training dataset.
#
# For the training dataset, residual degrees of freedom
# are n - p.
#
# For validation and test, we use the same parameter count
# as a descriptive predictive diagnostic.
#
# These validation/test ratios are not formal in-sample
# goodness-of-fit dispersion estimates.
# ---------------------------------------------------------

def calculate_dispersion(y, mu, n_parameters):

    n = len(y)

    degrees_of_freedom = n - n_parameters

    deviance = poisson_deviance(y, mu)

    pearson = pearson_chi_square(y, mu)

    deviance_dispersion = (
        deviance / degrees_of_freedom
    )

    pearson_dispersion = (
        pearson / degrees_of_freedom
    )

    return {
        "observations": n,
        "degrees_of_freedom": degrees_of_freedom,
        "deviance": deviance,
        "pearson_chi_square": pearson,
        "deviance_dispersion": deviance_dispersion,
        "pearson_dispersion": pearson_dispersion,
    }


# ---------------------------------------------------------
# Evaluate dispersion
# ---------------------------------------------------------

n_parameters = X_train.shape[1]

results = {

    "train": calculate_dispersion(
        y_train,
        mu_train,
        n_parameters
    ),

    "validation": calculate_dispersion(
        y_validation,
        mu_validation,
        n_parameters
    ),

    "test": calculate_dispersion(
        y_test,
        mu_test,
        n_parameters
    ),
}

results_df = pd.DataFrame(results).T

results_df.index.name = "dataset"

print("\nDispersion Results")
print("=" * 70)

print(results_df.round(4))


# ---------------------------------------------------------
# Interpret dispersion
# ---------------------------------------------------------
#
# A dispersion statistic near 1 is consistent with the
# Poisson variance assumption.
#
# Greater than 1:
#     Overdispersion
#
# Less than 1:
#     Underdispersion
#
# Important:
#
# Dispersion is a diagnostic, not a definitive statistical
# test. Its interpretation depends on model specification,
# sample size, dependence, and the distribution of residuals.
# ---------------------------------------------------------

print("\nInterpretation")
print("=" * 70)

for dataset, row in results_df.iterrows():

    pearson = row["pearson_dispersion"]

    deviance = row["deviance_dispersion"]

    if pearson > 1.2:
        interpretation = "Potential overdispersion"

    elif pearson < 0.8:
        interpretation = "Potential underdispersion"

    else:
        interpretation = "Approximately consistent with Poisson"

    print(f"\n{dataset.upper()}")

    print(f"Pearson dispersion: {pearson:.4f}")

    print(f"Deviance dispersion: {deviance:.4f}")

    print(f"Interpretation: {interpretation}")


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output_path = OUTPUT_DIR / "dispersion_summary.csv"

results_df.to_csv(output_path)

print(f"\nSaved results to {output_path}")

# ---------------------------------------------------------
# WHY IS DEVIANCE / DEGREES OF FREEDOM ANCHORED AROUND 1?
# ---------------------------------------------------------
#
# Our Poisson model assumes:
#
#     Y_i ~ Poisson(mu_i)
#
# Therefore:
#
#     E[Y_i]   = mu_i
#     Var(Y_i) = mu_i
#
# After fitting the model, we calculate the residual deviance:
#
#     D = 2 * sum[
#           y_i * log(y_i / mu_i) - (y_i - mu_i)
#         ]
#
# We estimate p model parameters across n observations.
#
# Under certain regularity conditions, when the model is
# correctly specified and the sample size is sufficiently
# large, the residual deviance approximately follows a
# chi-square distribution:
#
#     D ~ Chi-Square(n - p)
#
# Where:
#
#     n = number of observations
#     p = number of estimated model parameters
#     n - p = residual degrees of freedom
#
# A chi-square distribution has an expected value equal
# to its degrees of freedom:
#
#     E[Chi-Square(k)] = k
#
# Therefore:
#
#     E[D] ~= n - p
#
# Dividing both sides by the degrees of freedom:
#
#     E[D / (n - p)] ~= 1
#
# THAT IS WHY DEVIANCE / DEGREES OF FREEDOM IS ANCHORED
# AROUND 1.
#
# If our model is correctly specified, we expect the
# residual deviance to be approximately equal to the
# residual degrees of freedom.
#
# The ratio gives us a normalized measure of how much
# variability exists relative to what our assumed
# probability distribution says we should expect.
#
# ---------------------------------------------------------
