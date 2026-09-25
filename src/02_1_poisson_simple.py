"""
02_1_poisson_simple.py

A very simple Poisson frequency model.

This script intentionally uses:

    - 1 target
    - 1 independent variable
    - 1 exposure variable
    - no classes
    - no sklearn
    - lots of comments

The goal is to understand exactly what a Poisson insurance
frequency model is doing.

Our model is:

    log(mu) = log(Exposure) + beta_0 + beta_1 * X

where:

    mu       = expected number of claims
    Exposure = amount of time the policy was exposed to risk
    X        = our independent variable
    beta_0   = intercept
    beta_1   = coefficient for X

Equivalently:

    mu = Exposure * exp(beta_0 + beta_1 * X)


IMPORTANT:

ClaimNb is our TARGET.

Exposure is NOT an independent variable.

Exposure is an OFFSET.

Our independent variable will be one of the one-hot encoded
VehPower variables created by 01_train_validation_test_split.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. FILE PATHS
# =============================================================================

# Get the root directory of the project.
#
# This file lives in:
#
#     project/
#         src/
#             02_1_poisson_simple.py
#
# parents[1] therefore gives us:
#
#     project/
#
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Our train/validation/test files were created by:
#
#     01_train_validation_test_split.py
#
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


# =============================================================================
# 2. LOAD THE DATA
# =============================================================================

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)


print("Train shape:", train.shape)
print("Validation shape:", validation.shape)
print("Test shape:", test.shape)


# =============================================================================
# 3. DEFINE OUR TARGET
# =============================================================================

# TARGET
#
# ClaimNb is the number of claims made by the policy.
#
# Examples:
#
#     ClaimNb = 0
#     ClaimNb = 1
#     ClaimNb = 2
#
# This is what we are trying to predict.
#
# In insurance terminology, this is the CLAIM COUNT.
#
# We are therefore building a FREQUENCY model.
#
# Frequency is approximately:
#
#     number of claims / exposure
#
#
# Why use a Poisson model?
#
# Because ClaimNb is a count:
#
#     0, 1, 2, 3, ...
#
# The Poisson distribution is a natural starting point
# for modeling count data.

TARGET = "ClaimNb"


# =============================================================================
# 4. DEFINE OUR EXPOSURE
# =============================================================================

# EXPOSURE
#
# Exposure tells us how much time the policy was actually
# exposed to risk.
#
# For example:
#
#     Exposure = 1.0
#
# means approximately one full policy year.
#
#     Exposure = 0.5
#
# means approximately half a policy year.
#
#     Exposure = 0.25
#
# means approximately three months of exposure.
#
#
# Exposure is VERY important in insurance frequency modeling.
#
# Suppose we have two policies:
#
#     Policy A:
#         Exposure = 1.0
#         Claims   = 1
#
#     Policy B:
#         Exposure = 0.1
#         Claims   = 1
#
# Both have one claim.
#
# But Policy B had one claim in only 10% of a year,
# which represents a much higher observed claim frequency.
#
# Therefore we don't simply want to model:
#
#     ClaimNb
#
# We want to model claim counts while accounting for
# how much exposure each observation represents.
#
#
# IMPORTANT:
#
# Exposure is NOT an independent variable.
#
# It is an OFFSET.
#
# The Poisson model will use:
#
#     log(Exposure)
#
# as part of the linear predictor.

EXPOSURE = "Exposure"


# =============================================================================
# 5. DEFINE OUR ONE INDEPENDENT VARIABLE
# =============================================================================

# INDEPENDENT VARIABLE
#
# For this simple example we are going to use ONE variable.
#
# 01_train_validation_test_split.py has already converted
# our original variables into one-hot encoded columns.
#
# For example, VehPower might have originally looked like:
#
#     VehPower
#     --------
#     5
#     6
#     7
#     5
#     4
#
# After preprocessing it could look something like:
#
#     VehPower_4
#     VehPower_5
#     VehPower_6
#     VehPower_7
#
# with values of 0 or 1.
#
# For this simple model, we will use ONE of those columns.
#
# Change this name if the exact column name in your CSV is different.
#
# For example:
#
#     "VehPower_6"
#
# means:
#
#     1 -> vehicle power is 6
#     0 -> vehicle power is not 6
#
#
# This is our X.
#
# X = independent variable
#
# ClaimNb = target / dependent variable
#
# Exposure = offset

INDEPENDENT_VARIABLE = "VehPower_(3.999, 5.0]"


# =============================================================================
# 6. EXTRACT X, Y, AND EXPOSURE
# =============================================================================

# X = INDEPENDENT VARIABLE
#
# This is the thing we are asking:
#
#     "Does this characteristic help explain differences
#      in claim frequency?"
#
# Because our variable is one-hot encoded, X will contain
# only 0s and 1s.

X = train[INDEPENDENT_VARIABLE].to_numpy(dtype=float)


# Y = TARGET
#
# This is the actual number of claims observed for each policy.

y = train[TARGET].to_numpy(dtype=float)


# EXPOSURE
#
# This tells the model how much risk exposure each policy
# represents.

exposure = train[EXPOSURE].to_numpy(dtype=float)


# =============================================================================
# 7. LOOK AT WHAT WE ARE MODELING
# =============================================================================

print("\nFirst 10 observations:")
print()

for i in range(10):

    print(
        f"Observation {i}: "
        f"X={X[i]:.0f}, "
        f"ClaimNb={y[i]:.0f}, "
        f"Exposure={exposure[i]:.3f}"
    )


# At this point we have three important pieces of information:
#
#
#     X
#     |
#     |---- Independent variable
#     |
#     |---- Example: VehPower_6
#     |
#     |---- 0 or 1
#
#
#     y
#     |
#     |---- Target
#     |
#     |---- ClaimNb
#     |
#     |---- 0, 1, 2, ...
#
#
#     exposure
#     |
#     |---- Offset
#     |
#     |---- Amount of time exposed to risk
#     |
#     |---- Usually between 0 and 1 in this dataset
#
#
# The model will try to learn the relationship between X
# and y while accounting for exposure.


# =============================================================================
# 8. ADD AN INTERCEPT
# =============================================================================

# Our model contains TWO coefficients:
#
#     beta_0 = intercept
#     beta_1 = coefficient for X
#
#
# Mathematically:
#
#     beta_0 + beta_1 * X
#
#
# We therefore need a design matrix containing:
#
#     column 1 = 1
#     column 2 = X
#
#
# For example:
#
#     X = 1
#
# becomes:
#
#     [1, 1]
#
#
# and:
#
#     X = 0
#
# becomes:
#
#     [1, 0]

X_matrix = np.column_stack(
    [
        np.ones(len(X)),
        X,
    ]
)


# =============================================================================
# 9. INITIALIZE THE COEFFICIENTS
# =============================================================================

# We need to estimate:
#
#     beta_0
#     beta_1
#
# We'll start them both at zero.
#
# These are NOT the final values.
#
# The fitting algorithm will repeatedly update them until
# it finds values that provide a good fit to the data.

beta = np.zeros(2)


# So initially:
#
#     beta[0] = beta_0 = 0
#     beta[1] = beta_1 = 0

print("\nInitial coefficients:")
print("Intercept:", beta[0])
print("X coefficient:", beta[1])


# =============================================================================
# 10. CALCULATE THE OFFSET
# =============================================================================

# The Poisson model works on the logarithmic scale.
#
# The exposure therefore enters as:
#
#     log(Exposure)
#
#
# This is called the OFFSET.
#
# Notice that we are NOT estimating a coefficient for exposure.
#
# There is no:
#
#     beta_exposure
#
# Instead, the coefficient is fixed at 1:
#
#     1 * log(Exposure)
#
#
# This means:
#
#     log(mu)
#
# =
#
#     log(Exposure)
#     + beta_0
#     + beta_1 * X

offset = np.log(exposure)


# =============================================================================
# 11. FIT THE MODEL
# =============================================================================

# We will use Newton-Raphson to estimate beta_0 and beta_1.
#
# This is essentially the same mathematical idea used by
# standard GLM implementations.
#
# We are intentionally writing it out here so we can see
# what is happening.


max_iterations = 100
tolerance = 1e-8


for iteration in range(max_iterations):

    # -------------------------------------------------------------------------
    # 11a. Calculate the linear predictor
    # -------------------------------------------------------------------------
    #
    # The linear predictor is:
    #
    #     eta = log(Exposure)
    #           + beta_0
    #           + beta_1 * X
    #
    # In matrix notation:
    #
    #     eta = offset + X_matrix @ beta
    #
    #
    # Remember:
    #
    #     beta[0] = beta_0
    #     beta[1] = beta_1

    eta = offset + X_matrix @ beta


    # -------------------------------------------------------------------------
    # 11b. Convert from log scale back to claim-count scale
    # -------------------------------------------------------------------------
    #
    # Our model is:
    #
    #     log(mu) = eta
    #
    # Therefore:
    #
    #     mu = exp(eta)
    #
    #
    # mu is the expected number of claims.
    #
    # For example:
    #
    #     mu = 0.10
    #
    # means the model expects 0.10 claims for that observation.
    #
    # It does NOT mean we predict that the policy literally has
    # 0.10 claims.
    #
    # It means that across many similar policies, the average
    # number of claims would be approximately 0.10.

    mu = np.exp(eta)


    # -------------------------------------------------------------------------
    # 11c. Calculate the gradient
    # -------------------------------------------------------------------------
    #
    # The gradient tells us how we should change the coefficients
    # to improve the model fit.
    #
    # For Poisson regression:
    #
    #     gradient = X' @ (y - mu)
    #
    # where:
    #
    #     y  = actual claim count
    #     mu = predicted expected claim count

    gradient = X_matrix.T @ (y - mu)


    # -------------------------------------------------------------------------
    # 11d. Calculate the Hessian
    # -------------------------------------------------------------------------
    #
    # The Hessian contains the second derivatives of the
    # log-likelihood.
    #
    # For Poisson regression:
    #
    #     Hessian = -X' W X
    #
    # where W is a diagonal matrix containing mu.
    #
    # We don't need to explicitly construct the diagonal matrix.
    #
    # This expression gives us the same result:

    weights = mu

    hessian = -(X_matrix.T * weights) @ X_matrix


    # -------------------------------------------------------------------------
    # 11e. Calculate the Newton-Raphson step
    # -------------------------------------------------------------------------
    #
    # We solve:
    #
    #     Hessian * step = gradient
    #
    # rather than explicitly calculating:
    #
    #     inverse(Hessian)
    #
    # This is numerically more stable.

    step = np.linalg.solve(
        hessian,
        gradient,
    )


    # -------------------------------------------------------------------------
    # 11f. Update the coefficients
    # -------------------------------------------------------------------------

    beta -= step


    # -------------------------------------------------------------------------
    # 11g. Check convergence
    # -------------------------------------------------------------------------
    #
    # If the coefficient updates have become extremely small,
    # we've effectively reached the solution.

    if np.max(np.abs(step)) < tolerance:

        print(
            f"\nModel converged after "
            f"{iteration + 1} iterations."
        )

        break


else:

    print(
        "\nWarning: model did not converge after "
        f"{max_iterations} iterations."
    )


# =============================================================================
# 12. LOOK AT THE FINAL MODEL
# =============================================================================

beta_0 = beta[0]
beta_1 = beta[1]


print("\nFinal model:")
print()
print(
    "log(mu) = "
    f"log(Exposure) + "
    f"{beta_0:.6f} + "
    f"{beta_1:.6f} * {INDEPENDENT_VARIABLE}"
)


# =============================================================================
# 13. INTERPRET THE COEFFICIENT
# =============================================================================

# The coefficient beta_1 is on the LOG scale.
#
# Therefore, beta_1 itself is not the easiest number to interpret.
#
# We exponentiate it:
#
#     relativity = exp(beta_1)
#
#
# Because our independent variable is one-hot encoded:
#
#     X = 0
#
# means the observation is NOT in this category.
#
#     X = 1
#
# means the observation IS in this category.
#
#
# Therefore:
#
#     exp(beta_1)
#
# tells us the multiplicative change in expected claim count
# associated with X changing from 0 to 1,
# holding exposure constant.

relativity = np.exp(beta_1)


print()
print("Independent variable:", INDEPENDENT_VARIABLE)
print("Coefficient:", beta_1)
print("Relativity:", relativity)


# =============================================================================
# 14. MAKE PREDICTIONS
# =============================================================================

# Now that we've estimated beta_0 and beta_1,
# we can calculate expected claim counts.
#
# The formula is:
#
#     mu = Exposure * exp(beta_0 + beta_1 * X)
#
#
# Notice that exposure comes back into the formula here.
#
# This is important.
#
# A policy with twice as much exposure should have roughly
# twice the expected number of claims, all else equal.

eta = (
    np.log(exposure)
    + beta_0
    + beta_1 * X
)

prediction = np.exp(eta)


# =============================================================================
# 15. LOOK AT SOME PREDICTIONS
# =============================================================================

print("\nFirst 10 predictions:")
print()

for i in range(10):

    print(
        f"Observation {i}: "
        f"X={X[i]:.0f}, "
        f"Exposure={exposure[i]:.3f}, "
        f"Actual Claims={y[i]:.0f}, "
        f"Predicted Claims={prediction[i]:.4f}"
    )


# =============================================================================
# 16. POISSON DEVIANCE
# =============================================================================

# We need a way to measure how well the model fits the data.
#
# One common metric for Poisson models is POISSON DEVIANCE.
#
# Lower deviance means the predictions are closer to the
# observed claim counts under the Poisson likelihood.


prediction_for_deviance = np.maximum(
    prediction,
    1e-12,
)


# When y = 0, the term:
#
#     y * log(y / prediction)
#
# causes a mathematical problem because log(0) is undefined.
#
# For y = 0, the Poisson deviance contribution simplifies to:
#
#     prediction

deviance = np.where(
    y == 0,

    prediction_for_deviance,

    y * np.log(
        y / prediction_for_deviance
    )
    - (y - prediction_for_deviance),
)


# The factor of 2 is part of the standard Poisson deviance formula.

poisson_deviance = 2 * np.mean(deviance)


print()
print(
    "Training Poisson deviance:",
    poisson_deviance,
)


# =============================================================================
# 17. APPLY THE MODEL TO VALIDATION DATA
# =============================================================================

# We now take the coefficients learned from TRAINING data
# and apply them to the validation data.
#
# IMPORTANT:
#
# We do NOT refit the model.
#
# The validation data is being used to see how the model
# performs on data it did not use to estimate beta_0 and beta_1.

X_validation = validation[
    INDEPENDENT_VARIABLE
].to_numpy(dtype=float)

y_validation = validation[
    TARGET
].to_numpy(dtype=float)

exposure_validation = validation[
    EXPOSURE
].to_numpy(dtype=float)


# Apply the exact same model:

eta_validation = (
    np.log(exposure_validation)
    + beta_0
    + beta_1 * X_validation
)

prediction_validation = np.exp(
    eta_validation
)


# Calculate validation deviance.

prediction_validation_for_deviance = np.maximum(
    prediction_validation,
    1e-12,
)

deviance_validation = np.where(
    y_validation == 0,

    prediction_validation_for_deviance,

    y_validation
    * np.log(
        y_validation
        / prediction_validation_for_deviance
    )
    - (
        y_validation
        - prediction_validation_for_deviance
    ),
)

poisson_deviance_validation = (
    2 * np.mean(deviance_validation)
)


print(
    "Validation Poisson deviance:",
    poisson_deviance_validation,
)


# =============================================================================
# 18. SAVE VALIDATION PREDICTIONS
# =============================================================================

# Add our predictions to the validation data so we can inspect
# them later.

validation_output = validation.copy()

validation_output[
    "PredictedClaimNb"
] = prediction_validation


# Create the predictions directory.

PREDICTION_DIR = PROJECT_ROOT / "predictions"

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Save the results.

validation_output.to_csv(
    PREDICTION_DIR
    / "poisson_simple_validation_predictions.csv",
    index=False,
)


print()
print(
    "Saved validation predictions to:"
)

print(
    PREDICTION_DIR
    / "poisson_simple_validation_predictions.csv"
)


# =============================================================================
# 19. THE MODEL IN ONE PLACE
# =============================================================================

# Everything we just did can ultimately be summarized as:
#
#
#     TARGET
#     -------
#     ClaimNb
#
#     "How many claims occurred?"
#
#
#     EXPOSURE / OFFSET
#     -----------------
#     Exposure
#
#     "How long was this policy exposed to risk?"
#
#
#     INDEPENDENT VARIABLE
#     --------------------
#     VehPower_6
#
#     "Does this vehicle characteristic help explain
#      differences in claim frequency?"
#
#
#     MODEL
#     -----
#
#     log(E[ClaimNb])
#
#         =
#
#     log(Exposure)
#
#         +
#
#     beta_0
#
#         +
#
#     beta_1 * VehPower_6
#
#
#     OR:
#
#
#     E[ClaimNb]
#
#         =
#
#     Exposure
#
#         *
#
#     exp(
#         beta_0
#         +
#         beta_1 * VehPower_6
#     )
#
#
# The next step after understanding this simple model is to
# replace the single VehPower_6 variable with ALL of the
# one-hot encoded rating variables.
