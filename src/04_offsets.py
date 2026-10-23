"""
04_offsets.py

GLM From Scratch: Understanding Exposure Offsets
================================================

Objective
---------
Understand why insurance GLMs use exposure offsets and how they work.

We will:

1. Define the insurance problem.
2. Calculate expected claim counts with and without an offset.
3. Fit a simple Poisson model with an exposure offset.
4. Fit a Poisson model without an exposure offset.
5. Compare the results.
6. Connect the example to the general Poisson GLM.

Dependencies
------------
NumPy only.

Key insurance concepts
----------------------
ClaimNb:
    The observed number of claims for a policy.

Exposure:
    The amount of time a policy was observed, measured in policy-years.

Frequency:
    The number of claims per policy-year.

Offset:
    A known component of the linear predictor whose coefficient
    is fixed at 1.

For a Poisson frequency model:

    log(mu) = log(Exposure) + X @ beta

Therefore:

    mu = Exposure * exp(X @ beta)

Where:
    mu is the expected claim count.
    Exposure is the policy's exposure.
    X contains predictor variables.
    beta contains the model coefficients.

Important:
    Exposure is not an ordinary predictor. Its coefficient is
    fixed at 1 on the log scale.
"""

import numpy as np


# ============================================================
# 1. CREATE A SMALL INSURANCE EXAMPLE
# ============================================================

# Each row represents one insurance policy.
#
# We will use an intercept-only model, meaning that we are not
# accounting for differences in risk between policyholders yet.
#
# This keeps the example focused on exposure.

# Observed number of claims for each policy.
#
# ClaimNb is the TARGET VARIABLE.
#
# We want to model the expected number of claims.

ClaimNb = np.array([
    0,
    1,
    0,
    2,
    1,
    0,
    1,
    1
], dtype=float)


# Exposure in policy-years.
#
# Exposure = 1.0 means one full policy-year.
# Exposure = 0.5 means six months.
# Exposure = 0.25 means three months.
#
# Policies can have different exposure periods because they
# entered or left the portfolio at different times.

Exposure = np.array([
    1.00,
    1.00,
    0.50,
    1.00,
    0.75,
    0.25,
    0.50,
    1.00
], dtype=float)


# Number of policies.
n = len(ClaimNb)


print("=" * 65)
print("1. INSURANCE DATA")
print("=" * 65)

print(f"Number of policies: {n}")
print(f"Total observed claims: {ClaimNb.sum():.0f}")
print(f"Total exposure: {Exposure.sum():.2f} policy-years")
print()


# ============================================================
# 2. CALCULATE THE OBSERVED CLAIM FREQUENCY
# ============================================================

# Claim frequency is the number of claims divided by exposure.
#
# Frequency = Total Claims / Total Exposure
#
# Notice that we do NOT simply divide the number of claims
# by the number of policies.
#
# Policies with less than one year of exposure contribute
# less observation time to the calculation.

total_claims = ClaimNb.sum()
total_exposure = Exposure.sum()

observed_frequency = total_claims / total_exposure

print("=" * 65)
print("2. OBSERVED CLAIM FREQUENCY")
print("=" * 65)

print(f"Total claims: {total_claims:.2f}")
print(f"Total exposure: {total_exposure:.2f} policy-years")
print(f"Observed frequency: {observed_frequency:.4f} claims/year")
print()


# ============================================================
# 3. MODEL WITHOUT AN EXPOSURE OFFSET
# ============================================================

# Suppose we fit an intercept-only Poisson model WITHOUT
# accounting for exposure.
#
# The model becomes:
#
#     log(mu) = beta_0
#
# Therefore:
#
#     mu = exp(beta_0)
#
# Every policy receives the same expected claim count,
# regardless of how long it was observed.
#
# For an intercept-only Poisson model, the maximum-likelihood
# estimate is:
#
#     mu = Average Observed Claim Count
#
#     beta_0 = log(Average Observed Claim Count)
#
# This model estimates claims per observed policy, not claims
# per policy-year.

average_claim_count = ClaimNb.mean()

# Avoid log(0) in a hypothetical dataset containing no claims.
# In that case, the finite intercept-only maximum-likelihood
# estimate does not exist; the likelihood increases as the
# intercept tends toward negative infinity.
if total_claims == 0:
    raise ValueError(
        "The example contains no claims. "
        "An intercept-only Poisson model cannot estimate a "
        "finite intercept when all observed counts are zero."
    )

beta_without_offset = np.log(average_claim_count)

# Predict the same expected claim count for every policy.
prediction_without_offset = np.full(
    shape=n,
    fill_value=np.exp(beta_without_offset)
)


print("=" * 65)
print("3. MODEL WITHOUT AN EXPOSURE OFFSET")
print("=" * 65)

print("Model:")
print("    log(mu) = beta_0")
print()

print(f"Estimated intercept: {beta_without_offset:.4f}")
print(f"Expected claims per policy: {average_claim_count:.4f}")
print()

print("Policy predictions:")

for i in range(n):
    print(
        f"Policy {i + 1}: "
        f"Exposure = {Exposure[i]:.2f}, "
        f"Observed claims = {ClaimNb[i]:.0f}, "
        f"Predicted claims = {prediction_without_offset[i]:.4f}"
    )

print()


# ============================================================
# 4. MODEL WITH AN EXPOSURE OFFSET
# ============================================================

# Now we explicitly account for exposure.
#
# The model becomes:
#
#     log(mu) = beta_0 + log(Exposure)
#
# Using logarithm rules:
#
#     mu = exp(beta_0 + log(Exposure))
#
#     mu = exp(beta_0) * Exposure
#
# Define:
#
#     annual_frequency = exp(beta_0)
#
# Then:
#
#     mu = Exposure * annual_frequency
#
# The annual frequency is the expected number of claims
# per policy-year.
#
# The exposure offset converts this annualized frequency
# into the expected claim count over the observed period.
#
# For an intercept-only Poisson model with an offset,
# the maximum-likelihood estimate is:
#
#     annual_frequency = Total Claims / Total Exposure
#
# Therefore:
#
#     beta_0 = log(Total Claims / Total Exposure)

annual_frequency = total_claims / total_exposure

beta_with_offset = np.log(annual_frequency)

# Calculate the expected number of claims for each policy.
#
# The offset is log(Exposure).
#
# Prediction:
#
#     mu = exp(beta_0 + log(Exposure))
#
# This is mathematically equivalent to:
#
#     mu = Exposure * exp(beta_0)

offset = np.log(Exposure)

linear_predictor = beta_with_offset + offset

prediction_with_offset = np.exp(linear_predictor)


print("=" * 65)
print("4. MODEL WITH AN EXPOSURE OFFSET")
print("=" * 65)

print("Model:")
print("    log(mu) = beta_0 + log(Exposure)")
print()

print(f"Estimated intercept: {beta_with_offset:.4f}")
print(f"Estimated annual frequency: {annual_frequency:.4f}")
print()

print("Policy predictions:")

for i in range(n):
    print(
        f"Policy {i + 1}: "
        f"Exposure = {Exposure[i]:.2f}, "
        f"Observed claims = {ClaimNb[i]:.0f}, "
        f"Predicted claims = {prediction_with_offset[i]:.4f}"
    )

print()


# ============================================================
# 5. COMPARE THE TWO MODELS
# ============================================================

# Without an offset:
#
#     Every policy receives the same predicted claim count.
#
# With an offset:
#
#     Expected claims increase proportionally with exposure.
#
# For example, if two policies have the same risk profile,
# but one has twice the exposure of the other, the model
# predicts twice as many claims for the policy with twice
# the exposure.
#
# This proportional relationship is an assumption of the
# exposure-offset model.

print("=" * 65)
print("5. MODEL COMPARISON")
print("=" * 65)

print(
    f"{'Policy':<10}"
    f"{'Exposure':<12}"
    f"{'Observed':<12}"
    f"{'Without offset':<18}"
    f"{'With offset':<15}"
)

print("-" * 67)

for i in range(n):
    print(
        f"{i + 1:<10}"
        f"{Exposure[i]:<12.2f}"
        f"{ClaimNb[i]:<12.0f}"
        f"{prediction_without_offset[i]:<18.4f}"
        f"{prediction_with_offset[i]:<15.4f}"
    )

print()

print(
    "Without offset, every policy receives the same expected "
    "claim count."
)

print(
    "With offset, expected claim counts reflect each policy's "
    "exposure."
)

print()


# ============================================================
# 6. VERIFY THE OFFSET MATHEMATICALLY
# ============================================================

# The general model is:
#
#     eta = beta_0 + log(Exposure)
#
#     mu = exp(eta)
#
# By exponentiating:
#
#     mu = exp(beta_0 + log(Exposure))
#
#     mu = exp(beta_0) * exp(log(Exposure))
#
# Since exp(log(Exposure)) = Exposure:
#
#     mu = Exposure * exp(beta_0)
#
# Let's verify that both calculations produce the same
# predictions.

prediction_from_linear_predictor = np.exp(
    beta_with_offset + np.log(Exposure)
)

prediction_from_multiplication = (
    Exposure * np.exp(beta_with_offset)
)

assert np.allclose(
    prediction_from_linear_predictor,
    prediction_from_multiplication
)

print("=" * 65)
print("6. MATHEMATICAL VERIFICATION")
print("=" * 65)

print("Calculation A:")
print("    mu = exp(beta_0 + log(Exposure))")
print()

print("Calculation B:")
print("    mu = Exposure * exp(beta_0)")
print()

print("Do both calculations produce the same predictions?")
print(
    np.allclose(
        prediction_from_linear_predictor,
        prediction_from_multiplication
    )
)

print()


# ============================================================
# 7. CONNECT THE EXAMPLE TO A GENERAL POISSON GLM
# ============================================================

# So far, we have used an intercept-only model:
#
#     log(mu) = beta_0 + log(Exposure)
#
# Real insurance models include risk factors.
#
# For example:
#
#     X = matrix of predictor variables
#
#     beta = vector of learned coefficients
#
# The model becomes:
#
#     log(mu) = X @ beta + log(Exposure)
#
# Here, X @ beta represents the linear combination of
# predictors, including an intercept if X contains a column
# of ones.
#
# The corresponding expected claim count is:
#
#     mu = Exposure * exp(X @ beta)
#
# The offset does not need to be learned. It is calculated
# directly from the observed exposure.

print("=" * 65)
print("7. GENERAL POISSON GLM")
print("=" * 65)

print("Linear predictor:")
print("    eta = X @ beta + log(Exposure)")
print()

print("Expected claim count:")
print("    mu = exp(eta)")
print()

print("Equivalent expression:")
print("    mu = Exposure * exp(X @ beta)")
print()

print("Important properties:")
print("    1. ClaimNb is the target.")
print("    2. Exposure is the known observation period.")
print("    3. X contains the insurance risk factors.")
print("    4. beta contains the learned coefficients.")
print("    5. The exposure coefficient is fixed at 1.")
print("    6. exp(X @ beta) represents annualized frequency.")
print()


# ============================================================
# 8. IMPORTANT DISTINCTION: OFFSET VERSUS WEIGHT
# ============================================================

# An offset and a statistical weight are not interchangeable.
#
# OFFSET:
#     Adds a known quantity to the linear predictor.
#
#     eta = X @ beta + log(Exposure)
#
#     This models expected claims over the observed exposure.
#
# WEIGHT:
#     Changes the contribution of an observation to an
#     estimation procedure or objective function.
#
# For a Poisson count model with exposure, the offset is a
# direct way to represent different observation periods.
#
# An alternative formulation models claim frequency:
#
#     Frequency = ClaimNb / Exposure
#
# Under the standard Poisson exposure model:
#
#     Frequency has expected value exp(X @ beta)
#
#     Its variance depends on exposure:
#
#         Var(Frequency) = exp(X @ beta) / Exposure
#
# Therefore, fitting claim frequency with the appropriate
# likelihood or weighted objective can be related to fitting
# claim counts with an exposure offset.
#
# However, simply dividing ClaimNb by Exposure and fitting
# an ordinary unweighted Poisson model is generally not
# equivalent to the count model with an offset.
#
# For our project, we will model ClaimNb directly and use
# log(Exposure) as the offset.

print("=" * 65)
print("8. OFFSET VERSUS WEIGHT")
print("=" * 65)

print("Offset:")
print("    Adds log(Exposure) to the linear predictor.")
print()

print("Weight:")
print("    Changes an observation's contribution to estimation.")
print()

print("Our modeling approach:")
print("    Target: ClaimNb")
print("    Offset: log(Exposure)")
print("    Link:   log")
print("    Mean:   Exposure * exp(X @ beta)")
print()


# ============================================================
# END OF SCRIPT
# ============================================================

print("=" * 65)
print("END OF 04_offsets.py")
print("=" * 65)