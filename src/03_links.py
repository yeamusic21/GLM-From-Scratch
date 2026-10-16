"""
03_links.py

Demonstrate Generalized Linear Model (GLM) link functions using NumPy.

The purpose of this script is to understand how link functions connect
the linear predictor to the expected value of a target variable.

A GLM has three main components:

    1. Random component:
       The probability distribution of the target variable.

    2. Systematic component:
       The linear predictor, calculated using independent variables
       and model coefficients.

           eta = X @ beta

    3. Link function:
       A function that connects the expected target value to the
       linear predictor.

           g(mu) = eta

       Where:
           mu  = expected value of the target
           g   = link function
           eta = linear predictor

       The inverse link function converts the linear predictor back
       into the expected target value.

           mu = g_inverse(eta)


INSURANCE EXAMPLES
------------------

1. Poisson regression:
   Used for claim frequency.

   Target:
       ClaimNb

   Expected target:
       mu = E[ClaimNb]

   Link:
       log(mu) = eta

   Inverse link:
       mu = exp(eta)

   Because exp(eta) is always positive, the model cannot predict
   a negative expected claim count.

2. Gamma regression:
   Often used for positive claim severity.

   Target:
       Average claim cost, conditional on a claim occurring.

   A common link is the log link:

       log(mu) = eta

       mu = exp(eta)

3. Tweedie regression:
   Often used for insurance pure premium because it can model
   a distribution with a probability mass at zero and a continuous,
   positive distribution for nonzero claims.

   A common choice is the log link:

       log(mu) = eta

       mu = exp(eta)

4. Binomial regression:
   Used for binary outcomes, such as whether a policy has at
   least one claim.

   Target:
       HasClaim (0 or 1)

   Link:
       logit(p) = eta

   Inverse link:
       p = 1 / (1 + exp(-eta))


IMPORTANT
---------

This script demonstrates link functions only.

It does not fit a GLM, estimate coefficients, or calculate
claim frequency, severity, or pure premium from the dataset.
"""

import numpy as np


# ============================================================
# 1. IDENTITY LINK
# ============================================================
#
# The identity link is:
#
#     g(mu) = mu
#
# Therefore:
#
#     eta = mu
#
# The inverse link is:
#
#     mu = eta
#
# This is the simplest possible link function.
#
# It is commonly used in ordinary linear regression and can also
# be used in some GLMs.
#
# IMPORTANT:
# The identity link does not guarantee positive predictions.
# For example, a model could predict a negative claim cost.
# ============================================================


def identity_link(mu):
    """
    Convert the expected target value into the linear predictor.

    Parameters
    ----------
    mu : float or numpy.ndarray
        Expected target value.

    Returns
    -------
    float or numpy.ndarray
        Linear predictor eta.
    """

    return mu


def inverse_identity_link(eta):
    """
    Convert the linear predictor back into the expected target.

    With the identity link, the inverse is the identity itself.

        mu = eta
    """

    return eta


# ============================================================
# 2. LOG LINK
# ============================================================
#
# The log link is:
#
#     g(mu) = log(mu)
#
# Therefore:
#
#     eta = log(mu)
#
# To recover the expected target, apply the inverse link:
#
#     mu = exp(eta)
#
# WHY IS THIS USEFUL FOR INSURANCE?
#
# Claim frequency cannot be negative.
#
# Similarly, expected claim severity and pure premium should
# generally be nonnegative.
#
# The exponential function guarantees:
#
#     exp(eta) > 0
#
# for every finite value of eta.
#
# EXAMPLE:
#
#     eta = 0
#
#     mu = exp(0)
#
#     mu = 1
#
# If eta = 0.693:
#
#     mu = exp(0.693)
#
#     mu is approximately 2.
#
# The expected target has doubled relative to a linear predictor
# of zero.
# ============================================================


def log_link(mu):
    """
    Apply the log link.

        eta = log(mu)

    The expected target must be strictly positive.

    Parameters
    ----------
    mu : float or numpy.ndarray
        Positive expected target value.

    Returns
    -------
    float or numpy.ndarray
        Linear predictor eta.
    """

    mu = np.asarray(mu, dtype=float)

    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError(
            "The log link requires finite, strictly positive mu values."
        )

    return np.log(mu)


def inverse_log_link(eta):
    """
    Apply the inverse log link.

        mu = exp(eta)

    This converts the linear predictor into a positive expected
    target value.

    Parameters
    ----------
    eta : float or numpy.ndarray
        Linear predictor.

    Returns
    -------
    float or numpy.ndarray
        Positive expected target value.
    """

    eta = np.asarray(eta, dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        mu = np.exp(eta)

    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError(
            "The inverse log link produced a non-finite or "
            "non-positive prediction. Check the linear predictor."
        )

    return mu


# ============================================================
# 3. LOGIT LINK
# ============================================================
#
# The logit link is:
#
#     g(p) = log(p / (1 - p))
#
# Here, p represents a probability.
#
# The logit link converts probabilities between zero and one
# into values that can range from negative infinity to infinity.
#
# WHY IS THIS USEFUL?
#
# A linear predictor can be any real number:
#
#     -10, -2, 0, 2, 10, ...
#
# However, a probability must satisfy:
#
#     0 < p < 1
#
# The inverse logit, also called the logistic function, ensures
# the resulting probability is between zero and one.
#
# INVERSE LINK:
#
#     p = 1 / (1 + exp(-eta))
#
# EXAMPLE:
#
#     eta = 0
#
#     p = 1 / (1 + exp(0))
#
#     p = 0.5
#
# In insurance, this link can be used for a binary target such as
# whether a policyholder has at least one claim.
# ============================================================


def logit_link(p):
    """
    Apply the logit link.

        eta = log(p / (1 - p))

    The probability must be strictly between zero and one.

    Parameters
    ----------
    p : float or numpy.ndarray
        Probability.

    Returns
    -------
    float or numpy.ndarray
        Linear predictor eta.
    """

    p = np.asarray(p, dtype=float)

    if np.any(~np.isfinite(p)) or np.any((p <= 0) | (p >= 1)):
        raise ValueError(
            "The logit link requires probabilities strictly "
            "between zero and one."
        )

    return np.log(p / (1.0 - p))


def inverse_logit_link(eta):
    """
    Apply the inverse logit link.

        p = 1 / (1 + exp(-eta))

    The result is a probability between zero and one.

    This implementation uses separate formulas for positive
    and negative eta values to reduce the risk of numerical
    overflow.
    """

    eta = np.asarray(eta, dtype=float)

    if np.any(~np.isfinite(eta)):
        raise ValueError(
            "The inverse logit link requires finite eta values."
        )

    p = np.empty_like(eta)

    # For nonnegative eta:
    #
    #     p = 1 / (1 + exp(-eta))
    #
    # Since eta >= 0, exp(-eta) cannot overflow.

    nonnegative = eta >= 0

    p[nonnegative] = (
        1.0 / (1.0 + np.exp(-eta[nonnegative]))
    )

    # For negative eta, use an equivalent formula:
    #
    #     p = exp(eta) / (1 + exp(eta))
    #
    # Since eta < 0, exp(eta) cannot overflow.

    negative = ~nonnegative

    exp_eta = np.exp(eta[negative])

    p[negative] = exp_eta / (1.0 + exp_eta)

    return p


# ============================================================
# 4. POISSON FREQUENCY MODEL EXAMPLE
# ============================================================
#
# Now let's connect the link function to your freMTPL2 project.
#
# Suppose we want to predict claim frequency using one
# independent variable: BonusMalus.
#
# A simplified linear predictor is:
#
#     eta = beta_0 + beta_1 * BonusMalus
#
# Where:
#
#     beta_0      = intercept
#     beta_1      = coefficient for BonusMalus
#     BonusMalus  = independent variable
#
# For a Poisson model with a log link:
#
#     log(mu) = eta
#
# Therefore:
#
#     mu = exp(eta)
#
# Claim frequency is often modeled using exposure as well.
#
# With exposure:
#
#     log(mu) = log(Exposure) + eta
#
# Equivalently:
#
#     mu = Exposure * exp(eta)
#
# Here, mu is the expected CLAIM COUNT for the observation,
# not the annualized claim frequency.
#
# The expected claim frequency per unit of exposure is:
#
#     frequency = exp(eta)
#
# This distinction is important:
#
#     Expected claim count = Exposure * Expected frequency
#
# Exposure is an offset in the Poisson model. Its coefficient
# is fixed at one rather than estimated like an ordinary
# independent-variable coefficient.
# ============================================================


# ------------------------------------------------------------
# Define example model coefficients.
# ------------------------------------------------------------
#
# These are illustrative coefficients, NOT coefficients
# estimated from freMTPL2.
#
# In a real model, the training procedure estimates beta_0
# and beta_1 from the training data.

beta_0 = -2.0
beta_1 = 0.01


# ------------------------------------------------------------
# Define example policy observations.
# ------------------------------------------------------------
#
# BonusMalus is the independent variable.
#
# Exposure represents the proportion of a year for which
# each policy is observed.
#
# For example:
#
#     Exposure = 1.0 means one full year.
#     Exposure = 0.5 means half a year.
#
# These are illustrative observations.

bonus_malus = np.array([50.0, 100.0, 150.0])

exposure = np.array([1.0, 0.5, 1.0])


# ------------------------------------------------------------
# Step 1: Calculate the linear predictor.
# ------------------------------------------------------------
#
# eta = beta_0 + beta_1 * BonusMalus
#
# The linear predictor can be negative, zero, or positive.
#
# That is acceptable because eta is not itself the predicted
# claim count.

eta = beta_0 + beta_1 * bonus_malus


# ------------------------------------------------------------
# Step 2: Convert the linear predictor into claim frequency.
# ------------------------------------------------------------
#
# Apply the inverse log link:
#
#     frequency = exp(eta)
#
# The result is positive.
#
# This represents expected claims per unit of exposure.

predicted_frequency = inverse_log_link(eta)


# ------------------------------------------------------------
# Step 3: Calculate expected claim count.
# ------------------------------------------------------------
#
# The Poisson model uses exposure as an offset:
#
#     log(mu) = log(Exposure) + eta
#
# Exponentiating both sides gives:
#
#     mu = Exposure * exp(eta)
#
# Therefore:
#
#     expected_claim_count = exposure * predicted_frequency
#
# This is the quantity we compare with ClaimNb when fitting
# a Poisson claim-count model.

predicted_claim_count = exposure * predicted_frequency


# ============================================================
# 5. PRINT THE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("GLM LINK FUNCTION EXAMPLES")
print("=" * 70)


# ------------------------------------------------------------
# Identity link example
# ------------------------------------------------------------

print("\n1. IDENTITY LINK")
print("-" * 70)

identity_example = 5.0

print(f"Expected target (mu): {identity_example}")
print(f"Linear predictor:     {identity_link(identity_example)}")
print(
    f"Inverse link result:  "
    f"{inverse_identity_link(identity_example)}"
)


# ------------------------------------------------------------
# Log link example
# ------------------------------------------------------------

print("\n2. LOG LINK")
print("-" * 70)

log_example = 2.0

eta_log = log_link(log_example)
mu_log = inverse_log_link(eta_log)

print(f"Expected target (mu): {log_example}")
print(f"Linear predictor:     {eta_log:.6f}")
print(f"Inverse link result:  {mu_log:.6f}")


# ------------------------------------------------------------
# Logit link example
# ------------------------------------------------------------

print("\n3. LOGIT LINK")
print("-" * 70)

probability_example = 0.80

eta_logit = logit_link(probability_example)
probability_recovered = inverse_logit_link(eta_logit)

print(f"Original probability:  {probability_example:.6f}")
print(f"Linear predictor:      {eta_logit:.6f}")
print(f"Inverse link result:   {probability_recovered:.6f}")


# ------------------------------------------------------------
# Poisson frequency model example
# ------------------------------------------------------------

print("\n4. POISSON FREQUENCY MODEL")
print("-" * 70)

print(
    f"{'BonusMalus':>12} "
    f"{'Exposure':>12} "
    f"{'Linear Predictor':>18} "
    f"{'Frequency':>14} "
    f"{'Expected Claims':>18}"
)

print("-" * 78)

for i in range(len(bonus_malus)):
    print(
        f"{bonus_malus[i]:12.1f} "
        f"{exposure[i]:12.2f} "
        f"{eta[i]:18.6f} "
        f"{predicted_frequency[i]:14.6f} "
        f"{predicted_claim_count[i]:18.6f}"
    )


# ============================================================
# 6. VERIFY THAT THE LINK FUNCTIONS ARE INVERSES
# ============================================================
#
# A link function converts mu into eta.
#
# Its inverse converts eta back into mu.
#
# Therefore, applying a link and then its inverse should
# recover the original value, subject to floating-point
# precision.
# ============================================================


# Test the log link and its inverse.

original_mu = np.array([0.1, 0.5, 1.0, 2.0, 10.0])

recovered_mu = inverse_log_link(log_link(original_mu))

assert np.allclose(original_mu, recovered_mu)


# Test the logit link and its inverse.

original_p = np.array([0.01, 0.10, 0.25, 0.50, 0.90, 0.99])

recovered_p = inverse_logit_link(logit_link(original_p))

assert np.allclose(original_p, recovered_p)


# Test the identity link and its inverse.

original_identity = np.array([-2.0, 0.0, 1.0, 5.0])

recovered_identity = inverse_identity_link(
    identity_link(original_identity)
)

assert np.allclose(original_identity, recovered_identity)


print("\n5. INVERSE LINK VERIFICATION")
print("-" * 70)

print("Identity link: PASS")
print("Log link:      PASS")
print("Logit link:    PASS")

print("\nAll link function checks passed.")