"""
07_overdispersion.py

Explore overdispersion in insurance claim frequency.

A Poisson distribution assumes:

    E[Y] = lambda
    Var(Y) = lambda

In other words, the mean and variance are equal.

However, real insurance claim data often exhibits overdispersion:

    Var(Y) > E[Y]

This script:

1. Simulates Poisson-distributed claim counts.
2. Simulates overdispersed claim counts.
3. Compares their means and variances.
4. Explains why overdispersion matters when fitting a Poisson GLM.

The goal is to understand the statistical behavior before
introducing more complicated models.
"""

import numpy as np


# ---------------------------------------------------------
# 1. Simulate Poisson-distributed claim counts
# ---------------------------------------------------------

# Number of insurance policies
n = 100_000

# Expected number of claims per policy
lambda_poisson = 0.20

# Generate claim counts from a Poisson distribution.
#
# Each policy has the same expected claim frequency.
#
# Because this is a Poisson distribution:
#
#   E[Y] = lambda
#   Var(Y) = lambda

poisson_claims = np.random.poisson(
    lam=lambda_poisson,
    size=n
)


# ---------------------------------------------------------
# 2. Simulate overdispersed claim counts
# ---------------------------------------------------------

# We introduce variation in the underlying claim frequency.
#
# Instead of every policy having the same lambda,
# we allow lambda to vary across policies.
#
# Some policies will have a higher underlying claim frequency.
# Others will have a lower underlying claim frequency.
#
# This introduces additional variance into the claim counts.

individual_lambda = np.random.gamma(
    shape=2.0,
    scale=0.10,
    size=n
)

# Generate claim counts using each policy's individual lambda.
#
# Conditional on lambda, the claim count is still Poisson.
#
# However, because lambda varies across policies,
# the overall claim count distribution becomes overdispersed.

overdispersed_claims = np.random.poisson(
    lam=individual_lambda
)


# ---------------------------------------------------------
# 3. Calculate descriptive statistics
# ---------------------------------------------------------

poisson_mean = np.mean(poisson_claims)
poisson_variance = np.var(poisson_claims)

overdispersed_mean = np.mean(overdispersed_claims)
overdispersed_variance = np.var(overdispersed_claims)


# ---------------------------------------------------------
# 4. Display results
# ---------------------------------------------------------

print("\nPOISSON DISTRIBUTION")
print("--------------------")

print(f"Mean:     {poisson_mean:.4f}")
print(f"Variance: {poisson_variance:.4f}")

print("\nOVERDISPERSED DISTRIBUTION")
print("-------------------------")

print(f"Mean:     {overdispersed_mean:.4f}")
print(f"Variance: {overdispersed_variance:.4f}")


# ---------------------------------------------------------
# 5. Calculate variance-to-mean ratio
# ---------------------------------------------------------

# For a Poisson distribution:
#
#   Variance / Mean = 1
#
# When the ratio is greater than 1, we have evidence
# of overdispersion.
#
# When the ratio is less than 1, we have underdispersion.

poisson_ratio = poisson_variance / poisson_mean

overdispersed_ratio = (
    overdispersed_variance / overdispersed_mean
)

print("\nVARIANCE-TO-MEAN RATIO")
print("---------------------")

print(f"Poisson:         {poisson_ratio:.4f}")
print(f"Overdispersed:   {overdispersed_ratio:.4f}")


# ---------------------------------------------------------
# 6. Theoretical explanation
# ---------------------------------------------------------

# The Poisson distribution has:
#
#   E[Y] = lambda
#   Var(Y) = lambda
#
# When lambda varies across observations, we can use
# the law of total variance:
#
#   Var(Y) = E[Var(Y | lambda)] + Var(E[Y | lambda])
#
# Since Y | lambda follows a Poisson distribution:
#
#   E[Y | lambda] = lambda
#   Var(Y | lambda) = lambda
#
# Therefore:
#
#   Var(Y) = E[lambda] + Var(lambda)
#
# This is the key to understanding overdispersion.
#
# The overall variance is now larger than the mean
# whenever Var(lambda) > 0.
#
# In our simulation:
#
#   lambda ~ Gamma(shape=2, scale=0.10)
#
# For a Gamma distribution:
#
#   E[lambda] = shape * scale
#   Var(lambda) = shape * scale^2
#
# Therefore:
#
#   E[lambda] = 2 * 0.10 = 0.20
#
#   Var(lambda) = 2 * 0.10^2 = 0.02
#
# And:
#
#   E[Y] = 0.20
#
#   Var(Y) = 0.20 + 0.02 = 0.22
#
# The variance is larger than the mean because
# we introduced variation in the underlying lambda.

theoretical_mean = 0.20

theoretical_variance = (
    0.20 + 0.02
)

print("\nTHEORETICAL EXPECTATIONS")
print("------------------------")

print(f"Expected mean:     {theoretical_mean:.4f}")
print(f"Expected variance: {theoretical_variance:.4f}")
