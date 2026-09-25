"""
01_train_validation_test_split.py

Prepare the freMTPL2 dataset for downstream GLM modeling.

This script:

1. Loads the prepared freMTPL2 dataset.
2. Splits the data into train, validation, and test sets.
3. Converts continuous rating variables into categorical deciles.
4. Treats missing values as their own category.
5. One-hot encodes all rating variables.
6. Ensures train, validation, and test have identical feature columns.
7. Saves the resulting datasets for downstream modeling.

The resulting datasets contain:

    - ClaimNb
    - Exposure
    - TotalClaimAmount
    - original non-rating variables needed downstream
    - one-hot encoded rating variables

All model features are numeric 0/1 values.

The modeling scripts therefore do NOT need to perform additional
categorical encoding.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = Path("data/processed/freMTPL2.csv")
OUTPUT_DIR = Path("data/processed")

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

RANDOM_STATE = 42

TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

# Continuous variables are converted into decile categories.
CONTINUOUS_VARIABLES = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density",
]

# Variables that are already categorical.
CATEGORICAL_VARIABLES = [
    "VehBrand",
    "VehGas",
    "Area",
    "Region",
]

RATING_VARIABLES = (
    CONTINUOUS_VARIABLES
    + CATEGORICAL_VARIABLES
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load the raw freMTPL2 dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE)


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def split_data(df):
    """
    Split data into train, validation, and test datasets.

    The split is performed before calculating decile boundaries or
    one-hot encoding so that information from validation/test does
    not leak into the training data.
    """

    train, temp = train_test_split(
        df,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    validation, test = train_test_split(
        temp,
        test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Continuous variables
# ---------------------------------------------------------------------------

def calculate_decile_edges(train, variable):
    """
    Calculate decile boundaries using the training data only.

    The resulting boundaries are reused for validation and test data.
    """

    values = train[variable].dropna()

    # qcut can produce duplicate boundaries when many observations
    # have the same value. np.unique removes duplicate edges.
    edges = np.quantile(
        values,
        np.linspace(0, 1, 11),
    )

    edges = np.unique(edges)

    return edges


def convert_to_deciles(train, validation, test):
    """
    Convert continuous rating variables to categorical deciles.

    Decile boundaries are calculated from training data only.

    Missing values remain missing at this stage and are subsequently
    converted to the explicit 'Missing' category.
    """

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    for variable in CONTINUOUS_VARIABLES:

        edges = calculate_decile_edges(
            train,
            variable,
        )

        # If there are not enough unique values to create ten
        # categories, pd.cut will simply use the available bins.
        train[variable] = pd.cut(
            train[variable],
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )

        validation[variable] = pd.cut(
            validation[variable],
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )

        test[variable] = pd.cut(
            test[variable],
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )

        # Convert intervals to strings so that all rating variables
        # can be handled consistently.
        train[variable] = train[variable].astype("object")
        validation[variable] = validation[variable].astype("object")
        test[variable] = test[variable].astype("object")

    return train, validation, test


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def fill_missing_categories(train, validation, test):
    """
    Convert missing rating-variable values into an explicit category.

    This means missing values are represented by:

        variable_Missing = 1

    rather than being dropped or imputed.
    """

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    for variable in RATING_VARIABLES:

        train[variable] = train[variable].astype("object").fillna("Missing")
        validation[variable] = (
            validation[variable]
            .astype("object")
            .fillna("Missing")
        )
        test[variable] = (
            test[variable]
            .astype("object")
            .fillna("Missing")
        )

    return train, validation, test


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def create_one_hot_features(train, validation, test):
    """
    Convert all rating variables into one-hot binary features.

    Categories are learned from the training data.

    The same columns are then applied to validation and test.

    Example:

        VehGas

        A
        B

    becomes:

        VehGas_A
        VehGas_B

    with values of 0 or 1.

    Unlike traditional GLM reference-cell encoding, we intentionally
    keep ALL one-hot columns here.

    This gives downstream scripts a completely numeric modeling dataset.
    """

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    encoded_train = pd.DataFrame(
        index=train.index
    )

    encoded_validation = pd.DataFrame(
        index=validation.index
    )

    encoded_test = pd.DataFrame(
        index=test.index
    )

    for variable in RATING_VARIABLES:

        # Learn categories from training data only.
        categories = sorted(
            train[variable].astype(str).unique()
        )

        train_category = pd.Categorical(
            train[variable].astype(str),
            categories=categories,
        )

        validation_category = pd.Categorical(
            validation[variable].astype(str),
            categories=categories,
        )

        test_category = pd.Categorical(
            test[variable].astype(str),
            categories=categories,
        )

        train_encoded = pd.get_dummies(
            train_category,
            prefix=variable,
            dtype=np.int8,
        )

        validation_encoded = pd.get_dummies(
            validation_category,
            prefix=variable,
            dtype=np.int8,
        )

        test_encoded = pd.get_dummies(
            test_category,
            prefix=variable,
            dtype=np.int8,
        )

        # Explicitly align validation/test to training columns.
        validation_encoded = validation_encoded.reindex(
            columns=train_encoded.columns,
            fill_value=0,
        )

        test_encoded = test_encoded.reindex(
            columns=train_encoded.columns,
            fill_value=0,
        )

        encoded_train = pd.concat(
            [encoded_train, train_encoded],
            axis=1,
        )

        encoded_validation = pd.concat(
            [encoded_validation, validation_encoded],
            axis=1,
        )

        encoded_test = pd.concat(
            [encoded_test, test_encoded],
            axis=1,
        )

    return (
        encoded_train,
        encoded_validation,
        encoded_test,
    )


# ---------------------------------------------------------------------------
# Assemble final datasets
# ---------------------------------------------------------------------------

def create_final_datasets(
    train,
    validation,
    test,
    encoded_train,
    encoded_validation,
    encoded_test,
):
    """
    Combine the original downstream variables with the one-hot
    encoded rating variables.

    The original rating variables are removed.
    """

    train_base = train.drop(
        columns=RATING_VARIABLES
    )

    validation_base = validation.drop(
        columns=RATING_VARIABLES
    )

    test_base = test.drop(
        columns=RATING_VARIABLES
    )

    train_final = pd.concat(
        [train_base, encoded_train],
        axis=1,
    )

    validation_final = pd.concat(
        [validation_base, encoded_validation],
        axis=1,
    )

    test_final = pd.concat(
        [test_base, encoded_test],
        axis=1,
    )

    # Ensure all datasets have exactly the same columns and ordering.
    validation_final = validation_final.reindex(
        columns=train_final.columns
    )

    test_final = test_final.reindex(
        columns=train_final.columns
    )

    return (
        train_final,
        validation_final,
        test_final,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_datasets(train, validation, test):
    """Perform basic checks on the resulting datasets."""

    print("\nValidating datasets...")

    # Same columns.
    assert list(train.columns) == list(validation.columns)
    assert list(train.columns) == list(test.columns)

    # No missing values in model features.
    feature_columns = [
        column
        for column in train.columns
        if column not in [
            "ClaimNb",
            "Exposure",
            "TotalClaimAmount",
        ]
    ]

    for dataset_name, dataset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        missing = dataset[feature_columns].isna().sum().sum()

        assert missing == 0, (
            f"{dataset_name} contains {missing} missing "
            "feature values."
        )

    # One-hot variables should contain only 0/1.
    rating_columns = [
        column
        for column in train.columns
        if any(
            column.startswith(f"{variable}_")
            for variable in RATING_VARIABLES
        )
    ]

    for dataset_name, dataset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        unique_values = np.unique(
            dataset[rating_columns].to_numpy()
        )

        assert np.all(
            np.isin(unique_values, [0, 1])
        ), (
            f"{dataset_name} contains non-binary "
            "rating features."
        )

    print("Validation passed.")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_datasets(train, validation, test):
    """Save train, validation, and test datasets."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    test.to_csv(
        TEST_FILE,
        index=False,
    )

    print("\nSaved:")
    print(f"  {TRAIN_FILE}")
    print(f"  {VALIDATION_FILE}")
    print(f"  {TEST_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("Loading data...")

    df = load_data()

    print(f"Observations: {len(df):,}")
    print(f"Variables:    {len(df.columns):,}")

    # -----------------------------------------------------------------------
    # Split
    # -----------------------------------------------------------------------

    print("\nCreating train / validation / test split...")

    train, validation, test = split_data(df)

    print(f"Train:       {len(train):,}")
    print(f"Validation:  {len(validation):,}")
    print(f"Test:        {len(test):,}")

    # -----------------------------------------------------------------------
    # Continuous -> categorical
    # -----------------------------------------------------------------------

    print("\nConverting continuous variables to deciles...")

    (
        train,
        validation,
        test,
    ) = convert_to_deciles(
        train,
        validation,
        test,
    )

    # -----------------------------------------------------------------------
    # Missing -> explicit category
    # -----------------------------------------------------------------------

    print("Converting missing values to explicit categories...")

    (
        train,
        validation,
        test,
    ) = fill_missing_categories(
        train,
        validation,
        test,
    )

    # -----------------------------------------------------------------------
    # One-hot encoding
    # -----------------------------------------------------------------------

    print("Creating one-hot encoded features...")

    (
        encoded_train,
        encoded_validation,
        encoded_test,
    ) = create_one_hot_features(
        train,
        validation,
        test,
    )

    # -----------------------------------------------------------------------
    # Final datasets
    # -----------------------------------------------------------------------

    (
        train_final,
        validation_final,
        test_final,
    ) = create_final_datasets(
        train,
        validation,
        test,
        encoded_train,
        encoded_validation,
        encoded_test,
    )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    validate_datasets(
        train_final,
        validation_final,
        test_final,
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    save_datasets(
        train_final,
        validation_final,
        test_final,
    )

    print("\nFinal dataset dimensions:")
    print(f"Train:       {train_final.shape}")
    print(f"Validation:  {validation_final.shape}")
    print(f"Test:        {test_final.shape}")


if __name__ == "__main__":
    main()

