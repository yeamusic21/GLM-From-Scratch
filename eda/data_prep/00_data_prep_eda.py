"""
00_data_prep_eda.py

Quick exploratory analysis of the freMTPL2 dataset.

For continuous variables:
    - Divide non-missing observations into 10 deciles.
    - Treat missing values as a separate "Missing" category.
    - Plot data volume as bars.
    - Plot average TotalClaimAmount as a line.

For categorical variables:
    - Treat missing values as a separate "Missing" category.
    - Calculate average TotalClaimAmount by category.
    - Order categories by average claim amount.
    - Plot data volume as bars.
    - Plot average TotalClaimAmount as a line.

Plots are saved to the same directory as this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent.parent / "data" / "processed" / "freMTPL2.csv"

TARGET = "TotalClaimAmount"

CONTINUOUS_VARIABLES = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density",
]

CATEGORICAL_VARIABLES = [
    "VehBrand",
    "VehGas",
    "Area",
    "Region",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print(f"Loading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print()


# ---------------------------------------------------------------------------
# Basic checks
# ---------------------------------------------------------------------------

required_columns = (
    CONTINUOUS_VARIABLES
    + CATEGORICAL_VARIABLES
    + [TARGET]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ---------------------------------------------------------------------------
# Continuous variables
# ---------------------------------------------------------------------------

print("Continuous variables")
print("-" * 80)

for variable in CONTINUOUS_VARIABLES:

    print(f"Processing {variable}...")

    # Keep missing values so they can be analyzed separately.
    data = df[[variable, TARGET]].copy()

    # -----------------------------------------------------------------------
    # Create deciles from non-missing observations
    # -----------------------------------------------------------------------

    non_missing = data[data[variable].notna()].copy()

    non_missing["Bucket"] = pd.qcut(
        non_missing[variable],
        q=10,
        labels=False,
        duplicates="drop",
    )

    # Convert zero-based deciles to 1-based deciles
    non_missing["Bucket"] = non_missing["Bucket"] + 1

    # -----------------------------------------------------------------------
    # Create missing bucket
    # -----------------------------------------------------------------------

    missing = data[data[variable].isna()].copy()
    missing["Bucket"] = "Missing"

    # Combine
    bucketed = pd.concat(
        [
            non_missing[["Bucket", TARGET]],
            missing[["Bucket", TARGET]],
        ],
        ignore_index=True,
    )

    # -----------------------------------------------------------------------
    # Summarize
    # -----------------------------------------------------------------------

    summary = (
        bucketed.groupby("Bucket", observed=True)
        .agg(
            AvgTotalClaimAmount=(TARGET, "mean"),
            DataVolume=(TARGET, "size"),
        )
        .reset_index()
    )

    # Keep deciles in their natural order and Missing last.
    decile_rows = summary[
        summary["Bucket"].apply(lambda x: isinstance(x, (int, float)))
    ].sort_values("Bucket")

    missing_rows = summary[
        summary["Bucket"] == "Missing"
    ]

    summary = pd.concat(
        [decile_rows, missing_rows],
        ignore_index=True,
    )

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = range(len(summary))

    # Data volume
    ax1.bar(
        x,
        summary["DataVolume"],
        alpha=0.6,
    )

    ax1.set_xlabel(f"{variable} Decile")
    ax1.set_ylabel("Data Volume")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(
        summary["Bucket"].astype(str)
    )

    # Average claim amount
    ax2 = ax1.twinx()

    ax2.plot(
        x,
        summary["AvgTotalClaimAmount"],
        marker="o",
        linewidth=2,
    )

    ax2.set_ylabel("Average TotalClaimAmount")

    ax1.set_title(
        f"{variable}: Data Volume and Average TotalClaimAmount"
    )

    ax1.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = SCRIPT_DIR / f"{variable.lower()}_decile.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Categorical variables
# ---------------------------------------------------------------------------

print()
print("Categorical variables")
print("-" * 80)

for variable in CATEGORICAL_VARIABLES:

    print(f"Processing {variable}...")

    data = df[[variable, TARGET]].copy()

    # Explicitly preserve missing values.
    data[variable] = data[variable].astype("object")
    data[variable] = data[variable].where(
        data[variable].notna(),
        "Missing",
    )

    # -----------------------------------------------------------------------
    # Summarize
    # -----------------------------------------------------------------------

    summary = (
        data.groupby(variable, observed=True)
        .agg(
            AvgTotalClaimAmount=(TARGET, "mean"),
            DataVolume=(TARGET, "size"),
        )
        .sort_values("AvgTotalClaimAmount")
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = range(len(summary))

    # Data volume
    ax1.bar(
        x,
        summary["DataVolume"],
        alpha=0.6,
    )

    ax1.set_xlabel(variable)
    ax1.set_ylabel("Data Volume")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(
        summary[variable].astype(str),
        rotation=45,
        ha="right",
    )

    # Average claim amount
    ax2 = ax1.twinx()

    ax2.plot(
        x,
        summary["AvgTotalClaimAmount"],
        marker="o",
        linewidth=2,
    )

    ax2.set_ylabel("Average TotalClaimAmount")

    ax1.set_title(
        f"{variable}: Data Volume and Average TotalClaimAmount"
    )

    ax1.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = SCRIPT_DIR / f"{variable.lower()}_category.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {output_path}")


print()
print("EDA complete.")