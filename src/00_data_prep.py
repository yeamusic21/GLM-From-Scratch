"""
00_data_prep.py

Prepare the freMTPL2 dataset for GLM modeling.

This script:
1. Downloads the freMTPL2 dataset if it is not already present.
2. Extracts the raw CSV files.
3. Combines frequency and severity data at the policy level.
4. Creates a clean modeling dataset.
5. Saves the result locally for subsequent scripts.

Dataset:
    freMTPL2 - French Motor Third-Party Liability Claims

Primary modeling target:
    Claim frequency / pure premium

Expected output:
    data/processed/freMTPL2.csv
"""

from pathlib import Path
import zipfile
import urllib.request

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "ramapriya/fre-mtpl2"
)

ZIP_PATH = RAW_DIR / "freMTPL2.zip"

FREQ_FILE = RAW_DIR / "freMTPL2freq.csv"
SEV_FILE = RAW_DIR / "freMTPL2sev.csv"

OUTPUT_FILE = PROCESSED_DIR / "freMTPL2.csv"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_data() -> None:
    """Download the freMTPL2 dataset if it is not already available."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if FREQ_FILE.exists() and SEV_FILE.exists():
        print("Raw freMTPL2 data already exists.")
        return

    print("Downloading freMTPL2 dataset...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    print(f"Downloaded: {ZIP_PATH}")


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_data() -> None:
    """Extract the raw CSV files from the downloaded ZIP archive."""

    if FREQ_FILE.exists() and SEV_FILE.exists():
        return

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(RAW_DIR)

    # Some versions of the download place the files in a subdirectory.
    # Move them to the expected location if necessary.
    for filename in ("freMTPL2freq.csv", "freMTPL2sev.csv"):
        target = RAW_DIR / filename

        if target.exists():
            continue

        matches = list(RAW_DIR.rglob(filename))

        if not matches:
            raise FileNotFoundError(
                f"Could not find {filename} after extracting the dataset."
            )

        matches[0].rename(target)

    print("Extraction complete.")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load frequency and severity datasets."""

    frequency = pd.read_csv(FREQ_FILE, sep=";")
    severity = pd.read_csv(SEV_FILE, sep=";")

    return frequency, severity


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def prepare_data(
    frequency: pd.DataFrame,
    severity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine frequency and severity data at the policy level.

    Frequency data contains one row per policy.

    Severity data contains one row per claim, so claims are aggregated
    back to policy_id before joining with the frequency data.
    """

    # Aggregate claim payments to policy level.
    severity_by_policy = (
        severity
        .groupby("IDpol", as_index=False)["ClaimAmount"]
        .sum()
        .rename(columns={"ClaimAmount": "TotalClaimAmount"})
    )

    # Join claim severity onto the policy-level frequency data.
    data = frequency.merge(
        severity_by_policy,
        on="IDpol",
        how="left",
    )

    # Policies without a claim have no row in the severity dataset.
    data["TotalClaimAmount"] = data["TotalClaimAmount"].fillna(0.0)

    # Sanity check: claim count and total severity should agree.
    # A policy with ClaimNb == 0 should have zero total claim amount.
    zero_claims_with_payment = (
        (data["ClaimNb"] == 0)
        & (data["TotalClaimAmount"] > 0)
    ).sum()

    if zero_claims_with_payment > 0:
        raise ValueError(
            "Found policies with zero claims but positive claim amounts."
        )

    return data


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_data(data: pd.DataFrame) -> None:
    """Save the prepared dataset."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved processed data to: {OUTPUT_FILE}")
    print(f"Rows:    {len(data):,}")
    print(f"Columns: {len(data.columns)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete data preparation pipeline."""

    download_data()
    extract_data()

    frequency, severity = load_data()

    print(f"Frequency rows: {len(frequency):,}")
    print(f"Severity rows:  {len(severity):,}")

    data = prepare_data(frequency, severity)

    save_data(data)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()