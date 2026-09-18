"""
00_data_prep.py

Prepare the freMTPL2 dataset for GLM modeling.

This script:
1. Downloads the freMTPL2 frequency and severity data if necessary.
2. Combines frequency and severity data at the policy level.
3. Creates a clean modeling dataset.
4. Saves the result locally.

Dataset:
    freMTPL2 - French Motor Third-Party Liability Claims
"""

from pathlib import Path
import urllib.request

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FREQ_FILE = RAW_DIR / "freMTPL2freq.csv"
SEV_FILE = RAW_DIR / "freMTPL2sev.csv"

OUTPUT_FILE = PROCESSED_DIR / "freMTPL2.csv"

FREQ_URL = (
    "https://huggingface.co/datasets/mabilton/fremtpl2/"
    "resolve/main/freMTPL2freq.csv"
)

SEV_URL = (
    "https://huggingface.co/datasets/mabilton/fremtpl2/"
    "resolve/main/freMTPL2sev.csv"
)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, destination: Path) -> None:
    """Download a file if it does not already exist."""

    if destination.exists():
        print(f"Already exists: {destination}")
        return

    print(f"Downloading {destination.name}...")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request) as response:
        destination.write_bytes(response.read())

    print(f"Saved: {destination}")


def download_data() -> None:
    """Download the raw freMTPL2 files."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download_file(FREQ_URL, FREQ_FILE)
    download_file(SEV_URL, SEV_FILE)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the frequency and severity datasets."""

    frequency = pd.read_csv(FREQ_FILE)
    severity = pd.read_csv(SEV_FILE)

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

    Frequency data:
        One row per policy.

    Severity data:
        One row per claim.

    Claim amounts are aggregated to the policy level before joining.
    """

    severity_by_policy = (
        severity
        .groupby("IDpol", as_index=False)["ClaimAmount"]
        .sum()
        .rename(columns={"ClaimAmount": "TotalClaimAmount"})
    )

    data = frequency.merge(
        severity_by_policy,
        on="IDpol",
        how="left",
    )

    # Policies with no claims have no row in the severity data.
    data["TotalClaimAmount"] = (
        data["TotalClaimAmount"]
        .fillna(0.0)
    )

    return data


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_data(data: pd.DataFrame) -> None:
    """Save the prepared modeling dataset."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows:    {len(data):,}")
    print(f"Columns: {len(data.columns)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(runDL, runPrep) -> None:
    """Run the complete data preparation pipeline."""
    if runDL:
        download_data()

    if runPrep:
        frequency, severity = load_data()
        print(f"\nFrequency rows: {len(frequency):,}")
        print(f"Severity rows:  {len(severity):,}")

        data = prepare_data(
            frequency,
            severity,
        )

        save_data(data)

        print("\nData preparation complete.")


if __name__ == "__main__":
    main(False,True)