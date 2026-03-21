"""Data loading and preprocessing for the INCC pumped storage project."""

import os
import pandas as pd

# Project root is one level up from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_ACTUALS = os.path.join(PROJECT_ROOT, "data", "actuals")
SUBMISSIONS = os.path.join(PROJECT_ROOT, "output")


def load_train():
    """Load and sort the training dataset with parsed datetime."""
    df = pd.read_csv(os.path.join(DATA_RAW, "train.csv"))
    df["datetime_start"] = pd.to_datetime(df["datetime_start"], utc=True)
    df = df.sort_values("datetime_start").reset_index(drop=True)
    return df


def load_test():
    """Load and sort the test dataset with parsed datetime."""
    df = pd.read_csv(os.path.join(DATA_RAW, "test.csv"))
    df["datetime_start"] = pd.to_datetime(df["datetime_start"], utc=True)
    df = df.sort_values("datetime_start").reset_index(drop=True)
    return df


def load_plant_metadata():
    """Load plant metadata (16 Spanish pumped storage units)."""
    return pd.read_csv(os.path.join(DATA_RAW, "plant_metadata.csv"))


def load_unavailability(kind="prod"):
    """Load unavailability data. kind='prod' or 'cons'."""
    filename = f"{kind}_unavailable.csv"
    return pd.read_csv(os.path.join(DATA_RAW, filename))


def load_actuals():
    """Load test actuals (ground truth) for evaluation."""
    return pd.read_csv(os.path.join(DATA_ACTUALS, "test_actuals.csv"))


def load_sample_submission():
    """Load the sample submission format."""
    return pd.read_csv(os.path.join(DATA_RAW, "sample_submission.csv"))


def save_submission(df, filename):
    """Save a submission DataFrame to the submissions folder."""
    path = os.path.join(SUBMISSIONS, filename)
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
    return path
