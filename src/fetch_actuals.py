"""Fetch actual Spanish hydro pumped storage production from Energy-Charts API."""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

API_URL = "https://api.energy-charts.info/public_power"


def fetch_month(year, month):
    """Fetch one month of Spanish pumped storage data (15-min resolution)."""
    start = f"{year}-{month:02d}-01T00:00Z"
    # Get first day of next month
    if month == 12:
        end = f"{year + 1}-01-01T00:00Z"
    else:
        end = f"{year}-{month + 1:02d}-01T00:00Z"

    print(f"  Fetching {year}-{month:02d}...", end=" ")
    resp = requests.get(API_URL, params={
        "country": "es",
        "start": start,
        "end": end,
    })
    resp.raise_for_status()
    data = resp.json()

    timestamps = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)

    gen = cons = None
    for pt in data["production_types"]:
        if pt["name"] == "Hydro pumped storage":
            gen = pt["data"]
        elif pt["name"] == "Hydro pumped storage consumption":
            cons = pt["data"]

    df = pd.DataFrame({
        "datetime_utc": timestamps,
        "ps_generation": gen,
        "ps_consumption": cons,
    })
    print(f"{len(df)} rows")
    return df


def main():
    # Fetch Jul 2024 (for first 2 test rows at CEST midnight) through Mar 2025
    # Extra month at each end ensures boundary hours have full 15-min data
    months = [(2024, m) for m in range(7, 13)] + [(2025, m) for m in range(1, 4)]
    chunks = []
    for year, month in months:
        chunk = fetch_month(year, month)
        chunks.append(chunk)
        time.sleep(0.5)  # be polite to the API

    raw = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal 15-min rows fetched (with overlaps): {len(raw)}")

    # Drop duplicate timestamps from month-boundary overlaps
    raw = raw.drop_duplicates(subset="datetime_utc").sort_values("datetime_utc").reset_index(drop=True)
    print(f"After dedup: {len(raw)}")

    # Net production = generation + consumption (consumption is already negative)
    raw["ps_net"] = raw["ps_generation"].fillna(0) + raw["ps_consumption"].fillna(0)

    # Resample from 15-min to 1-hour (average MW over the hour)
    raw = raw.set_index("datetime_utc")
    hourly = raw.resample("1h").mean()
    hourly = hourly.reset_index()
    print(f"Hourly rows after resampling: {len(hourly)}")

    # Load test.csv and align
    test = pd.read_csv("../data/raw/test.csv")
    test["datetime_utc"] = pd.to_datetime(test["datetime_start"], utc=True)

    merged = test[["id", "datetime_utc"]].merge(
        hourly[["datetime_utc", "ps_net", "ps_generation", "ps_consumption"]],
        on="datetime_utc",
        how="left",
    )

    matched = merged["ps_net"].notna().sum()
    missing = merged["ps_net"].isna().sum()
    print(f"\nMatched: {matched}/{len(test)} test rows")
    if missing > 0:
        print(f"Missing: {missing} rows")

    # Save
    out = merged[["id", "ps_net", "ps_generation", "ps_consumption"]].rename(
        columns={"ps_net": "es_total_ps_actual"}
    )
    out.to_csv("../data/actuals/test_actuals.csv", index=False)
    print(f"\nSaved to test_actuals.csv")

    # Quick stats
    vals = out["es_total_ps_actual"].dropna()
    print(f"\n=== Actual es_total_ps stats ===")
    print(f"  Mean:  {vals.mean():.1f} MW")
    print(f"  Std:   {vals.std():.1f} MW")
    print(f"  Min:   {vals.min():.1f} MW")
    print(f"  Max:   {vals.max():.1f} MW")


if __name__ == "__main__":
    main()
