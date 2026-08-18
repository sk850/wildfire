"""
NOAA National Weather Service (NWS) fetcher.

Pulls current forecast + active alerts for a lat/lon point. This is the
live/near-term half of your feature set (paired with FIRMS for fire
history) — good for the streaming pipeline since it has no auth and no
strict rate limit, just a required User-Agent header.

Setup:
    pip install requests pandas

Usage:
    python nws_fetcher.py
"""

import time
from pathlib import Path
import os

import pandas as pd
import requests

NWS_BASE_URL = "https://api.weather.gov"

# NWS requires a descriptive User-Agent identifying your app + contact info.
# Requests without one get rejected — this isn't optional.
HEADERS = {
    "User-Agent": f"disaster-risk-dashboard ({os.environ.get("EMAIL")})",
    "Accept": "application/geo+json",
}

# Example point — swap in your target region's coordinates.
LATITUDE = 34.05
LONGITUDE = -118.25

OUTPUT_DIR = Path("data/raw/nws")


def get_point_metadata(lat: float, lon: float) -> dict:
    """Step 1 of the NWS API's linked-data pattern: given a lat/lon, find
    the URLs for that point's forecast, forecastHourly, and alerts."""
    url = f"{NWS_BASE_URL}/points/{lat},{lon}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_forecast(lat: float, lon: float, hourly: bool = False) -> pd.DataFrame:
    """Fetch the forecast for a point. Follows the /points -> /gridpoints
    linked-data chain rather than hardcoding a gridpoint URL, since office/
    grid codes vary by location and can change if NWS re-maps a region."""
    meta = get_point_metadata(lat, lon)
    forecast_key = "forecastHourly" if hourly else "forecast"
    forecast_url = meta["properties"][forecast_key]

    resp = requests.get(forecast_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    periods = resp.json()["properties"]["periods"]

    df = pd.DataFrame(periods)
    df["latitude"] = lat
    df["longitude"] = lon
    df["fetched_at"] = pd.Timestamp.now('UTC')
    return df


def get_active_alerts(lat: float, lon: float) -> pd.DataFrame:
    """Fetch active weather alerts (red-flag warnings, flood watches, etc.)
    for a point — useful both as a feature and as ground-truth-ish labels
    for 'was there an official warning here' during model evaluation."""
    url = f"{NWS_BASE_URL}/alerts/active"
    resp = requests.get(url, headers=HEADERS, params={"point": f"{lat},{lon}"}, timeout=15)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    if not features:
        return pd.DataFrame()

    rows = [f["properties"] for f in features]
    return pd.DataFrame(rows)


def save_snapshot(df: pd.DataFrame, name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    print(f"Fetching NWS forecast + alerts for ({LATITUDE}, {LONGITUDE})...")

    start = time.time()
    forecast_df = get_forecast(LATITUDE, LONGITUDE)
    alerts_df = get_active_alerts(LATITUDE, LONGITUDE)
    elapsed = time.time() - start

    print(f"Fetched in {elapsed:.1f}s — {len(forecast_df)} forecast periods, "
          f"{len(alerts_df)} active alerts")

    fpath = save_snapshot(forecast_df, "forecast")
    print(f"Saved forecast to {fpath}")

    if not alerts_df.empty:
        apath = save_snapshot(alerts_df, "alerts")
        print(f"Saved alerts to {apath}")
        print(alerts_df[["event", "severity", "headline"]].head())
    else:
        print("No active alerts for this point right now.")
