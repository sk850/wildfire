"""
Filters FIRMS fire detections down to likely wildfire activity, excluding
agricultural burning, industrial heat, and other thermal false positives.

FIRMS has no field that says "this is a wildfire" — VIIRS/MODIS just
report thermal anomalies, full stop. This module applies three
progressively stronger filters:

    1. Confidence  (cheap, built into the FIRMS response already)
    2. FRP          (cheap, built into the FIRMS response already)
    3. Land cover   (requires an external lookup — this is the one that
                     actually targets your "exclude agriculture" question)


NLCD land cover codes relevant to this filter:
    81 = Pasture/Hay
    82 = Cultivated Crops
(both are "planted/cultivated" classes — i.e. agriculture)

Setup:
    pip install requests pandas
"""

import time
from functools import lru_cache

import pandas as pd
import requests


NLCD_IDENTIFY_URL = "https://smallscale.nationalmap.gov/arcgis/rest/services/LandCover/MapServer/identify"

AGRICULTURAL_NLCD_CODES = {81, 82}  # Pasture/Hay, Cultivated Crops

# Confidence values differ by sensor:
#   VIIRS: "l" (low), "n" (nominal), "h" (high)
#   MODIS: 0-100 integer
MIN_MODIS_CONFIDENCE = 60         
DROP_VIIRS_LOW_CONFIDENCE = True   


MIN_FRP_MW = 5.0


def filter_by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Drop low-confidence detections. Handles both VIIRS (letter codes)
    and MODIS (0-100 numeric) confidence formats, since which one you get
    depends on which SENSOR you queried FIRMS with."""
    if df.empty or "confidence" not in df.columns:
        return df

    is_viirs_format = df["confidence"].astype(str).isin(["l", "n", "h"]).any()

    if is_viirs_format:
        if DROP_VIIRS_LOW_CONFIDENCE:
            return df[df["confidence"] != "l"].copy()
        return df.copy()
    else:
        numeric_confidence = pd.to_numeric(df["confidence"], errors="coerce")
        return df[numeric_confidence >= MIN_MODIS_CONFIDENCE].copy()


def filter_by_frp(df: pd.DataFrame, min_frp: float = MIN_FRP_MW) -> pd.DataFrame:
    """Drop low-intensity detections likely to be small agricultural burns
    or industrial heat rather than significant fire activity."""
    if df.empty or "frp" not in df.columns:
        return df

    numeric_frp = pd.to_numeric(df["frp"], errors="coerce")
    return df[numeric_frp >= min_frp].copy()


def inspect_frp_distribution(df: pd.DataFrame) -> None:
    """Print FRP percentiles so you can calibrate MIN_FRP_MW to your actual
    region rather than trusting the default blind. Run this once on a
    sample of your target region's detections before locking in a
    threshold."""
    if df.empty or "frp" not in df.columns:
        print("No FRP data to inspect.")
        return

    numeric_frp = pd.to_numeric(df["frp"], errors="coerce").dropna()
    print("FRP distribution (MW):")
    print(numeric_frp.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))


@lru_cache(maxsize=10000)
def _get_nlcd_code(latitude: float, longitude: float) -> int | None:
    """Point lookup against the NLCD identify endpoint. Cached because
    FIRMS detections often cluster spatially (same fire, many pixels) —
    no need to re-query the same rounded coordinate repeatedly.

    Returns the NLCD class code, or None if the lookup fails (e.g. point
    falls outside CONUS coverage, or the service is unreachable) — treat
    None as "unknown", not "safe to include", when deciding what to do
    with it.
    """
    params = {
        "geometry": f"{longitude},{latitude}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": "all",
        "returnGeometry": "false",
        "f": "json",
    }

    try:
        resp = requests.get(NLCD_IDENTIFY_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        # "value" typically holds the raw class code as a string
        raw_value = results[0].get("value")
        return int(raw_value) if raw_value is not None else None
    except (requests.RequestException, ValueError, KeyError):
        return None


def filter_by_landcover(df: pd.DataFrame, request_delay: float = 0.1) -> pd.DataFrame:
    """Drop detections that fall on agricultural land (NLCD codes 81/82).

    This does one NLCD lookup per row, which is slow for large batches —
    fine for a v1 project processing a few hundred detections at a time,
    but if you scale up, replace this with a local raster read (rasterio +
    a downloaded NLCD GeoTIFF clipped to your region) instead of one HTTP
    call per point.

    request_delay adds a small pause between uncached lookups to be a
    polite API citizen — this is a shared public USGS service, not a
    dedicated endpoint for your project.
    """
    if df.empty:
        return df

    keep_mask = []
    for _, row in df.iterrows():
        lat = round(float(row["latitude"]), 4)
        lon = round(float(row["longitude"]), 4)

        misses_before = _get_nlcd_code.cache_info().misses
        nlcd_code = _get_nlcd_code(lat, lon)
        was_cache_miss = _get_nlcd_code.cache_info().misses > misses_before

        if was_cache_miss:
            time.sleep(request_delay)

        # Unknown land cover: keep the detection rather than silently
        # dropping it — you want a human to notice unresolved points, not
        # have them vanish. Flag separately if you want to audit these.
        is_agricultural = nlcd_code in AGRICULTURAL_NLCD_CODES
        keep_mask.append(not is_agricultural)

    return df[keep_mask].copy()


def filter_fire_detections(df: pd.DataFrame, apply_landcover: bool = True) -> pd.DataFrame:
    """Full pipeline: confidence -> FRP -> land cover. Each stage is cheap
    to expensive, so cheaper filters run first to shrink the dataset
    before the slower per-point land cover lookups."""
    before = len(df)

    df = filter_by_confidence(df)
    after_confidence = len(df)

    df = filter_by_frp(df)
    after_frp = len(df)

    if apply_landcover:
        df = filter_by_landcover(df)
    after_landcover = len(df)

    print(
        f"Filtering: {before} -> {after_confidence} (confidence) "
        f"-> {after_frp} (FRP) -> {after_landcover} (land cover)"
    )
    return df


if __name__ == "__main__":
    # Quick manual test against a couple of known points: one rural/
    # agricultural (should be filtered out), one forested (should survive).
    test_df = pd.DataFrame([
        {"latitude": 39.0, "longitude": -95.0, "confidence": "h", "frp": 20.0},  # Kansas farmland
        {"latitude": 39.3, "longitude": -120.3, "confidence": "h", "frp": 50.0},  # Sierra Nevada forest
    ])

    inspect_frp_distribution(test_df)
    filtered = filter_fire_detections(test_df)
    print(filtered)
