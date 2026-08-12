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

Land cover uses the USGS National Land Cover Database (NLCD), queried via
its public ArcGIS ImageServer "identify" endpoint — a point-in-time REST
lookup, no bulk raster download needed. NLCD only covers the US (+ PR/USVI);
if your target region is outside the US, you'd swap in a global product
like ESA WorldCover instead (same filtering logic, different source).

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

# USGS-hosted NLCD land cover MapServer (100m resolution). This is a public,
# no-auth-required identify endpoint. Point/param format follows the
# standard ArcGIS REST "identify" operation; verify against
# https://www.mrlc.gov/ or the service's own /identify docs if this ever
# 404s, since GIS service URLs do occasionally get reorganized.
NLCD_IDENTIFY_URL = "https://smallscale.nationalmap.gov/arcgis/rest/services/LandCover/MapServer/identify"

AGRICULTURAL_NLCD_CODES = {81, 82}  # Pasture/Hay, Cultivated Crops

# Confidence values differ by sensor:
#   VIIRS: "l" (low), "n" (nominal), "h" (high)
#   MODIS: 0-100 integer
MIN_MODIS_CONFIDENCE = 60          # drop below this for MODIS rows
DROP_VIIRS_LOW_CONFIDENCE = True   # drop rows where confidence == "l"

# FRP threshold in megawatts — small agricultural/industrial heat sources
# typically sit low; large wildfires spike much higher. This number is a
# starting point, not a validated constant — calibrate it against your
# target region's actual FRP distribution before trusting it (see
# `inspect_frp_distribution` below).
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

    Returns the NLCD class code. Raises on failure rather than returning
    None — a filter whose entire job is to exclude false positives must
    not silently degrade to "let everything through" when a lookup fails.
    Callers decide how to handle failures explicitly (see
    filter_by_landcover's on_lookup_failure parameter), instead of that
    decision being made implicitly here.
    """
    params = {
        "geometry": f"{longitude},{latitude}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": "all",
        "returnGeometry": "false",
        "f": "json",
    }

    resp = requests.get(NLCD_IDENTIFY_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None  # legitimate "outside coverage area" result, not a failure
    raw_value = results[0].get("value")
    return int(raw_value) if raw_value is not None else None


def filter_by_landcover(df: pd.DataFrame, request_delay: float = 0.1,
                         on_lookup_failure: str = "raise") -> pd.DataFrame:
    """Drop detections that fall on agricultural land (NLCD codes 81/82).

    on_lookup_failure controls what happens when the NLCD service is
    unreachable or errors out:
        "raise" (default) - stop immediately; you should know your filter
                             isn't working rather than silently pass
                             unfiltered data downstream
        "keep_unfiltered"  - let failed-lookup rows through as-is, but
                             print a loud warning with a count, so it's
                             visible in logs rather than invisible
        "drop"             - treat lookup failures as agricultural (safer
                             for precision, at the cost of losing some
                             legitimate wildfire detections you couldn't
                             classify)

    This does one NLCD lookup per row, which is slow for large batches —
    fine for a v1 project processing a few hundred detections at a time,
    but if you scale up, replace this with a local raster read (rasterio +
    a downloaded NLCD GeoTIFF clipped to your region) instead of one HTTP
    call per point.

    NOTE: this service currently serves NLCD 2001 land cover — over two
    decades old. Land use has changed since then; treat this as a rough
    filter, not ground truth, and consider a more recent land cover
    source (e.g. MRLC's newer NLCD releases, or ESA WorldCover) if
    precision here matters a lot to your results.
    """
    if df.empty:
        return df

    if on_lookup_failure not in {"raise", "keep_unfiltered", "drop"}:
        raise ValueError(f"Invalid on_lookup_failure: {on_lookup_failure}")

    keep_mask = []
    failures = 0

    for _, row in df.iterrows():
        lat = round(float(row["latitude"]), 4)
        lon = round(float(row["longitude"]), 4)

        misses_before = _get_nlcd_code.cache_info().misses

        try:
            nlcd_code = _get_nlcd_code(lat, lon)
        except (requests.RequestException, ValueError, KeyError) as e:
            failures += 1
            if on_lookup_failure == "raise":
                raise RuntimeError(
                    f"Land cover lookup failed for ({lat}, {lon}): {e}. "
                    f"Set on_lookup_failure='keep_unfiltered' or 'drop' to "
                    f"tolerate this, but understand the tradeoff first — "
                    f"see filter_by_landcover's docstring."
                ) from e
            elif on_lookup_failure == "keep_unfiltered":
                keep_mask.append(True)
                continue
            else:  # "drop"
                keep_mask.append(False)
                continue

        was_cache_miss = _get_nlcd_code.cache_info().misses > misses_before
        if was_cache_miss:
            time.sleep(request_delay)

        is_agricultural = nlcd_code in AGRICULTURAL_NLCD_CODES
        keep_mask.append(not is_agricultural)

    if failures > 0 and on_lookup_failure == "keep_unfiltered":
        print(f"  WARNING: {failures} land cover lookups failed and were "
              f"passed through UNFILTERED — these rows were not checked "
              f"for agricultural land.")

    return df[keep_mask].copy()


def filter_fire_detections(df: pd.DataFrame, apply_landcover: bool = True,
                            on_lookup_failure: str = "raise") -> pd.DataFrame:
    """Full pipeline: confidence -> FRP -> land cover. Each stage is cheap
    to expensive, so cheaper filters run first to shrink the dataset
    before the slower per-point land cover lookups.

    on_lookup_failure is passed through to filter_by_landcover — see its
    docstring. Default "raise" means a broken/unreachable land cover
    service stops the pipeline loudly rather than silently shipping
    unfiltered data.
    """
    before = len(df)

    df = filter_by_confidence(df)
    after_confidence = len(df)

    df = filter_by_frp(df)
    after_frp = len(df)

    if apply_landcover:
        df = filter_by_landcover(df, on_lookup_failure=on_lookup_failure)
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