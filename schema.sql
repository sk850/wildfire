-- Postgres schema for the disaster risk pipeline's feature store.
--
-- THREE tables, covering two genuinely different data flows:
--
--   weather_features_live       -> LIVE inference input, sourced from NWS
--                                   forecasts via spark_transform.py.
--                                   Forward-looking only, no historical
--                                   depth. Used to feed an ALREADY-TRAINED
--                                   model real-time predictions.
--
--   historical_weather_features -> TRAINING predictors, sourced from the
--                                   Open-Meteo Archive API via
--                                   historical_weather_fetcher.py +
--                                   load_historical_features.py (a plain
--                                   batch job, not Spark streaming — the
--                                   data volume doesn't need it).
--
--   fire_events                 -> LABELS (did fire activity occur),
--                                   sourced from FIRMS.
--
-- weather_features_live and historical_weather_features must NEVER be
-- confused for one another. NWS has no historical backfill capability,
-- so weather_features_live can only ever contain data from whenever you
-- started running the fetcher onward — joining it against past
-- fire_events rows silently returns almost nothing. Training joins use
-- historical_weather_features instead. See spark_transform.py's module
-- docstring for the full explanation of why these were split.
--
-- Training-time joins use an explicit time lag (e.g. join weather from
-- day N to fire_events on day N+1) — never join same-day rows and treat
-- both as "features", or you leak the label into the training data.

CREATE DATABASE disaster_risk;  -- run this line separately/manually if your
                                 -- client doesn't allow CREATE DATABASE inside
                                 -- a multi-statement script (many don't)

\c disaster_risk

-- Optional but recommended: enables geospatial indexing/queries later
-- (nearest-neighbor lookups, radius queries) instead of only exact
-- lat/lon equality. Skip if you don't want the extra dependency for v1 —
-- everything below still works with plain DOUBLE PRECISION columns.
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================
-- weather_features_live: LIVE inference input, one row per (location, 6h window)
-- Sourced from NWS forecasts. Forward-looking only — see note above.
-- ============================================================
CREATE TABLE IF NOT EXISTS weather_features_live (
    id              BIGSERIAL PRIMARY KEY,
    window_start    TIMESTAMP NOT NULL,
    window_end      TIMESTAMP NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    avg_temp        DOUBLE PRECISION,
    avg_humidity    DOUBLE PRECISION,
    avg_wind_mph    DOUBLE PRECISION,
    inserted_at     TIMESTAMP NOT NULL DEFAULT now(),

    -- Prevents duplicate rows if Spark reprocesses a micro-batch after a
    -- restart (foreachBatch + checkpointing is at-least-once, not
    -- exactly-once, unless the sink itself is idempotent).
    CONSTRAINT weather_features_live_unique_window
        UNIQUE (window_start, latitude, longitude)
);

CREATE INDEX IF NOT EXISTS idx_weather_features_live_location
    ON weather_features_live (latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_weather_features_live_window_start
    ON weather_features_live (window_start);

-- ============================================================
-- historical_weather_features: TRAINING predictors, one row per
-- (location, day). Sourced from Open-Meteo's true historical archive —
-- this is the table with real depth going back years, which
-- weather_features_live can never have.
-- ============================================================
CREATE TABLE IF NOT EXISTS historical_weather_features (
    id                    BIGSERIAL PRIMARY KEY,
    observation_date      DATE NOT NULL,
    latitude              DOUBLE PRECISION NOT NULL,
    longitude             DOUBLE PRECISION NOT NULL,
    temp_max              DOUBLE PRECISION,
    temp_min              DOUBLE PRECISION,
    precipitation_mm      DOUBLE PRECISION,
    wind_speed_max_kmh    DOUBLE PRECISION,
    wind_gusts_max_kmh    DOUBLE PRECISION,
    inserted_at           TIMESTAMP NOT NULL DEFAULT now(),

    CONSTRAINT historical_weather_features_unique_day
        UNIQUE (observation_date, latitude, longitude)
);

CREATE INDEX IF NOT EXISTS idx_historical_weather_features_location
    ON historical_weather_features (latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_historical_weather_features_date
    ON historical_weather_features (observation_date);

-- ============================================================
-- fire_events: label/outcome data, one row per (location, day)
-- ============================================================
CREATE TABLE IF NOT EXISTS fire_events (
    id                     BIGSERIAL PRIMARY KEY,
    day_start              TIMESTAMP NOT NULL,
    day_end                TIMESTAMP NOT NULL,
    latitude               DOUBLE PRECISION NOT NULL,
    longitude              DOUBLE PRECISION NOT NULL,
    fire_detection_count   INTEGER NOT NULL DEFAULT 0,
    inserted_at            TIMESTAMP NOT NULL DEFAULT now(),

    CONSTRAINT fire_events_unique_day
        UNIQUE (day_start, latitude, longitude)
);

CREATE INDEX IF NOT EXISTS idx_fire_events_location
    ON fire_events (latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_fire_events_day_start
    ON fire_events (day_start);

-- ============================================================
-- CORRECT training-time join (reference only — not executed here)
-- ============================================================
-- Uses historical_weather_features (real historical depth), NOT
-- weather_features_live (forecast-only, no backfill). Labels a day as
-- "fire occurred" using detections from the FOLLOWING day relative to
-- the weather observation, and rounds lat/lon to bucket nearby points
-- into the same grid cell (adjust precision to your actual grid size).
--
-- SELECT
--     hwf.observation_date,
--     hwf.latitude, hwf.longitude,
--     hwf.temp_max, hwf.temp_min, hwf.precipitation_mm,
--     hwf.wind_speed_max_kmh, hwf.wind_gusts_max_kmh,
--     COALESCE(fe.fire_detection_count, 0) > 0 AS fire_occurred
-- FROM historical_weather_features hwf
-- LEFT JOIN fire_events fe
--     ON round(fe.latitude::numeric, 2) = round(hwf.latitude::numeric, 2)
--    AND round(fe.longitude::numeric, 2) = round(hwf.longitude::numeric, 2)
--    AND fe.day_start = hwf.observation_date + interval '1 day';