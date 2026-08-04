-- Postgres schema for the disaster risk pipeline's feature store.
--
-- Two tables, deliberately kept separate rather than merged:
--   weather_features -> PREDICTORS (conditions before a fire, if any)
--   fire_events       -> LABELS (whether fire activity actually occurred)
--

CREATE DATABASE disaster_risk;  

\c disaster_risk

CREATE TABLE IF NOT EXISTS weather_features (
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
    CONSTRAINT weather_features_unique_window
        UNIQUE (window_start, latitude, longitude)
);

CREATE INDEX IF NOT EXISTS idx_weather_features_location
    ON weather_features (latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_weather_features_window_start
    ON weather_features (window_start);

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


