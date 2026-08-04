"""
Spark Structured Streaming job: consumes raw weather + fire events from
Kafka, engineers risk-relevant features, and writes results to Postgres.

Two independent streaming queries run side by side:
    - weather-raw -> weather_features table  (PREDICTORS: conditions over time)
    - firms-raw   -> fire_events table        (LABELS: did a fire actually occur)


Run with (note: you MUST use spark-submit with the Kafka + Postgres
connector packages, not `python spark_transform.py` directly):

    spark-submit \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,\
org.postgresql:postgresql:42.7.3 \
        spark_transform.py

Setup:
    pip install pyspark
    A running Kafka broker at KAFKA_BOOTSTRAP_SERVERS
    A running Postgres instance with the tables from schema.sql applied
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, col, count, from_json, regexp_extract, to_timestamp, window,
)
from pyspark.sql.types import (
    DoubleType, StringType, StructField, StructType, TimestampType,
)

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

POSTGRES_URL = os.environ.get("POSTGRES_URL", "jdbc:postgresql://localhost:5432/disaster_risk")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

WEATHER_TABLE = "weather_features"
FIRE_TABLE = "fire_events"


CHECKPOINT_BASE = os.environ.get("CHECKPOINT_BASE", "/tmp/spark-checkpoints")
WEATHER_CHECKPOINT_DIR = f"{CHECKPOINT_BASE}/weather-features"
FIRE_CHECKPOINT_DIR = f"{CHECKPOINT_BASE}/fire-events"


WEATHER_SCHEMA = StructType([
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("temperature", DoubleType()),
    StructField("relativeHumidity", DoubleType()),
    StructField("windSpeed", StringType()),  # NWS returns this as a string like "10 mph"
    StructField("fetched_at", TimestampType()),
])

FIRMS_SCHEMA = StructType([
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("confidence", StringType()),
    StructField("acq_date", StringType()),
    StructField("acq_time", StringType()),
])


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("disaster-risk-feature-engineering")
        .getOrCreate()
    )


def read_kafka_stream(spark: SparkSession, topic: str):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )


def parse_json_stream(raw_df, schema):
    """Kafka gives us key/value as bytes; extract and parse the JSON value."""
    return (
        raw_df
        .selectExpr("CAST(value AS STRING) as json_value", "timestamp as kafka_ts")
        .select(from_json(col("json_value"), schema).alias("data"), col("kafka_ts"))
        .select("data.*", "kafka_ts")
    )


def parse_windspeed_mph(windspeed_str_col):
    """NWS returns windSpeed as a free-text string ('10 mph', '5 to 10 mph').
    Extract the first number as a rough numeric feature."""
    return regexp_extract(windspeed_str_col, r"(\d+)", 1).cast(DoubleType())


def engineer_weather_features(parsed_df):
    """Rolling-window aggregates over a 6-hour tumbling window per location.
    These are the actual model inputs — raw point-in-time readings are
    noisy, but trends (is humidity dropping, is wind picking up) are what
    correlate with fire risk."""
    df = parsed_df.withColumn("wind_mph", parse_windspeed_mph(col("windSpeed")))

    windowed = (
        df
        .withWatermark("fetched_at", "1 hour")  # bounds how late data can arrive before being dropped
        .groupBy(
            window(col("fetched_at"), "6 hours"),
            col("latitude"),
            col("longitude"),
        )
        .agg(
            avg("temperature").alias("avg_temp"),
            avg("relativeHumidity").alias("avg_humidity"),
            avg("wind_mph").alias("avg_wind_mph"),
        )
    )

    return windowed.select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        "latitude", "longitude",
        "avg_temp", "avg_humidity", "avg_wind_mph",
    )


def engineer_fire_events(parsed_df):
    """Aggregate fire detection counts per grid cell per day. This is the
    LABEL side of the pipeline — 'did fire activity occur here' — not a
    predictive feature. """
    df = parsed_df.withColumn("detected_at", to_timestamp(col("acq_date")))

    windowed = (
        df
        .withWatermark("detected_at", "1 day")
        .groupBy(
            window(col("detected_at"), "1 day"),
            col("latitude"),
            col("longitude"),
        )
        .agg(count("*").alias("fire_detection_count"))
    )

    return windowed.select(
        col("window.start").alias("day_start"),
        col("window.end").alias("day_end"),
        "latitude", "longitude",
        "fire_detection_count",
    )


def make_postgres_writer(table_name: str):
    """Returns a foreachBatch-compatible function bound to a specific table.
    foreachBatch is the standard way to sink a stream to a JDBC target —
    Structured Streaming doesn't have a native streaming Postgres writer,
    so each micro-batch gets written via the regular batch JDBC path."""

    def write_batch(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            return

        (
            batch_df.write
            .format("jdbc")
            .option("url", POSTGRES_URL)
            .option("dbtable", table_name)
            .option("user", POSTGRES_USER)
            .option("password", POSTGRES_PASSWORD)
            .option("driver", "org.postgresql.Driver")
            .mode("append")
            .save()
        )
        print(f"[{table_name} batch {batch_id}] wrote {batch_df.count()} rows")

    return write_batch


def run():
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # --- Weather stream (features) ---
    raw_weather = read_kafka_stream(spark, "weather-raw")
    parsed_weather = parse_json_stream(raw_weather, WEATHER_SCHEMA)
    weather_features = engineer_weather_features(parsed_weather)

    weather_query = (
        weather_features.writeStream
        .foreachBatch(make_postgres_writer(WEATHER_TABLE))
        .outputMode("update")
        .option("checkpointLocation", WEATHER_CHECKPOINT_DIR)
        .trigger(processingTime="5 minutes")
        .start()
    )

    # --- FIRMS stream (labels) ---
    raw_firms = read_kafka_stream(spark, "firms-raw")
    parsed_firms = parse_json_stream(raw_firms, FIRMS_SCHEMA)
    fire_events = engineer_fire_events(parsed_firms)

    fire_query = (
        fire_events.writeStream
        .foreachBatch(make_postgres_writer(FIRE_TABLE))
        .outputMode("update")
        .option("checkpointLocation", FIRE_CHECKPOINT_DIR)
        .trigger(processingTime="1 hour")  # fire data updates less frequently than weather
        .start()
    )

    # Both queries run concurrently; block until either terminates (e.g. on error or shutdown).
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    run()
