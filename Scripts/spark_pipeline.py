# ============================================================
# spark_pipeline.py
# AWS EMR Spark Yellow Taxi Big Data Analytics Pipeline
# With Stage Execution Time Tracking
# ============================================================


from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import boto3
import pandas as pd
import io
import time



# ============================================================
# Spark Session
# ============================================================

spark = (
    SparkSession.builder
    .appName("YellowTaxi_BigData_Analytics")
    .getOrCreate()
)



# ============================================================
# Configuration
# ============================================================

input_glob = (
    "s3://bda-yellowtaxi/dataset/*.parquet"
)

zones_path = (
    "s3://bda-yellowtaxi/dataset/taxi_zone_lookup.csv"
)


BUCKET = "bda-yellowtaxi"

OUTPUT_PREFIX = "output/"

OUTPUT_PATH = (
    f"s3://{BUCKET}/{OUTPUT_PREFIX}"
)



# ============================================================
# Runtime Tracking
# ============================================================

stage_times = []

pipeline_start = time.time()



# ============================================================
# Stage 1 - Load Dataset
# ============================================================

t0 = time.time()


df = spark.read.parquet(input_glob)

zones = spark.read.csv(
    zones_path,
    header=True,
    inferSchema=True
)


row_count_raw = df.count()


stage_times.append(
    {
        "stage":"Load Data",
        "execution_time_sec":
            round(time.time()-t0,2)
    }
)


print(
    f"Raw row count: {row_count_raw:,}"
)

print(
    f"Load Data Time: {stage_times[-1]['execution_time_sec']} sec"
)



# ============================================================
# Stage 2 - Cleaning
# ============================================================

t0 = time.time()


df = df.withColumn(
    "trip_duration_min",
    (
        F.unix_timestamp(
            "tpep_dropoff_datetime"
        )
        -
        F.unix_timestamp(
            "tpep_pickup_datetime"
        )
    ) / 60.0
)



df_clean = df.filter(

    (F.col("fare_amount") > 0)
    &
    (F.col("trip_distance") > 0)
    &
    (F.col("passenger_count") > 0)
    &
    (F.col("trip_duration_min") > 0.5)
    &
    (F.col("trip_duration_min") < 180)
    &
    F.col("PULocationID").isNotNull()
    &
    F.col("DOLocationID").isNotNull()

)



df_clean = (
    df_clean
    .withColumn(
        "pickup_hour",
        F.hour(
            "tpep_pickup_datetime"
        )
    )
    .withColumn(
        "pickup_dow",
        F.date_format(
            "tpep_pickup_datetime",
            "EEEE"
        )
    )
    .cache()
)



row_count_clean = df_clean.count()



stage_times.append(
    {
        "stage":"Data Cleaning",
        "execution_time_sec":
            round(time.time()-t0,2)
    }
)



print(
    f"Clean row count: {row_count_clean:,}"
)

print(
    f"Cleaning Time: {stage_times[-1]['execution_time_sec']} sec"
)





# ============================================================
# Stage 3 - Pickup Zone Join
# ============================================================

t0 = time.time()



zones_pu = (
    zones
    .select(
        "LocationID",
        "Borough",
        "Zone"
    )
    .withColumnRenamed(
        "LocationID",
        "PULocationID"
    )
    .withColumnRenamed(
        "Borough",
        "PU_Borough"
    )
    .withColumnRenamed(
        "Zone",
        "PU_Zone"
    )
)



df_joined = (
    df_clean
    .join(
        F.broadcast(zones_pu),
        on="PULocationID",
        how="left"
    )
    .withColumn(
        "tip_pct",
        F.col("tip_amount")
        /
        F.col("fare_amount")
        *
        100
    )
    .cache()
)



df_joined.count()



stage_times.append(
    {
        "stage":"Join Pickup Zone",
        "execution_time_sec":
            round(time.time()-t0,2)
    }
)



print(
    f"Join Pickup Zone Time: {stage_times[-1]['execution_time_sec']} sec"
)



# ============================================================
# Stage 4 - Aggregations
# ============================================================

t0=time.time()



agg_zone_hour = (

    df_joined
    .groupBy(
        "PU_Borough",
        "PU_Zone",
        "pickup_hour"
    )

    .agg(

        F.count("fare_amount")
        .alias("trip_count"),

        F.avg("fare_amount")
        .alias("avg_fare"),

        F.avg("trip_distance")
        .alias("avg_distance"),

        F.avg("trip_duration_min")
        .alias("avg_duration_min")

    )

)



stage_times.append(
{
"stage":"Aggregation Zone Hour",
"execution_time_sec":
round(time.time()-t0,2)
}
)


print(
"Aggregation Zone Hour completed"
)


# ============================================================
# Stage 4b - Aggregation Day of Week
# ============================================================

t0 = time.time()


agg_dow = (

    df_joined

    .groupBy(
        "pickup_dow"
    )

    .agg(

        F.count("fare_amount")
        .alias("trip_count"),

        F.sum("total_amount")
        .alias("total_revenue"),

        F.avg("tip_pct")
        .alias("avg_tip_pct")

    )

)



stage_times.append(
    {
        "stage":"Aggregation Day Of Week",
        "execution_time_sec":
            round(time.time()-t0,2)
    }
)


print(
    f"Aggregation Day Of Week Time: "
    f"{stage_times[-1]['execution_time_sec']} sec"
)



# ============================================================
# Stage 4c - Aggregation Payment Type
# ============================================================

t0 = time.time()


payment_label = (

    F.when(
        F.col("payment_type")==1,
        "Credit Card"
    )

    .when(
        F.col("payment_type")==2,
        "Cash"
    )

    .when(
        F.col("payment_type")==3,
        "No Charge"
    )

    .when(
        F.col("payment_type")==4,
        "Dispute"
    )

    .otherwise(
        "Unknown"
    )

)



df_joined = df_joined.withColumn(
    "payment_type_label",
    payment_label
)



agg_payment = (

    df_joined

    .groupBy(
        "payment_type_label"
    )

    .agg(

        F.count("fare_amount")
        .alias("trip_count"),

        F.avg("fare_amount")
        .alias("avg_fare"),

        F.avg("tip_pct")
        .alias("avg_tip_pct"),

        F.sum("total_amount")
        .alias("total_revenue")

    )

)



stage_times.append(
    {
        "stage":"Aggregation Payment Type",
        "execution_time_sec":
            round(time.time()-t0,2)
    }
)



print(
    f"Aggregation Payment Type Time: "
    f"{stage_times[-1]['execution_time_sec']} sec"
)




# ============================================================
# Stage 4d - Borough Flow Analysis
# ============================================================

t0 = time.time()



zones_do = (

    zones

    .select(
        "LocationID",
        "Borough",
        "Zone"
    )

    .withColumnRenamed(
        "LocationID",
        "DOLocationID"
    )

    .withColumnRenamed(
        "Borough",
        "DO_Borough"
    )

    .withColumnRenamed(
        "Zone",
        "DO_Zone"
    )

)



df_flow = (

    df_joined

    .join(
        F.broadcast(zones_do),
        on="DOLocationID",
        how="left"
    )

)



agg_borough_flow = (

    df_flow

    .groupBy(
        "PU_Borough",
        "DO_Borough"
    )

    .agg(

        F.count("fare_amount")
        .alias("trip_count"),

        F.avg("fare_amount")
        .alias("avg_fare"),

        F.avg("trip_distance")
        .alias("avg_distance")

    )

)



stage_times.append(
    {
        "stage":"Join Dropoff Zone + Borough Flow",
        "execution_time_sec":
            round(time.time()-t0,2)
    }
)



print(
    f"Borough Flow Time: "
    f"{stage_times[-1]['execution_time_sec']} sec"
)




# ============================================================
# Stage 5 - Machine Learning
# ============================================================

t0 = time.time()



ml_df = (

    df_clean

    .select(

        "fare_amount",
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "PULocationID"

    )

    .dropna()

)



ml_df = ml_df.withColumn(

    "PULocationID",

    F.col("PULocationID")
    .cast(DoubleType())

)



ohe = OneHotEncoder(

    inputCol="PULocationID",

    outputCol="PULocationID_ohe"

)



assembler = VectorAssembler(

    inputCols=[

        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "PULocationID_ohe"

    ],

    outputCol="features"

)



lr = LinearRegression(

    featuresCol="features",

    labelCol="fare_amount"

)



train_df, test_df = ml_df.randomSplit(

    [0.8,0.2],

    seed=42

)



ohe_model = ohe.fit(train_df)



train_vec = assembler.transform(

    ohe_model.transform(train_df)

)



test_vec = assembler.transform(

    ohe_model.transform(test_df)

)



lr_model = lr.fit(train_vec)



stage_times.append(

    {
        "stage":"ML Training",
        "execution_time_sec":
            round(time.time()-t0,2)
    }

)



print(
    f"ML Training Time: "
    f"{stage_times[-1]['execution_time_sec']} sec"
)



# ------------------------------------------------------------
# ML Evaluation
# ------------------------------------------------------------


t0 = time.time()



predictions = lr_model.transform(test_vec)



rmse = RegressionEvaluator(

    labelCol="fare_amount",

    predictionCol="prediction",

    metricName="rmse"

).evaluate(predictions)



r2 = RegressionEvaluator(

    labelCol="fare_amount",

    predictionCol="prediction",

    metricName="r2"

).evaluate(predictions)



stage_times.append(

    {
        "stage":"ML Prediction Evaluation",

        "execution_time_sec":
            round(time.time()-t0,2)
    }

)



print(
    f"RMSE: {rmse:.3f}"
)

print(
    f"R2: {r2:.3f}"
)




# ============================================================
# Stage 6 - Save Parquet Outputs
# ============================================================


print("Saving parquet files...")



agg_zone_hour.write.mode(
    "overwrite"
).parquet(
    OUTPUT_PATH+"agg_zone_hour.parquet"
)



agg_dow.write.mode(
    "overwrite"
).parquet(
    OUTPUT_PATH+"agg_day_of_week.parquet"
)



agg_payment.write.mode(
    "overwrite"
).parquet(
    OUTPUT_PATH+"agg_payment_type.parquet"
)



agg_borough_flow.write.mode(
    "overwrite"
).parquet(
    OUTPUT_PATH+"agg_borough_flow.parquet"
)



print("Parquet saved.")




# ============================================================
# Stage 7 - Save Metrics CSV
# ============================================================


total_time = round(

    time.time()-pipeline_start,

    2

)



stage_times.append(

    {
        "stage":"Total Pipeline",

        "execution_time_sec":
            total_time
    }

)



metrics_df = pd.DataFrame(stage_times)



metrics_df["platform"] = (
    "AWS Spark (EMR)"
)



metrics_df["raw_row_count"] = (
    row_count_raw
)



metrics_df["clean_row_count"] = (
    row_count_clean
)



metrics_df["rmse"] = (
    round(rmse,3)
)



metrics_df["r2"] = (
    round(r2,3)
)



metrics_df = metrics_df[

    [

        "platform",

        "stage",

        "execution_time_sec",

        "raw_row_count",

        "clean_row_count",

        "rmse",

        "r2"

    ]

]



print(metrics_df)



csv_buffer = io.StringIO()



metrics_df.to_csv(

    csv_buffer,

    index=False

)



boto3.client("s3").put_object(

    Bucket=BUCKET,

    Key="output/metrics_spark.csv",

    Body=csv_buffer.getvalue()

)



print(
    "metrics_spark.csv saved."
)



print("==============================")
print("Spark Pipeline Completed")
print("==============================")


spark.stop()
