# IST3134-BIG-DATA-ANALYTICS-IN-THE-CLOUD
## Group Assignment — NYC Yellow Taxi: AWS Spark vs Python

Comparing AWS Big Data platform (S3 + EMR/Spark) against a local Python
(pandas + scikit-learn) pipeline on the same NYC Yellow Taxi dataset
(Jan–Mar 2026), across runtime, memory, and model accuracy.

## Problem Statement

How do fare amounts, trip duration, and demand vary across NYC pickup
zones and time of day, and can we predict trip fare from ride
characteristics at scale? We compare an AWS Spark (EMR) implementation
against an equivalent local Python implementation to demonstrate the
practical advantages/tradeoffs of Big Data Analytics platforms.

## Repo Structure

```
.
├── aws_spark_pipeline.py       # Spark/PySpark pipeline — runs on EMR
├── python_pandas_pipeline.py   # pandas + scikit-learn pipeline — runs locally
├── compare_results.py          # combines both metrics_*.json into a chart + table
├── requirements.txt            # Python dependencies (local side only)
├── data/                       # (not committed — see Dataset section)
└── output/                     # aggregation results, generated on run
```

## Dataset

- Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Files used: `yellow_tripdata_2026-01.parquet`, `yellow_tripdata_2026-02.parquet`,
  `yellow_tripdata_2026-03.parquet`, `taxi_zone_lookup.csv`
- Size: ~11M rows combined, 181.9MB compressed / ~1.89GB in-memory
- Raw files are **not committed to GitHub** (too large) — download them
  from the TLC link above into a local `data/` folder before running.

## Setup

### Python (local) side

```bash
pip install -r requirements.txt
```

Place the 3 monthly Parquet files + `taxi_zone_lookup.csv` into `./data/`.

Run:
```bash
python python_pandas_pipeline.py \
  --input ./data/ \
  --zones ./data/taxi_zone_lookup.csv \
  --output ./output/
```

This produces `metrics_python.json` and Parquet aggregation outputs in `./output/`.

### AWS (EMR/Spark) side

1. Upload the Parquet files + zone lookup CSV to S3:
   ```bash
   aws s3 cp ./data/ s3://your-bucket-name/raw/ --recursive
   ```
2. Launch a small EMR cluster (1 primary + 2 core nodes, Spark application enabled).
3. Submit the job (via SSH to the primary node, or an EMR Notebook):
   ```bash
   spark-submit aws_spark_pipeline.py \
     --input s3://your-bucket-name/raw/ \
     --zones s3://your-bucket-name/raw/taxi_zone_lookup.csv \
     --output s3://your-bucket-name/processed/
   ```
4. Download `metrics_spark.json` from the cluster (or write it to S3 and pull it down)
   back into this repo folder, so it sits next to `metrics_python.json`.
5. **Terminate the cluster** once done to avoid ongoing charges.

### Comparing results

Once both `metrics_spark.json` and `metrics_python.json` exist in the
project root:

```bash
python compare_results.py
```

Outputs a stage-by-stage runtime table in the terminal plus
`runtime_comparison.png` — use this chart directly in the report's
output analysis / comparison section.

## Notes

- Both pipelines use identical cleaning thresholds and model features,
  so the comparison is apples-to-apples (see Stage 2 filters and Stage
  5 feature list in each script).
- `python_pandas_pipeline.py` tracks peak memory per stage via
  `tracemalloc`, in addition to runtime — useful for the "where pandas
  struggles at scale" argument in the report.
- If the lecturer asks for a larger dataset, add more months to
  `data/` / the S3 `raw/` prefix — both scripts already read
  everything in the folder, no code changes needed.

## Authors

Yew Jia Wen (22070817) — Python/pandas pipeline, benchmarking, comparison analysis

Tan Zhong Qing (22031892) — AWS S3/EMR setup, Spark pipeline, MapReduce/Spark analysis
