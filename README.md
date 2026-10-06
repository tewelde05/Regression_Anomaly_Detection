# Regression-Based Robot Current Anomaly Detection

# AI Assisted

## Project Summary

This project extends the CSCN8010 Data Stream Visualization Workshop into a predictive-maintenance workflow. Eight independent univariate linear regression models learn expected robot-axis current from elapsed time. Synthetic testing data is then streamed through the models. Sustained positive residuals generate Alert or Error events.

### Student - Id

- Senay Teweldebrhan (9120588)

## Data

The training file is `data/RMBR4-2_export_test.csv`. It contains 39,672 timestamped records. `Axis #1` through `Axis #8` contain current readings; the empty `Axis #9` through `Axis #14` columns are not used.

The raw zero readings are preserved because they represent stopped periods. They are excluded from model training because including 25,487 all-zero rows would pull the active-current regression lines downward. Each model is trained only on positive readings for its own axis.

## Workflow

1. Pull `Time` and `Axis #1`–`Axis #8` from the Neon PostgreSQL `robot_telemetry` table.
2. Parse timestamps, sort chronologically, and calculate elapsed seconds.
3. Label a row `STOPPED` when all eight readings are zero.
4. Train one `Time → Axis` linear regression model for every axis using its positive readings.
5. Calculate each training residual as `observed − predicted`.
6. Discover axis-specific thresholds from positive training residuals.
7. Generate synthetic testing data from the trained predictions and empirical residual distributions.
8. Standardize synthetic values with training means/standard deviations and Min-Max normalize them with training minima/maxima.
9. Replay the synthetic CSV chronologically into the PostgreSQL `synthetic_telemetry_stream` table.
10. Require deviations to persist continuously for `T = 6 seconds`.
11. Overlay Alert and Error markers on testing plots and save the event log.

## Synthetic Data Scaling

Scaling parameters come only from positive training readings pulled from the relational database. The original ampere columns remain available for regression predictions and operational thresholds.

```text
Z score = (synthetic current - training mean) / training standard deviation
Min-Max = (synthetic current - training minimum) / (training maximum - training minimum)
```

The generated CSV contains `Axis #n Z Score` and `Axis #n Normalized` fields for all eight axes. Normalized values are clipped to the 0–1 interval. `results/synthetic_scaling_metadata.csv` records the training statistics and formulas used.

## Regression and Alert Rules

For each axis:

```text
Predicted current = intercept + slope × elapsed seconds
Residual = observed current − predicted current
MinC = 95th percentile of positive training residuals
MaxC = 99th percentile of positive training residuals
T = 6 continuous seconds
```

Status rules:

- `STOPPED`: observed current equals zero.
- `NORMAL`: positive reading that does not satisfy a sustained threshold.
- `ALERT`: residual is at least MinC continuously for at least T seconds.
- `ERROR`: residual is at least MaxC continuously for at least T seconds.

Axis-specific thresholds are used because the eight axes have different current ranges. Error takes priority over Alert when both rules apply.

P95 leaves approximately 5% of positive training residuals at or above MinC, while P99 leaves approximately 1% at or above MaxC. `results/threshold_evidence.csv` records these counts and proportions for every axis.

The telemetry is sampled approximately every two seconds. The selected `T = 6 seconds` therefore requires about three consecutive abnormal readings. The sensitivity comparison in `results/duration_sensitivity.csv` shows that shorter windows create more events, while longer windows may delay or suppress early warnings.

## Project Structure

```text
data/
  RMBR4-2_export_test.csv
  synthetic_test_data.csv
  alert_error_log.csv
notebooks/
  Regression_Anomaly_Detection.ipynb
  DataStreamVisualization_workshop.ipynb
src/
  data_preparation.py
  regression_model.py
  synthetic_data.py
  synthetic_streaming.py
  alert_detection.py
  regression_visualization.py
  run_regression_pipeline.py
  database.py
  StreamingSimulator.py
  anomaly_detection.py
  visualization.py
  main.py
results/
  regression_model_summary.csv
  synthetic_scaling_metadata.csv
  threshold_evidence.csv
  duration_sensitivity.csv
  testing_predictions.csv
  axis_1_regression.png ... axis_8_regression.png
  axis_1_residuals.png ... axis_8_residuals.png
  axis_1_events.png ... axis_8_events.png
tests/
requirements.txt
.env.example
```

## Setup

1. Create and activate a virtual environment.

   Windows PowerShell:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the versioned dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Create `.env` from `.env.example` and set a valid Neon PostgreSQL connection URL.

   ```env
   DATABASE_URL=postgresql+psycopg2://username:password@host/database?sslmode=require
   ```

4. Confirm that the `robot_telemetry` table contains `Time` and `Axis #1` through `Axis #8`.

## Run the Required Database Workflow

From the project root:

```powershell
python -m src.run_regression_pipeline --source database --duration 6 --stream-to-database --stream-delay 2
```

The application pulls training rows from Neon before model fitting, meeting the cloud-database requirement.

For a faster classroom demonstration, use `--stream-delay 0.05`. The rows remain ordered by their telemetry timestamps.

For offline reproducibility only, the included CSV can be used:

```powershell
python -m src.run_regression_pipeline --source csv --duration 6
```

## Outputs

- `results/regression_model_summary.csv`: slope, intercept, model metrics, MinC, and MaxC for all axes.
- `data/synthetic_test_data.csv`: generated testing data.
- `results/synthetic_scaling_metadata.csv`: database-training statistics used for Z-score and Min-Max scaling.
- `results/threshold_evidence.csv`: P95/P99 counts and proportions.
- `results/duration_sensitivity.csv`: comparison of T = 2, 4, 6, and 10 seconds.
- `results/testing_predictions.csv`: observations, predictions, residuals, and statuses.
- `data/alert_error_log.csv`: structured Alert and Error events with their durations.
- `results/*.png`: regression, residual-distribution, and Alert/Error plots.

The included reproducibility run generated three sustained events: an Axis #2 Alert, an Axis #5 Error, and an Axis #8 Error. Results can differ if the random seed, thresholds, or duration are changed.

## Sample Results

### Axis #2 Regression Line

![Axis #2 regression](results/axis_2_regression.png)

### Axis #2 Residual Thresholds

![Axis #2 residual distribution](results/axis_2_residuals.png)

### Axis #2 Alert with Duration Annotation

![Axis #2 alert](results/axis_2_events.png)

## Security

The database URL is never stored in source code or notebooks. `.env` is excluded by `.gitignore`.

## Tests

```powershell
pytest -q
```

The tests verify stopped-row handling, positive-reading training selection, plotting behavior, and sustained event detection.

## Results/ Screenshot

### Regression and residual analysis

![Axis regression analysis](results/axis_1_regression.png)

### Alert detected

![Alert event](./results/axis_2_events.png)

### Error detected

![Error event](results/axis_5_events.png)
