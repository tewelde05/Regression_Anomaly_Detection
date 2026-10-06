"""Generate reproducible testing telemetry from training residual metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .data_preparation import AXES
    from .regression_model import AxisRegressionResult
except ImportError:
    from data_preparation import AXES
    from regression_model import AxisRegressionResult


def generate_synthetic_test_data(
    training_data: pd.DataFrame,
    results: dict[str, AxisRegressionResult],
    rows: int = 360,
    interval_seconds: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create testing rows and inject sustained Alert and Error examples.

    Normal variation is sampled from each fitted model's empirical residuals,
    so the synthetic data retains training-data behavior without copying rows.
    """
    rng = np.random.default_rng(random_state)
    start = training_data["Time"].max() + pd.Timedelta(seconds=interval_seconds)
    times = pd.date_range(start, periods=rows, freq=f"{interval_seconds}s")
    elapsed = (
        times - training_data["Time"].min()
    ).total_seconds().to_numpy()
    synthetic = pd.DataFrame({"Time": times, "Elapsed Seconds": elapsed})

    for axis in AXES:
        result = results[axis]
        expected = result.model.predict(
            pd.DataFrame({"Elapsed Seconds": elapsed})
        )
        empirical = result.residuals["Residual"].to_numpy()
        central = empirical[
            (empirical >= np.percentile(empirical, 10))
            & (empirical <= np.percentile(empirical, 90))
        ]
        noise = rng.choice(central, size=rows, replace=True)
        synthetic[axis] = np.maximum(expected + noise, 0.01)

    # A short initial stopped period demonstrates that zeros are not anomalies.
    synthetic.loc[:9, AXES] = 0.0

    # Known sustained deviations ensure the rule can be demonstrated and tested.
    synthetic.loc[80:89, "Axis #2"] += results["Axis #2"].min_c * 1.20
    synthetic.loc[170:181, "Axis #5"] += results["Axis #5"].max_c * 1.20
    synthetic.loc[260:268, "Axis #8"] += results["Axis #8"].min_c * 1.25
    synthetic["Trait"] = "current"

    # Create rubric-required scaling fields from database training statistics.
    # Original ampere columns are preserved for predictions and alert rules.
    scaled_columns = []
    for axis in AXES:
        training_values = results[axis].residuals[axis]
        training_mean = float(training_values.mean())
        training_std = float(training_values.std(ddof=0))
        training_min = float(training_values.min())
        training_max = float(training_values.max())

        z_column = f"{axis} Z Score"
        normalized_column = f"{axis} Normalized"
        synthetic[z_column] = (
            synthetic[axis] - training_mean
        ) / training_std
        synthetic[normalized_column] = (
            (synthetic[axis] - training_min)
            / (training_max - training_min)
        ).clip(0, 1)
        scaled_columns.extend([z_column, normalized_column])

    return synthetic[
        ["Trait", "Time", "Elapsed Seconds", *AXES, *scaled_columns]
    ]


def scaling_metadata(
    results: dict[str, AxisRegressionResult],
) -> pd.DataFrame:
    """Document database-training statistics used to scale synthetic data."""
    rows = []
    for axis in AXES:
        values = results[axis].residuals[axis]
        rows.append(
            {
                "Axis": axis,
                "Training Mean": float(values.mean()),
                "Training Standard Deviation": float(values.std(ddof=0)),
                "Training Minimum": float(values.min()),
                "Training Maximum": float(values.max()),
                "Standardization Formula": "(x - training mean) / training std",
                "Normalization Formula": "(x - training min) / (training max - training min)",
            }
        )
    return pd.DataFrame(rows)
