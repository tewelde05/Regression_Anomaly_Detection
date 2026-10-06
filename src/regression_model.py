"""Train one Time-to-Current linear regression model for each robot axis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .data_preparation import AXES, active_axis_rows
except ImportError:  # Allows direct execution from the src directory.
    from data_preparation import AXES, active_axis_rows


@dataclass
class AxisRegressionResult:
    """Model, residual evidence, and thresholds for one axis."""

    axis: str
    model: LinearRegression
    slope: float
    intercept: float
    r_squared: float
    mae: float
    rmse: float
    min_c: float
    max_c: float
    training_rows: int
    residuals: pd.DataFrame


def fit_axis_model(
    prepared_data: pd.DataFrame,
    axis: str,
    alert_percentile: float = 95.0,
    error_percentile: float = 99.0,
) -> AxisRegressionResult:
    """Fit one model using positive current readings and derive MinC/MaxC."""
    training = active_axis_rows(prepared_data, axis)
    x = training[["Elapsed Seconds"]]
    y = training[axis]

    model = LinearRegression()
    model.fit(x, y)
    predicted = model.predict(x)
    residual = y.to_numpy() - predicted
    positive_residuals = residual[residual > 0]
    if positive_residuals.size == 0:
        raise ValueError(f"{axis} has no positive residuals for threshold discovery.")

    min_c = float(np.percentile(positive_residuals, alert_percentile))
    max_c = float(np.percentile(positive_residuals, error_percentile))
    residual_frame = training[["Time", "Elapsed Seconds", axis]].copy()
    residual_frame["Predicted"] = predicted
    residual_frame["Residual"] = residual

    return AxisRegressionResult(
        axis=axis,
        model=model,
        slope=float(model.coef_[0]),
        intercept=float(model.intercept_),
        r_squared=float(r2_score(y, predicted)),
        mae=float(mean_absolute_error(y, predicted)),
        rmse=float(np.sqrt(mean_squared_error(y, predicted))),
        min_c=min_c,
        max_c=max_c,
        training_rows=len(training),
        residuals=residual_frame,
    )


def fit_all_axis_models(
    prepared_data: pd.DataFrame,
    axes: list[str] | None = None,
) -> dict[str, AxisRegressionResult]:
    """Train separate models for Axis #1 through Axis #8."""
    axes = axes or AXES
    return {axis: fit_axis_model(prepared_data, axis) for axis in axes}


def model_summary(
    results: dict[str, AxisRegressionResult],
) -> pd.DataFrame:
    """Create the grading-ready slope, intercept, metric, and threshold table."""
    rows = []
    for axis, result in results.items():
        rows.append(
            {
                "Axis": axis,
                "Training Rows": result.training_rows,
                "Slope": result.slope,
                "Intercept": result.intercept,
                "R Squared": result.r_squared,
                "MAE": result.mae,
                "RMSE": result.rmse,
                "MinC (P95 Positive Residual)": result.min_c,
                "MaxC (P99 Positive Residual)": result.max_c,
            }
        )
    return pd.DataFrame(rows)
