"""Apply continuous-time Alert and Error rules to synthetic telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .data_preparation import AXES
    from .regression_model import AxisRegressionResult
except ImportError:
    from data_preparation import AXES
    from regression_model import AxisRegressionResult


def _qualifying_runs(
    times: pd.Series,
    condition: pd.Series,
    minimum_seconds: float,
) -> list[tuple[pd.Index, float]]:
    """Return continuous true runs meeting the required elapsed duration."""
    times = pd.to_datetime(times, utc=True)
    positive_intervals = times.diff().dt.total_seconds().dropna()
    sample_interval = float(positive_intervals.median()) if not positive_intervals.empty else 0.0
    gap_limit = sample_interval * 1.5 if sample_interval > 0 else np.inf
    breaks = (~condition) | times.diff().dt.total_seconds().gt(gap_limit)
    groups = breaks.cumsum()
    runs = []
    for _, indexes in condition[condition].groupby(groups[condition]).groups.items():
        indexes = pd.Index(indexes)
        duration = (
            times.loc[indexes[-1]] - times.loc[indexes[0]]
        ).total_seconds() + sample_interval
        if duration >= minimum_seconds:
            runs.append((indexes, float(duration)))
    return runs


def detect_alerts_and_errors(
    testing_data: pd.DataFrame,
    results: dict[str, AxisRegressionResult],
    minimum_duration_seconds: float = 6.0,
    axes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict expected current and log sustained positive deviations."""
    axes = axes or AXES
    detected = testing_data.copy()
    events: list[dict] = []

    for axis in axes:
        result = results[axis]
        predicted = result.model.predict(detected[["Elapsed Seconds"]])
        residual = detected[axis].to_numpy() - predicted
        stopped = detected[axis].eq(0)
        detected[f"{axis} Predicted"] = predicted
        detected[f"{axis} Residual"] = residual
        detected[f"{axis} Status"] = np.where(stopped, "STOPPED", "NORMAL")

        error_condition = pd.Series(
            (~stopped) & (residual >= result.max_c), index=detected.index
        )
        error_indexes: set[int] = set()
        for indexes, duration in _qualifying_runs(
            detected["Time"], error_condition, minimum_duration_seconds
        ):
            detected.loc[indexes, f"{axis} Status"] = "ERROR"
            error_indexes.update(indexes.tolist())
            events.append(
                {
                    "Axis": axis,
                    "Event": "ERROR",
                    "Start Time": detected.loc[indexes[0], "Time"],
                    "End Time": detected.loc[indexes[-1], "Time"],
                    "Duration Seconds": duration,
                    "Threshold": result.max_c,
                    "Maximum Residual": float(
                        detected.loc[indexes, f"{axis} Residual"].max()
                    ),
                }
            )

        alert_condition = pd.Series(
            (~stopped) & (residual >= result.min_c), index=detected.index
        )
        for indexes, duration in _qualifying_runs(
            detected["Time"], alert_condition, minimum_duration_seconds
        ):
            alert_only = indexes.difference(pd.Index(error_indexes))
            if alert_only.empty:
                continue
            detected.loc[alert_only, f"{axis} Status"] = "ALERT"
            events.append(
                {
                    "Axis": axis,
                    "Event": "ALERT",
                    "Start Time": detected.loc[alert_only[0], "Time"],
                    "End Time": detected.loc[alert_only[-1], "Time"],
                    "Duration Seconds": duration,
                    "Threshold": result.min_c,
                    "Maximum Residual": float(
                        detected.loc[alert_only, f"{axis} Residual"].max()
                    ),
                }
            )

    event_log = pd.DataFrame(events)
    if not event_log.empty:
        event_log = event_log.sort_values(["Start Time", "Axis"]).reset_index(drop=True)
    return detected, event_log
