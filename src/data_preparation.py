"""Prepare robot telemetry for regression-based anomaly detection."""

from __future__ import annotations

import pandas as pd


AXES = [f"Axis #{number}" for number in range(1, 9)]


def prepare_telemetry(
    dataframe: pd.DataFrame,
    time_column: str = "Time",
    axes: list[str] | None = None,
) -> pd.DataFrame:
    """Return chronological telemetry with numeric axes and elapsed seconds.

    Raw zero readings are retained. A row is STOPPED only when every selected
    axis is zero; otherwise it is ACTIVE.
    """
    axes = axes or AXES
    required = [time_column, *axes]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = dataframe[required].copy()
    prepared[time_column] = pd.to_datetime(
        prepared[time_column], errors="coerce", utc=True
    )
    prepared[axes] = prepared[axes].apply(pd.to_numeric, errors="coerce")
    prepared = prepared.dropna(subset=[time_column]).sort_values(time_column)
    prepared = prepared.drop_duplicates(subset=[time_column]).reset_index(drop=True)

    start_time = prepared[time_column].min()
    prepared["Elapsed Seconds"] = (
        prepared[time_column] - start_time
    ).dt.total_seconds()
    prepared["Operating State"] = (
        prepared[axes].fillna(0).ne(0).any(axis=1).map({True: "ACTIVE", False: "STOPPED"})
    )
    return prepared


def active_axis_rows(
    dataframe: pd.DataFrame,
    axis: str,
) -> pd.DataFrame:
    """Select positive readings for one axis for model training."""
    if axis not in dataframe.columns:
        raise ValueError(f"Unknown axis column: {axis}")
    active = dataframe.loc[dataframe[axis].gt(0)].copy()
    if active.empty:
        raise ValueError(f"{axis} does not contain positive training readings.")
    return active


def zero_reading_summary(
    dataframe: pd.DataFrame,
    axes: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize zero and positive readings without changing the raw data."""
    axes = axes or AXES
    rows = []
    for axis in axes:
        values = pd.to_numeric(dataframe[axis], errors="coerce")
        rows.append(
            {
                "Axis": axis,
                "Zero Readings": int(values.eq(0).sum()),
                "Positive Readings": int(values.gt(0).sum()),
                "Missing Readings": int(values.isna().sum()),
            }
        )
    return pd.DataFrame(rows)
