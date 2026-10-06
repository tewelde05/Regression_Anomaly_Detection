"""Centralized plots for regression evidence and detected events."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .regression_model import AxisRegressionResult
except ImportError:
    from regression_model import AxisRegressionResult


STATUS_COLORS = {"ALERT": "#f59e0b", "ERROR": "#ef4444"}


def plot_training_regression(
    result: AxisRegressionResult,
    output_path: str | Path | None = None,
):
    """Plot observed active readings and their fitted regression line."""
    frame = result.residuals.sort_values("Elapsed Seconds")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(frame["Time"], frame[result.axis], s=7, alpha=0.25, label="Observed")
    ax.plot(frame["Time"], frame["Predicted"], color="#2563eb", lw=2, label="Regression line")
    ax.set(title=f"{result.axis}: Training Data and Regression Line", xlabel="Time", ylabel="Current (A)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_residual_distribution(
    result: AxisRegressionResult,
    output_path: str | Path | None = None,
):
    """Plot residual evidence with the discovered MinC and MaxC."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(result.residuals["Residual"], bins=60, color="#64748b", alpha=0.8)
    ax.axvline(result.min_c, color=STATUS_COLORS["ALERT"], lw=2, label=f"MinC={result.min_c:.2f}")
    ax.axvline(result.max_c, color=STATUS_COLORS["ERROR"], lw=2, label=f"MaxC={result.max_c:.2f}")
    ax.set(title=f"{result.axis}: Residual Distribution", xlabel="Observed − Predicted Current (A)", ylabel="Frequency")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_test_events(
    detected: pd.DataFrame,
    axis: str,
    event_log: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
):
    """Overlay Alert and Error markers on synthetic observations."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(detected["Time"], detected[axis], color="#475569", lw=1, label="Observed test current")
    ax.plot(detected["Time"], detected[f"{axis} Predicted"], color="#2563eb", lw=2, label="Predicted current")
    for status in ["ALERT", "ERROR"]:
        points = detected[detected[f"{axis} Status"].eq(status)]
        if not points.empty:
            ax.scatter(points["Time"], points[axis], s=36, color=STATUS_COLORS[status], label=status, zorder=3)
    if event_log is not None and not event_log.empty:
        axis_events = event_log[event_log["Axis"].eq(axis)]
        for _, event in axis_events.iterrows():
            event_points = detected[
                detected["Time"].between(event["Start Time"], event["End Time"])
            ]
            if event_points.empty:
                continue
            peak_index = event_points[axis].idxmax()
            ax.annotate(
                f"{event['Event']}: {event['Duration Seconds']:.0f}s",
                (detected.loc[peak_index, "Time"], detected.loc[peak_index, axis]),
                xytext=(8, 10),
                textcoords="offset points",
                color=STATUS_COLORS[event["Event"]],
                weight="bold",
            )
    ax.set(title=f"{axis}: Synthetic Test Predictions and Events", xlabel="Time", ylabel="Current (A)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig
