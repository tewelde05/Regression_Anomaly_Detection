"""Predictive-maintenance classification for robot telemetry."""

import numpy as np
import pandas as pd


class AnomalyDetector:
    """Classify each axis using historical P95 and P99 thresholds."""

    def __init__(self, normal_behavior, axes, confirmation_count=1):
        self.normal_behavior = normal_behavior
        self.axes = axes
        self.confirmation_count = confirmation_count
        self.anomaly_counts = {axis: 0 for axis in axes}

    def classify_axis(self, axis, current):
        if pd.isna(current) or current <= 0:
            self.anomaly_counts[axis] = 0
            return "STOPPED", "NO ACTION"

        p95 = self.normal_behavior.loc[axis, "P95"]
        p99 = self.normal_behavior.loc[axis, "P99"]
        if pd.isna(p95) or pd.isna(p99):
            self.anomaly_counts[axis] = 0
            return "UNKNOWN", "CHECK HISTORICAL DATA"

        if current > p99:
            self.anomaly_counts[axis] += 1
            if self.anomaly_counts[axis] >= self.confirmation_count:
                return "ANOMALY", "MAINTENANCE REQUIRED"
            return "WARNING", "MONITOR CURRENT SPIKE"

        self.anomaly_counts[axis] = 0
        if current > p95:
            return "WARNING", "CHECK OR RESET"
        return "NORMAL", "CONTINUE MONITORING"

    def robot_state(self, record):
        currents = pd.to_numeric(
            pd.Series([record.get(axis, np.nan) for axis in self.axes]),
            errors="coerce",
        )
        return "STOPPED" if currents.fillna(0).le(0).all() else "RUNNING"

    @staticmethod
    def dashboard_message(events, robot_state):
        for status, message, color in (
            ("ANOMALY", "ANOMALY: MAINTENANCE REQUIRED", "red"),
            ("WARNING", "WARNING: CHECK OR RESET", "darkorange"),
        ):
            affected = sorted({e["Axis"] for e in events if e["Status"] == status})
            if affected:
                return f"{message}: {', '.join(affected)}", color

        if robot_state == "STOPPED":
            return "NO ACTIVE ANOMALY: ROBOT STOPPED", "green"
        return "NORMAL OPERATION", "green"
