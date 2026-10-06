"""Run the modular robot predictive-maintenance dashboard."""

import time
from collections import deque

import numpy as np
import pandas as pd
from IPython.display import clear_output, display

from anomaly_detection import AnomalyDetector
from data_loader import calculate_normal_behavior
from database import create_database_engine, load_telemetry_from_database, verify_telemetry_table
from StreamingSimulator import PostgreSQLStream
from visualization import render_maintenance_dashboard


TABLE_NAME = "robot_telemetry"
DATABASE_SCHEMA = "public"
ROBOT_NAME = "ROBOT 3"
REFRESH_SECONDS = 2
TIME_WINDOW_SECONDS = 90
MAX_STREAM_RECORDS = 120
MAX_EVENT_ROWS = 5
ANOMALY_CONFIRMATION_COUNT = 1
AXES = [f"Axis #{number}" for number in range(1, 9)]


def prepare_record(data_point):
    """Convert a streamed row into clean timestamp and numeric values."""
    record = data_point.to_dict() if hasattr(data_point, "to_dict") else dict(data_point)
    record["Time"] = pd.to_datetime(record.get("Time"), errors="coerce")
    if pd.isna(record["Time"]):
        record["Time"] = pd.Timestamp.now()
    for axis in AXES:
        record[axis] = pd.to_numeric(record.get(axis, np.nan), errors="coerce")
    return record


def run_dashboard(max_records=None):
    """Load the baseline and continuously display streamed database records."""
    engine = create_database_engine()
    columns = ["Time", *AXES]
    verify_telemetry_table(engine, TABLE_NAME, columns, DATABASE_SCHEMA)
    historical = load_telemetry_from_database(engine, TABLE_NAME, columns, DATABASE_SCHEMA)
    normal_behavior = calculate_normal_behavior(historical, AXES)
    display(normal_behavior)

    stream = PostgreSQLStream(engine, columns, table_name=TABLE_NAME, schema=DATABASE_SCHEMA)
    detector = AnomalyDetector(normal_behavior, AXES, ANOMALY_CONFIRMATION_COUNT)
    stream_history = deque(maxlen=MAX_STREAM_RECORDS)
    recent_events = deque(maxlen=MAX_EVENT_ROWS)
    previous_status = {axis: "NORMAL" for axis in AXES}
    processed = 0

    try:
        while max_records is None or processed < max_records:
            data_point = stream.nextDataPoint()
            if data_point is None:
                time.sleep(REFRESH_SECONDS)
                continue

            record = prepare_record(data_point)
            robot_state = detector.robot_state(record)
            active_events = []
            for axis in AXES:
                status, action = detector.classify_axis(axis, record[axis])
                record[f"{axis} Status"] = status
                if status in {"WARNING", "ANOMALY"}:
                    active_events.append({"Axis": axis, "Status": status, "Action": action})
                if status in {"WARNING", "ANOMALY"} and status != previous_status[axis]:
                    recent_events.appendleft({
                        "Time": record["Time"].strftime("%H:%M:%S"),
                        "Axis": axis,
                        "Current": round(float(record[axis]), 2),
                        "P95": round(float(normal_behavior.loc[axis, "P95"]), 2),
                        "Status": status,
                        "Action": action,
                    })
                previous_status[axis] = status

            stream_history.append(record)
            message, message_color = detector.dashboard_message(active_events, robot_state)
            history = pd.DataFrame(stream_history).sort_values("Time")
            window_start = record["Time"] - pd.Timedelta(seconds=TIME_WINDOW_SECONDS)
            visible = history[history["Time"].between(window_start, record["Time"])]

            clear_output(wait=True)
            render_maintenance_dashboard(
                visible, record, recent_events, AXES, robot_state, message,
                message_color, ROBOT_NAME, TIME_WINDOW_SECONDS,
            )
            processed += 1
            print(f"Streamed record: {processed}")
            print(f"Latest timestamp: {record['Time']}")
            print(f"Robot state: {robot_state}")
            print(f"Recommendation: {message}")
            time.sleep(REFRESH_SECONDS)
    finally:
        engine.dispose()


if __name__ == "__main__":
    run_dashboard()
