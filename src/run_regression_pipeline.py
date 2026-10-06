"""Run the complete regression-based predictive-maintenance workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .alert_detection import detect_alerts_and_errors
    from .data_preparation import AXES, prepare_telemetry
    from .regression_model import fit_all_axis_models, model_summary
    from .regression_visualization import (
        plot_residual_distribution,
        plot_test_events,
        plot_training_regression,
    )
    from .synthetic_data import generate_synthetic_test_data, scaling_metadata
except ImportError:
    from alert_detection import detect_alerts_and_errors
    from data_preparation import AXES, prepare_telemetry
    from regression_model import fit_all_axis_models, model_summary
    from regression_visualization import (
        plot_residual_distribution,
        plot_test_events,
        plot_training_regression,
    )
    from synthetic_data import generate_synthetic_test_data, scaling_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_training_data(source: str) -> pd.DataFrame:
    """Load training telemetry from Neon or from the reproducibility CSV."""
    if source == "database":
        try:
            from .database import create_database_engine, load_telemetry_from_database
        except ImportError:
            from database import create_database_engine, load_telemetry_from_database
        engine = create_database_engine()
        try:
            return load_telemetry_from_database(
                engine,
                table_name="robot_telemetry",
                columns=["Time", *AXES],
                schema="public",
            )
        finally:
            engine.dispose()
    return pd.read_csv(DATA_DIR / "RMBR4-2_export_test.csv")


def run(
    source: str = "database",
    minimum_duration_seconds: float = 6.0,
    stream_to_database: bool = False,
    stream_delay_seconds: float = 2.0,
):
    """Train models, generate test data, detect events, and save evidence."""
    RESULTS_DIR.mkdir(exist_ok=True)
    training = prepare_telemetry(load_training_data(source))
    models = fit_all_axis_models(training)
    summary = model_summary(models)
    summary.to_csv(RESULTS_DIR / "regression_model_summary.csv", index=False)

    testing = generate_synthetic_test_data(training, models)
    testing.to_csv(DATA_DIR / "synthetic_test_data.csv", index=False)
    scaling_metadata(models).to_csv(
        RESULTS_DIR / "synthetic_scaling_metadata.csv", index=False
    )
    detected, events = detect_alerts_and_errors(
        testing,
        models,
        minimum_duration_seconds=minimum_duration_seconds,
    )
    detected.to_csv(RESULTS_DIR / "testing_predictions.csv", index=False)
    events.to_csv(DATA_DIR / "alert_error_log.csv", index=False)

    threshold_rows = []
    for axis, result in models.items():
        positive = result.residuals.loc[result.residuals["Residual"] > 0, "Residual"]
        threshold_rows.append(
            {
                "Axis": axis,
                "Positive Residual Count": len(positive),
                "At or Above MinC": int(positive.ge(result.min_c).sum()),
                "MinC Proportion (%)": float(positive.ge(result.min_c).mean() * 100),
                "At or Above MaxC": int(positive.ge(result.max_c).sum()),
                "MaxC Proportion (%)": float(positive.ge(result.max_c).mean() * 100),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(
        RESULTS_DIR / "threshold_evidence.csv", index=False
    )

    duration_rows = []
    for duration in [2.0, 4.0, 6.0, 10.0]:
        _, duration_events = detect_alerts_and_errors(
            testing, models, minimum_duration_seconds=duration
        )
        duration_rows.append(
            {
                "T Seconds": duration,
                "Approximate Consecutive Readings": int(duration / 2),
                "Alert Count": int(duration_events["Event"].eq("ALERT").sum()) if not duration_events.empty else 0,
                "Error Count": int(duration_events["Event"].eq("ERROR").sum()) if not duration_events.empty else 0,
                "Total Events": len(duration_events),
            }
        )
    pd.DataFrame(duration_rows).to_csv(
        RESULTS_DIR / "duration_sensitivity.csv", index=False
    )

    if stream_to_database:
        try:
            from .database import create_database_engine
            from .synthetic_streaming import stream_synthetic_csv_to_database
        except ImportError:
            from database import create_database_engine
            from synthetic_streaming import stream_synthetic_csv_to_database
        stream_engine = create_database_engine()
        try:
            written = stream_synthetic_csv_to_database(
                DATA_DIR / "synthetic_test_data.csv",
                stream_engine,
                delay_seconds=stream_delay_seconds,
            )
            print(f"Synthetic rows streamed to PostgreSQL: {written}")
        finally:
            stream_engine.dispose()

    for axis, result in models.items():
        safe_axis = axis.lower().replace(" ", "_").replace("#", "")
        figures = [
            plot_training_regression(result, RESULTS_DIR / f"{safe_axis}_regression.png"),
            plot_residual_distribution(result, RESULTS_DIR / f"{safe_axis}_residuals.png"),
            plot_test_events(
                detected,
                axis,
                event_log=events,
                output_path=RESULTS_DIR / f"{safe_axis}_events.png",
            ),
        ]
        for figure in figures:
            plt.close(figure)

    print("Regression pipeline completed.")
    print(f"Training source: {source}")
    print(f"T: {minimum_duration_seconds:.1f} seconds")
    print(f"Events logged: {len(events)}")
    print(summary.round(4).to_string(index=False))
    return summary, detected, events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["database", "csv"], default="database")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--stream-to-database", action="store_true")
    parser.add_argument("--stream-delay", type=float, default=2.0)
    arguments = parser.parse_args()
    run(
        arguments.source,
        arguments.duration,
        stream_to_database=arguments.stream_to_database,
        stream_delay_seconds=arguments.stream_delay,
    )
