import pandas as pd

from src.alert_detection import detect_alerts_and_errors
from src.data_preparation import AXES, prepare_telemetry
from src.regression_model import fit_all_axis_models
from src.synthetic_data import generate_synthetic_test_data


def test_synthetic_data_produces_sustained_events():
    rows = []
    for i in range(120):
        row = {
            "Time": pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(seconds=2 * i)
        }
        for number, axis in enumerate(AXES, start=1):
            row[axis] = number + 0.02 * i + (0.1 if i % 2 else -0.1)
        rows.append(row)

    training = prepare_telemetry(pd.DataFrame(rows))
    models = fit_all_axis_models(training)
    testing = generate_synthetic_test_data(training, models, rows=300)
    for axis in AXES:
        assert f"{axis} Z Score" in testing.columns
        assert f"{axis} Normalized" in testing.columns
        assert testing[f"{axis} Normalized"].between(0, 1).all()
    _, events = detect_alerts_and_errors(testing, models, minimum_duration_seconds=6)

    assert not events.empty
    assert set(events["Event"]).issubset({"ALERT", "ERROR"})
    assert events["Duration Seconds"].ge(6).all()
