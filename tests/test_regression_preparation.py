import pandas as pd

from src.data_preparation import AXES, active_axis_rows, prepare_telemetry


def test_zero_rows_are_preserved_but_excluded_from_axis_training():
    rows = []
    for time, value in [
        ("2022-10-17T12:00:00Z", 0.0),
        ("2022-10-17T12:00:02Z", 2.5),
    ]:
        row = {"Time": time, **{axis: 0.0 for axis in AXES}}
        row["Axis #1"] = value
        rows.append(row)

    prepared = prepare_telemetry(pd.DataFrame(rows))
    training = active_axis_rows(prepared, "Axis #1")

    assert len(prepared) == 2
    assert prepared["Operating State"].tolist() == ["STOPPED", "ACTIVE"]
    assert training["Axis #1"].tolist() == [2.5]
