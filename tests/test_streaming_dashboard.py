import pandas as pd

from src.StreamingSimulator import StreamingSimulator


def test_streaming_dashboard_tracks_axis_current_over_time(tmp_path):
    csv_path = tmp_path / "robot_stream.csv"
    history = pd.DataFrame(
        [
            {
                "Time": "2022-10-17 12:18:23.660000",
                "Axis #1": 1.0,
                "Axis #2": 2.0,
                "Axis #3": 3.0,
                "Axis #4": 4.0,
                "Axis #5": 5.0,
                "Axis #6": 6.0,
                "Axis #7": 7.0,
                "Axis #8": 8.0,
            },
            {
                "Time": "2022-10-17 12:18:25.472000",
                "Axis #1": 2.0,
                "Axis #2": 3.0,
                "Axis #3": 4.0,
                "Axis #4": 5.0,
                "Axis #5": 6.0,
                "Axis #6": 7.0,
                "Axis #7": 8.0,
                "Axis #8": 9.0,
            },
        ]
    )
    history.to_csv(csv_path, index=False)

    simulator = StreamingSimulator(str(csv_path), "sqlite:///:memory:")
    data_point = history.iloc[0]
    fig, ax = simulator.plotDataPoint(data_point, history)

    assert len(ax.lines) == 8
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Current"
    assert fig is not None
