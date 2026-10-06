import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import deque

from database import quote_identifier


class StreamingSimulator:

    def __init__(self, file_path, database_url):
        self.df = pd.read_csv(file_path)
        self.current_index = 0
        self.database_url = database_url

    def reset(self):
        """Reset stream position so the next read starts from the first row."""
        self.current_index = 0
        return self

    def nextDataPoint(self):
        if self.current_index >= len(self.df):
            return None

        data_point = self.df.iloc[self.current_index]

        self.current_index += 1

        return data_point

    def saveToDatabase(self, data_point):

        # Open a database connection
        engine = create_engine(self.database_url)

        # Convert Series to Python native dictionary, then to DataFrame
        native_dict = {k: (v.item() if hasattr(v, 'item') else v) for k, v in data_point.to_dict().items()}
        record_df = pd.DataFrame([native_dict])

        # Insert the record into PostgreSQL
        record_df.to_sql(
            "robot_stream",
            engine,
            if_exists="append",
            index=False
        )

        # Close the database connection
        engine.dispose()

        print("Record saved to database.")

    def plotDataPoint(self, data_point, stream_history=None):
        """Plot each robot axis current over time as a multi-line dashboard."""
        axes = [
            "Axis #1",
            "Axis #2",
            "Axis #3",
            "Axis #4",
            "Axis #5",
            "Axis #6",
            "Axis #7",
            "Axis #8",
        ]

        if stream_history is None:
            stream_history = pd.DataFrame([data_point.to_dict()])
        elif not stream_history.empty:
            stream_history = pd.concat(
                [stream_history, pd.DataFrame([data_point.to_dict()])],
                ignore_index=True,
            )

        if "Time" not in stream_history.columns:
            stream_history["Time"] = pd.NaT

        stream_history = stream_history[["Time", *axes]].copy()
        stream_history["Time"] = pd.to_datetime(stream_history["Time"], errors="coerce")

        clear_output(wait=True)
        plt.figure(figsize=(12, 6))

        for axis in axes:
            axis_history = stream_history[["Time", axis]].dropna().copy()
            if axis_history.empty:
                continue
            plt.plot(axis_history["Time"], axis_history[axis], marker="o", linewidth=2, label=axis)

        plt.title("Robot Current by Axis Over Time")
        plt.xlabel("Time")
        plt.ylabel("Current")
        plt.xticks(rotation=45)
        plt.legend(title="Robot Axis")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return plt.gcf(), plt.gca()

    def startStreaming(self, number_of_records=10, reset=True):
        import time

        if reset:
            self.reset()

        stream_history = pd.DataFrame()

        for i in range(number_of_records):
            data_point = self.nextDataPoint()

            if data_point is None:
                print("End of CSV file.")
                break

            self.saveToDatabase(data_point)
            stream_history = pd.concat(
                [stream_history, pd.DataFrame([data_point.to_dict()])],
                ignore_index=True,
            )
            self.plotDataPoint(data_point, stream_history)

            print(f"Streamed record {i + 1}")
            print(f"Time: {data_point['Time']}")

            time.sleep(2)


class PostgreSQLStream:
    """Read PostgreSQL telemetry sequentially using the simulator API."""

    def __init__(
        self,
        engine,
        columns,
        table_name="robot_telemetry",
        schema="public",
        batch_size=100,
    ):
        self.engine = engine
        self.columns = columns
        self.table_name = table_name
        self.schema = schema
        self.batch_size = batch_size
        self.offset = 0
        self.pending_records = deque()

    def reset(self):
        self.offset = 0
        self.pending_records.clear()
        return self

    def _load_next_batch(self):
        selected = ", ".join(
            quote_identifier(self.engine, column) for column in self.columns
        )
        query = text(
            f"SELECT {selected} "
            f"FROM {quote_identifier(self.engine, self.schema)}."
            f"{quote_identifier(self.engine, self.table_name)} "
            f"ORDER BY {quote_identifier(self.engine, 'Time')} ASC "
            "LIMIT :batch_size OFFSET :row_offset"
        )
        with self.engine.connect() as connection:
            result = connection.execute(
                query,
                {"batch_size": self.batch_size, "row_offset": self.offset},
            )
            self.pending_records.extend(dict(row._mapping) for row in result)

    def nextDataPoint(self):
        """Return the next row, or None when all current rows are consumed."""
        if not self.pending_records:
            self._load_next_batch()
        if not self.pending_records:
            return None

        self.offset += 1
        return pd.Series(self.pending_records.popleft())
