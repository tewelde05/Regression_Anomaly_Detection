"""Replay synthetic CSV telemetry into PostgreSQL in timestamp order."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine


def stream_synthetic_csv_to_database(
    csv_path: str | Path,
    engine: Engine,
    table_name: str = "synthetic_telemetry_stream",
    delay_seconds: float = 2.0,
    replace_table: bool = True,
    max_records: int | None = None,
) -> int:
    """Simulate CSV-to-database flow by inserting one chronological row at a time."""
    stream = pd.read_csv(csv_path)
    stream["Time"] = pd.to_datetime(stream["Time"], errors="coerce", utc=True)
    stream = stream.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    if max_records is not None:
        stream = stream.head(max_records)

    if replace_table:
        stream.head(0).to_sql(table_name, engine, if_exists="replace", index=False)

    records_written = 0
    for _, row in stream.iterrows():
        pd.DataFrame([row]).to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
        )
        records_written += 1
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return records_written
