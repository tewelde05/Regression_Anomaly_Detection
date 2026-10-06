"""PostgreSQL operations for robot telemetry."""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, inspect, text


def create_database_engine() -> Engine:
    """Create an engine from the Neon URL stored in the local .env file."""
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL was not found. Copy .env.example to .env and add "
            "your Neon PostgreSQL connection URL."
        )
    return create_engine(database_url, pool_pre_ping=True)


def test_connection(engine: Engine) -> None:
    """Confirm that PostgreSQL accepts the connection."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    print("Database connection successful!")


def save_telemetry(
    dataframe: pd.DataFrame,
    engine: Engine,
    table_name: str = "robot_telemetry",
) -> None:
    """Replace a PostgreSQL table with the supplied telemetry."""
    dataframe.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        chunksize=1000,
        method="multi",
    )
    print(f"Data successfully stored in PostgreSQL table '{table_name}'.")


def count_records(engine: Engine, table_name: str = "robot_telemetry") -> int:
    """Count stored records. The table name is restricted to safe characters."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError("Invalid table name.")

    with engine.connect() as connection:
        result = connection.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
        return int(result.scalar_one())


def quote_identifier(engine: Engine, identifier: str) -> str:
    """Quote a PostgreSQL table, schema, or column identifier safely."""
    return engine.dialect.identifier_preparer.quote(identifier)


def verify_telemetry_table(
    engine: Engine,
    table_name: str,
    required_columns: list[str],
    schema: str = "public",
) -> None:
    """Confirm that a telemetry table and its required columns exist."""
    database_inspector = inspect(engine)
    available_tables = database_inspector.get_table_names(schema=schema)
    if table_name not in available_tables:
        raise RuntimeError(
            f"Required table '{table_name}' was not found. "
            f"Available tables: {available_tables}"
        )

    available_columns = {
        column["name"]
        for column in database_inspector.get_columns(table_name, schema=schema)
    }
    missing_columns = set(required_columns) - available_columns
    if missing_columns:
        raise RuntimeError(
            f"Table '{table_name}' is missing columns: {sorted(missing_columns)}"
        )


def load_telemetry_from_database(
    engine: Engine,
    table_name: str,
    columns: list[str],
    schema: str = "public",
) -> pd.DataFrame:
    """Load ordered telemetry columns from PostgreSQL."""
    selected_columns = ", ".join(quote_identifier(engine, col) for col in columns)
    query = text(
        f"SELECT {selected_columns} "
        f"FROM {quote_identifier(engine, schema)}.{quote_identifier(engine, table_name)} "
        f"ORDER BY {quote_identifier(engine, 'Time')} ASC"
    )
    return pd.read_sql_query(query, engine)
