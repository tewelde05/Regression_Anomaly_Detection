"""Interactive Plotly charts for robot telemetry."""

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def find_axis_columns(dataframe: pd.DataFrame) -> list[str]:
    """Find populated numeric columns whose names begin with 'Axis #'."""
    return [
        column
        for column in dataframe.columns
        if column.startswith("Axis #")
        and dataframe[column].notna().any()
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]


def plot_robot_telemetry(
    dataframe: pd.DataFrame,
    time_column: str = "Time",
    output_html: str | None = "robot_telemetry.html",
) -> Figure:
    """Plot each populated robot axis against time and optionally save HTML."""
    axis_columns = find_axis_columns(dataframe)
    if not axis_columns:
        raise ValueError("No populated numeric columns beginning with 'Axis #' were found.")

    telemetry = dataframe[[time_column, *axis_columns]].melt(
        id_vars=time_column,
        var_name="Axis",
        value_name="Value",
    ).dropna(subset=["Value"])

    figure = px.line(
        telemetry,
        x=time_column,
        y="Value",
        color="Axis",
        title="Robot Hand Activity Over Time",
        labels={time_column: "Time", "Value": "Axis value"},
    )
    figure.update_layout(hovermode="x unified", template="plotly_white")
    figure.update_xaxes(rangeslider_visible=True)

    if output_html:
        figure.write_html(output_html)
        print(f"Interactive graph saved as '{output_html}'.")

    figure.show()
    return figure


def _style_event_table(table, events):
    colors = {"ANOMALY": "#f4b6b6", "WARNING": "#fff2cc"}
    for row_number, event in enumerate(events, start=1):
        for column_number in range(6):
            table[(row_number, column_number)].set_facecolor(
                colors.get(event["Status"], "#d9ead3")
            )


def render_maintenance_dashboard(
    visible_history, record, recent_events, axes, robot_state,
    dashboard_message, message_color, robot_name="ROBOT 3",
    time_window_seconds=90,
):
    """Render one frame of the predictive-maintenance dashboard."""
    window_end = record["Time"]
    window_start = window_end - pd.Timedelta(seconds=time_window_seconds)
    figure = plt.figure(figsize=(20, 12), dpi=110, facecolor="white", constrained_layout=True)
    grid = GridSpec(3, 1, figure=figure, height_ratios=[1.25, 7.5, 2.25], hspace=0.12)
    banner_ax, chart_ax, table_ax = [figure.add_subplot(grid[i]) for i in range(3)]

    banner_ax.axis("off")
    connection_text = (
        f"CONNECTED   |   {record['Time']:%Y-%m-%d}   |   {record['Time']:%A}   |   "
        f"{record['Time']:%H:%M:%S}   |   {robot_name}: {robot_state}"
    )
    state_color = "green" if robot_state == "RUNNING" else "#8b0000"
    banner_ax.text(0.5, 0.78, connection_text, ha="center", va="center", color="white",
                   fontsize=11, fontweight="bold",
                   bbox={"boxstyle": "square,pad=0.35", "facecolor": state_color,
                         "edgecolor": state_color})
    banner_ax.text(0.5, 0.48, "Predictive Maintenance Dashboard", ha="center",
                   va="center", color="red", fontsize=14, fontweight="bold")
    banner_ax.text(0.5, 0.14, dashboard_message, ha="center", va="center",
                   color=message_color, fontsize=12, fontweight="bold")

    colors = plt.cm.tab10.colors
    for index, axis in enumerate(axes):
        if axis not in visible_history.columns:
            continue
        values = pd.to_numeric(visible_history[axis], errors="coerce")
        chart_ax.plot(visible_history["Time"], values, linewidth=1.7,
                      color=colors[index % len(colors)], label=axis)
        status = visible_history.get(
            f"{axis} Status", pd.Series("NORMAL", index=visible_history.index)
        ).fillna("NORMAL")
        for label, color, edge, size, order in (
            ("WARNING", "orange", "darkorange", 80, 5),
            ("ANOMALY", "red", "darkred", 95, 6),
        ):
            mask = status.eq(label)
            chart_ax.scatter(visible_history.loc[mask, "Time"], values.loc[mask],
                             marker="X", s=size, color=color, edgecolor=edge,
                             linewidth=1, zorder=order)

    chart_ax.set_title(f"Robot Current by Axis — Latest {time_window_seconds} Seconds",
                       fontsize=15, fontweight="bold", pad=12)
    chart_ax.set_xlabel("Time", fontsize=12, labelpad=10)
    chart_ax.set_ylabel("Current (A)", fontsize=12)
    chart_ax.grid(True, alpha=0.25)
    chart_ax.legend(title="Robot Axis", loc="upper right", ncol=2, fontsize=8)
    chart_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    chart_ax.xaxis.set_major_locator(mdates.SecondLocator(interval=10))
    chart_ax.tick_params(axis="x", rotation=45)
    chart_ax.set_xlim(window_start, window_end)

    table_ax.axis("off")
    table_ax.set_title("RECENT ANOMALY EVENTS", fontsize=11, fontweight="bold", pad=12)
    columns = ["Time", "Axis", "Current", "P95", "Status", "Action"]
    events = list(recent_events)
    rows = [[event[column] for column in columns] for event in events]
    if not rows:
        rows = [["-", "-", "-", "-", "NORMAL", "NO RECENT ANOMALIES"]]
    table = table_ax.table(cellText=rows, colLabels=columns, cellLoc="center",
                           colLoc="center", loc="center", bbox=[0, 0, 1, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    for column_number in range(len(columns)):
        table[(0, column_number)].set_facecolor("green")
        table[(0, column_number)].set_text_props(color="white", fontweight="bold")
    if events:
        _style_event_table(table, events)
    else:
        for column_number in range(len(columns)):
            table[(1, column_number)].set_facecolor("#d9ead3")
    plt.show()
    return figure
