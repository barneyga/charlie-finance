"""Reusable Plotly chart builders for the macro dashboard."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charlie.storage.db import Database
from charlie.storage.models import query_series


CHART_TEMPLATE = "plotly_dark"
COLORS = ["#60a5fa", "#f97316", "#22c55e", "#a78bfa", "#f472b6", "#facc15"]


def _add_recession_bars(fig: go.Figure, db: Database, start: str | None = None):
    """Add shaded recession bars from USREC series."""
    try:
        rec = query_series(db, "USREC", start=start)
        if rec.empty:
            return

        in_recession = False
        rec_start = None
        for date, val in rec.items():
            if val == 1 and not in_recession:
                rec_start = date
                in_recession = True
            elif val == 0 and in_recession:
                fig.add_vrect(
                    x0=rec_start, x1=date,
                    fillcolor="rgba(128,128,128,0.15)",
                    line_width=0,
                    layer="below",
                )
                in_recession = False
        if in_recession and rec_start is not None:
            fig.add_vrect(
                x0=rec_start, x1=rec.index[-1],
                fillcolor="rgba(128,128,128,0.15)",
                line_width=0,
                layer="below",
            )
    except Exception:
        pass


def time_series_chart(
    df: pd.DataFrame | pd.Series,
    title: str,
    db: Database | None = None,
    yaxis_title: str = "",
    recession_bars: bool = True,
    height: int = 400,
) -> go.Figure:
    """Line chart for one or more time series."""
    fig = go.Figure()

    if isinstance(df, pd.Series):
        df = df.to_frame()

    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))

    if recession_bars and db is not None:
        start = df.index[0].strftime("%Y-%m-%d") if len(df) > 0 else None
        _add_recession_bars(fig, db, start)

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def yield_curve_snapshot(
    curves: dict[str, pd.Series],
    title: str = "Yield Curve",
    height: int = 400,
) -> go.Figure:
    """Plot multiple yield curve snapshots overlaid."""
    fig = go.Figure()
    for i, (label, curve) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=curve.index,
            y=curve.values,
            name=label,
            mode="lines+markers",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        xaxis_title="Tenor",
        yaxis_title="Yield (%)",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def bar_chart(
    series: pd.Series,
    title: str,
    color_positive: str = "#22c55e",
    color_negative: str = "#ef4444",
    height: int = 400,
) -> go.Figure:
    """Bar chart with positive/negative coloring."""
    colors = [color_positive if v >= 0 else color_negative for v in series.values]
    fig = go.Figure(go.Bar(
        x=series.index,
        y=series.values,
        marker_color=colors,
    ))
    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        hovermode="x",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def normalized_returns_chart(
    df: pd.DataFrame,
    title: str,
    height: int = 400,
) -> go.Figure:
    """Rebase multiple series to 100 at start date for comparison."""
    fig = go.Figure()

    for i, col in enumerate(df.columns):
        series = df[col].dropna()
        if series.empty:
            continue
        normalized = (series / series.iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))

    fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        yaxis_title="Indexed (100 = start)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def horizontal_bar_chart(
    data: dict[str, float],
    title: str,
    suffix: str = "%",
    height: int = 400,
) -> go.Figure:
    """Horizontal bar chart for returns/rankings."""
    labels = list(data.keys())
    values = list(data.values())
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}{suffix}" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=100, r=20, t=50, b=40),
    )
    return fig


def dual_axis_chart(
    series1: pd.Series,
    series2: pd.Series,
    title: str,
    y1_title: str = "",
    y2_title: str = "",
    db: Database | None = None,
    height: int = 400,
) -> go.Figure:
    """Two series on separate y-axes."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=series1.index, y=series1.values,
            name=series1.name or "Series 1",
            line=dict(color=COLORS[0], width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=series2.index, y=series2.values,
            name=series2.name or "Series 2",
            line=dict(color=COLORS[1], width=2),
        ),
        secondary_y=True,
    )

    if db is not None:
        start = min(
            series1.index[0] if len(series1) > 0 else pd.Timestamp.now(),
            series2.index[0] if len(series2) > 0 else pd.Timestamp.now(),
        ).strftime("%Y-%m-%d")
        _add_recession_bars(fig, db, start)

    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=50, b=40),
    )
    fig.update_yaxes(title_text=y1_title, secondary_y=False)
    fig.update_yaxes(title_text=y2_title, secondary_y=True)
    return fig


def gauge_chart(
    value: float,
    title: str,
    subtitle: str = "",
    height: int = 280,
) -> go.Figure:
    """Fear/Greed gauge (0-100). Green = greed, red = fear."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "", "font": {"size": 48}},
        title={"text": f"{title}<br><span style='font-size:14px;color:gray'>{subtitle}</span>"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#444"},
            "bar": {"color": "#fff", "thickness": 0.2},
            "bgcolor": "#1e1e1e",
            "steps": [
                {"range": [0, 20], "color": "#22c55e"},
                {"range": [20, 40], "color": "#86efac"},
                {"range": [40, 60], "color": "#eab308"},
                {"range": [60, 80], "color": "#f97316"},
                {"range": [80, 100], "color": "#ef4444"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig
