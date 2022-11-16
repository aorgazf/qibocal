import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import exp


# T1
def t1_time_msr_phase(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"Time": "ns"})

    try:
        data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}")
    except:
        data_fit = DataUnits()

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    datasets = []
    copy = data.df.copy()
    for i in range(len(copy)):
        datasets.append(copy.drop_duplicates("Time"))
        copy.drop(datasets[-1].index, inplace=True)
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["Time"].pint.to("ns").pint.magnitude,
                y=datasets[-1]["MSR"].pint.to("uV").pint.magnitude,
                marker_color="rgb(100, 0, 255)",
                opacity=0.3,
                name="MSR",
                showlegend=not bool(i),
                legendgroup="group1",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=datasets[-1]["Time"].pint.to("ns").pint.magnitude,
                y=datasets[-1]["phase"].pint.to("rad").pint.magnitude,
                marker_color="rgb(102, 180, 71)",
                name="phase",
                opacity=0.3,
                showlegend=not bool(i),
                legendgroup="group2",
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=data.df.Time.drop_duplicates().pint.to("ns").pint.magnitude,
            y=data.df.groupby("Time")["MSR"]
            .pint.to("uV")
            .mean()
            .pint.magnitude,  # CHANGE THIS TO
            name="average MSR",
            marker_color="rgb(100, 0, 255)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.df.Time.drop_duplicates().pint.to("ns").pint.magnitude,
            y=data.df.groupby("Time")["phase"].mean().pint.magnitude,
            name="average phase",
            marker_color="rgb(204, 102, 102)",
        ),
        row=1,
        col=2,
    )

    # add fitting trace
    if len(data) > 0 and len(data_fit) > 0:
        timerange = np.linspace(
            min(data.get_values("Time", "ns")),
            max(data.get_values("Time", "ns")),
            2 * len(data),
        )
        params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
        fig.add_trace(
            go.Scatter(
                x=timerange,
                y=exp(
                    timerange,
                    data_fit.get_values("popt0"),
                    data_fit.get_values("popt1"),
                    data_fit.get_values("popt2"),
                ),
                name="Fit MSR",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            dict(
                font=dict(color="black", size=12),
                x=0,
                y=-0.20,
                showarrow=False,
                text=f"Estimated {params[0]} is {data_fit.df[params[0]][0]:.1f} ns.",
                textangle=0,
                xanchor="left",
                xref="paper",
                yref="paper",
            )
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )
    return fig
