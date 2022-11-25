import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import DataUnits


def frequency_flux_msr_phase__matrix(folder, routine, qubit, format):
    fluxes = []
    for i in range(25):  # FIXME: 25 is hardcoded
        file = f"{folder}/data/{routine}/data_q{qubit}_f{i}.csv"
        if os.path.exists(file):
            fluxes += [i]

    if len(fluxes) < 1:
        nb = 1
    else:
        nb = len(fluxes) + 1
    fig = make_subplots(
        rows=2,
        cols=nb,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        x_title="Frequency (Hz)",
        y_title="Current (A)",
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for j in fluxes:
        if j == fluxes[-1]:
            showscale = True
        else:
            showscale = False
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}_f{j}")
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
                z=data.get_values("MSR", "V"),
                showscale=showscale,
            ),
            row=1,
            col=j + 1,
        )
        fig.add_trace(
            go.Heatmap(
                x=data.get_values("frequency", "GHz"),
                y=data.get_values("current", "A"),
                z=data.get_values("phase", "rad"),
                showscale=showscale,
            ),
            row=2,
            col=j + 1,
        )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )
    return fig
