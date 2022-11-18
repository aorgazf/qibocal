import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits


def frequency_fidelity(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(quantities={"frequency": "Hz", "fidelity": "dimensionless"})

    fig = go.Figure(
        go.Scatter(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("fidelity", "dimensionless"),
        )
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Fidelity (prob)",
    )
    return fig


def power_fidelity(folder, routine, qubit, format):
    try:
        data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
    except:
        data = DataUnits(
            quantities={"amplitude": "dimensionless", "fidelity": "dimensionless"}
        )

    fig = go.Figure(
        go.Scatter(
            x=data.get_values("amplitude", "dimensionless"),
            y=data.get_values("fidelity", "dimensionless"),
        )
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (a.u.)",
        yaxis_title="Fidelity (prob)",
    )
    return fig
