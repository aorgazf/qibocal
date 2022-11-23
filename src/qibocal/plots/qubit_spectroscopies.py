import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import DataUnits
from qibocal.fitting.utils import lorenzian


def frequency_attenuation_msr_phase(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")
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

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("attenuation", "dB"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("attenuation", "dB"),
            z=data.get_values("phase", "rad"),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Attenuation (dB)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Attenuation (dB)",
    )
    return fig
