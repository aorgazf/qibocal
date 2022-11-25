import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import DataUnits
from qibocal.fitting.utils import lorenzian


def frequency_attenuation_1D_msr_phase(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")

    frequency = data.get_values("frequency", "GHz").to_numpy()
    attenuation = data.get_values("attenuation", "dB").to_numpy()
    x = frequency[attenuation == attenuation[0]]
    y = attenuation[frequency == frequency[0]]

    msr = data.get_values("MSR", "V").to_numpy()
    phase = data.get_values("phase", "degree").to_numpy()

    # z = np.ones((len(x),len(y)))*np.nan
    # z_phase = np.ones((len(x),len(y)))*np.nan

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

    for yi in y:
        fig.add_trace(
            go.Scatter(x=x, y=msr[yi == attenuation], name=f"MSR, att = {yi} dB"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=phase[yi == attenuation], name=f"Phase, att = {yi} dB"),
            row=1,
            col=2,
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Attenuation (dB)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Attenuation (dB)",
    )
    return fig


def frequency_amplitude_1D_msr_phase(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, routine, format, f"data_q{qubit}")

    frequency = data.get_values("frequency", "GHz").to_numpy()
    amplitude = data.get_values("amplitude", "dimensionless").to_numpy()
    x = frequency[amplitude == amplitude[0]]
    y = amplitude[frequency == frequency[0]]

    msr = data.get_values("MSR", "V").to_numpy()
    phase = data.get_values("phase", "degree").to_numpy()

    # z = np.ones((len(x),len(y)))*np.nan
    # z_phase = np.ones((len(x),len(y)))*np.nan

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

    for yi in y:
        fig.add_trace(
            go.Scatter(x=x, y=msr[yi == amplitude], name=f"MSR, A = {yi}"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=phase[yi == amplitude], name=f"Phase, A = {yi}"),
            row=1,
            col=2,
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="Amplitude",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Amplitude",
    )
    return fig


def frequency_amplitude_msr_phase(folder, routine, qubit, format):
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
            y=data.get_values("amplitude", "dimensionless"),
            z=data.get_values("MSR", "V"),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("frequency", "GHz"),
            y=data.get_values("amplitude", "dimensionless"),
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
