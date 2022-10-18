# -*- coding: utf-8 -*-
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, Dataset
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


# For cryoscope
def cryoscope(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    import numpy as np

    MY_tag = "MY"
    MX_tag = "MX"

    y = data.get_values("MSR", "uV")[data.df["component"] == MY_tag].to_numpy()
    x = data.get_values("MSR", "uV")[data.df["component"] == MX_tag].to_numpy()
    flux_pulse_duration = data.get_values("flux_pulse_duration", "ns")[
        data.df["component"] == MY_tag
    ].to_numpy()
    flux_pulse_amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")[
        data.df["component"] == MY_tag
    ].to_numpy()
    phi = np.arctan(y / x)
    phi = np.unwrap(phi)

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "phase (rad)",
            "dphi_dt (Hz)",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            x=flux_pulse_duration,
            y=flux_pulse_amplitude,
            z=phi,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    total_dphi_dt = np.array([])
    total_flux_pulse_duration = np.array([])
    total_flux_pulse_amplitude = np.array([])
    amplitudes = data.df[data.df["component"] == MY_tag][
        "flux_pulse_amplitude"
    ].pint.magnitude.unique()
    for amplitude in amplitudes:
        y = (
            data.df[data.df["component"] == MY_tag][
                data.df["flux_pulse_amplitude"] == amplitude
            ]["MSR"]
            .pint.to("uV")
            .pint.magnitude.to_numpy()
        )
        x = (
            data.df[data.df["component"] == MX_tag][
                data.df["flux_pulse_amplitude"] == amplitude
            ]["MSR"]
            .pint.to("uV")
            .pint.magnitude.to_numpy()
        )
        flux_pulse_duration = (
            data.df[data.df["component"] == MY_tag][
                data.df["flux_pulse_amplitude"] == amplitude
            ]["flux_pulse_duration"]
            .pint.to("ns")
            .pint.magnitude.to_numpy()
        )
        phi = np.arctan(y / x)
        phi = np.unwrap(phi)
        flux_pulse_duration_step = flux_pulse_duration[1] - flux_pulse_duration[0]
        dphi_dt = np.diff(phi) / flux_pulse_duration_step * 1e9
        flux_pulse_duration = flux_pulse_duration[1:]
        delta = np.mean(dphi_dt)
        total_dphi_dt = np.concatenate((total_dphi_dt, dphi_dt))
        total_flux_pulse_duration = np.concatenate(
            (total_flux_pulse_duration, flux_pulse_duration)
        )
        total_flux_pulse_amplitude = np.concatenate(
            (total_flux_pulse_amplitude, [amplitude] * len(dphi_dt))
        )

    fig.add_trace(
        go.Heatmap(
            x=total_flux_pulse_duration,
            y=total_flux_pulse_amplitude,
            z=total_dphi_dt,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="flux pulse duration (ns)",
        yaxis_title="flux pulse amplitude (dimensionless)",
        xaxis2_title="flux pulse duration (ns)",
        yaxis2_title="dphi_dt (Hz)",
    )
    return fig
