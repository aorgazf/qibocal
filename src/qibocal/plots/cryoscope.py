# -*- coding: utf-8 -*-
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy
import scipy.signal as ss
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.signal import lfilter

from qibocal.data import Data, Dataset
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


# For cryoscope
def cryoscope(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"
    MZ_tag = "MZ"

    # Extracting Data
    z = data.get_values("prob", "dimensionless")[
        data.df["component"] == MZ_tag
    ].to_numpy()
    x = data.get_values("prob", "dimensionless")[
        data.df["component"] == MX_tag
    ].to_numpy()
    y = data.get_values("prob", "dimensionless")[
        data.df["component"] == MY_tag
    ].to_numpy()
    x = x[: len(z)]
    y = y[: len(z)]

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MZ_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MZ_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    # Making figure
    figs = []

    figs = [
        make_subplots(
            rows=2,
            cols=1,
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
            subplot_titles=(
                "X_raw",
                "Y_raw",
            ),
        ),
        make_subplots(
            rows=2,
            cols=1,
            horizontal_spacing=0.1,
            vertical_spacing=0.2,
            subplot_titles=(
                "X_norm",
                "Y_norm",
            ),
        ),
    ]

    figs[0].add_trace(
        go.Heatmap(
            x=flux_pulse_duration,
            y=flux_pulse_amplitude,
            z=x,
            colorbar=dict(len=0.46, y=0.75),
        ),
        row=1,
        col=1,
    )

    figs[0].add_trace(
        go.Heatmap(
            x=flux_pulse_duration,
            y=flux_pulse_amplitude,
            z=y,
            colorbar=dict(len=0.46, y=0.25),
        ),
        row=2,
        col=1,
    )
    figs[0].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (a.u.)",
        xaxis2_title="Pulse duration (ns)",
        yaxis2_title="Amplitude (a.u.)",
        title=f"Raw data",
    )

    # Smoothing and -1/1 center
    smoothing = False
    x_norm = np.empty((len(amplitude_unique), len(duration_unique)))
    y_norm = np.empty((len(amplitude_unique), len(duration_unique)))
    for i, amp in enumerate(amplitude_unique):
        if smoothing:
            x_norm[i, :], y_norm[i, :] = normalize_sincos(
                x[amplitude == flux_pulse_amplitude[i]],
                y[amplitude == flux_pulse_amplitude[i]],
            )  # FIXME: not working yet
        else:  # To put on one line
            idx = np.where(flux_pulse_amplitude == amp)
            if amp == flux_pulse_amplitude[-1]:
                n = int(np.where(flux_pulse_duration[-1] == duration_unique)[0][0]) + 1
            else:
                n = len(duration_unique)
            x_norm[i, :n] = x[idx]
            x_norm[i, :n] = x_norm[i, :n] - 0.5
            x_norm[i, :n] = x_norm[i, :n] / max(abs(x_norm[i, :n]))
            y_norm[i, :n] = y[idx]
            y_norm[i, :n] = y_norm[i, :n] - 0.5
            y_norm[i, :n] = y_norm[i, :n] / max(abs(y_norm[i, :n]))

    figs[1].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=x_norm,
            colorbar=dict(len=0.46, y=0.75),
        ),
        row=1,
        col=1,
    )

    figs[1].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=y_norm,
            colorbar=dict(len=0.46, y=0.25),
        ),
        row=2,
        col=1,
    )
    figs[1].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (a.u.)",
        xaxis2_title="Pulse duration (ns)",
        yaxis2_title="Amplitude (a.u.)",
        title=f"Norm data",
    )

    # Plot the phase between X and Y
    figs += [go.Figure()]
    phi = np.arctan2(x_norm, y_norm)
    figs[2].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phi,
        ),
    )
    figs[2].update_layout(
        xaxis_title="Pulse duration (ns)", yaxis_title="Amplitude (a.u.)", title=f"Phi"
    )

    # Plot the unwrapped
    figs += [go.Figure()]
    phi_unwrap = np.unwrap(
        phi, axis=0
    )  # Axis 0 = unwrap along row <==> amplitude / Axis 1 = unwrap along row <==> duration
    figs[3].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phi_unwrap,
        ),
    )
    figs[3].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (a.u.)",
        title=f"Phi unwrapped",
    )

    # Plot frequency shift (z) for amplitude duration = real pulse
    figs += [go.Figure()]

    # dt = np.diff(duration_unique) * 1e-9
    # dphi_dt_unwrap = np.diff(phi_unwrap, axis=1) / dt
    dphi_dt_unwrap = phi_unwrap * 1e9 / duration_unique
    detuning = dphi_dt_unwrap / (2 * np.pi)
    figs[4].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=detuning,
        ),
    )
    figs[4].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (a.u.)",
        title=f"Detuning (Hz)",
    )

    # From the Amplitude(Detunning) function, reconstruct the pulse, Amplitude(time) from the measured detunning
    figs += [
        go.Figure()
    ]  # 5 for the stable detuning frequency as a function of amplitude
    figs += [go.Figure()]  # 6 for the reconstructed pulse and its fit

    detuning_median = np.median(detuning, axis=1)  # Stable detuning after a long pulse
    figs[5].add_trace(
        go.Scatter(
            x=amplitude_unique,
            y=detuning_median,
        ),
    )
    figs[5].update_layout(
        xaxis_title="Amplitude (a.u.)",
        yaxis_title="Detuning (Hz)",
        title=f"Detuning as a function of amplitude",
    )

    pulses = np.empty((len(amplitude_unique), len(duration_unique)))
    a_b_fits = np.empty((len(amplitude_unique), 4))
    for i, amp in enumerate(amplitude_unique):
        awg_reconstructed_pulse = np.interp(
            detuning[i, :],  # np.append(detuning[i, :], [detuning_median[i]]),
            detuning_median,
            amplitude_unique,
        )
        figs[6].add_trace(
            go.Scatter(
                visible=False,
                x=duration_unique,
                y=awg_reconstructed_pulse,
                name=f"A = {amp}",
            ),
        )
        b_fit, a_fit = cryoscope_fit(
            duration_unique,
            amp * np.ones(len(duration_unique)),
            awg_reconstructed_pulse,
        )
        a_b_fits[i, :] = np.append(a_fit, b_fit)
        pulses[i, :] = awg_reconstructed_pulse

    figs[6].data[-1].visible = True

    steps = []
    for i, amp in enumerate(amplitude_unique):
        figs[6].add_trace(
            go.Scatter(
                visible=False,
                x=duration_unique,
                y=lfilter(
                    a_b_fits[i, 2:-1],
                    a_b_fits[i, 0:2],
                    amp * np.ones(len(duration_unique)),
                ),
                name=f"A_fit = {amp}",
            ),
        )
        step = dict(
            method="restyle",
            args=["visible", [False] * 2 * len(amplitude_unique)],
            label=str(amp),
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        step["args"][1][i + len(amplitude_unique)] = True
        steps.append(step)
    sliders = [
        dict(steps=steps, active=10, currentvalue={"prefix": "Amplitude (a.u.) = "})
    ]
    figs[6].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (a.u.)",
        title=f"Reconstructed pulse",
        sliders=sliders,
    )

    return figs


def normalize_sincos(x, y, window_size_frac=61, window_size=None, do_envelope=True):

    if window_size is None:
        window_size = len(x) // window_size_frac

        # window size for savgol filter must be odd
        window_size -= (window_size + 1) % 2

    x = ss.savgol_filter(x, window_size, 0, 0)  ## why poly order 0
    y = ss.savgol_filter(y, window_size, 0, 0)

    return x, y  # this is returning just the noise,


def cryoscope_fit(x_data, y_data, awg_reconstructed_pulse):
    pguess = [1, 0, 1, 0]  # b0  # b1  # a0  # a1
    try:
        popt, pcov = curve_fit(
            lambda t, *p: cryoscope_step_response(t, p, awg_reconstructed_pulse),
            x_data,
            y_data,
            p0=pguess,
        )
        b_fit = popt[[0, 1]]
        a_fit = popt[[2, 3]]
    except:
        a_fit = [np.nan, np.nan]
        b_fit = [np.nan, np.nan]

    return b_fit, a_fit


def cryoscope_step_response(t, p, awg_reconstructed_pulse):
    # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
    # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
    # p = [b0, b1, a0, a1]

    b_fit = p[:2]
    a_fit = p[2:]
    return lfilter(b_fit, a_fit, awg_reconstructed_pulse)
