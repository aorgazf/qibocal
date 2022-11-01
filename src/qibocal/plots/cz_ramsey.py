# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy
import scipy.signal as ss
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.signal import lfilter

from qibocal.data import Data, Dataset
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


# For cz_ramsey
def snz(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")

    ON = "ON"
    OFF = "OFF"

    # Making figure
    figs = {}
    combinations = np.unique(
        np.vstack(
            (data.df["targetqubit"].to_numpy(), data.df["controlqubit"].to_numpy())
        ).transpose(),
        axis=0,
    )
    m_types = pd.unique(data.df["Mtype"])
    for i in combinations:
        q_target = i[0]
        q_control = i[1]
        for m_type in m_types:
            # Extracting Data
            QubitControl_ON = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == ON)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_control)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitControl_OFF = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_control)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitTarget_ON = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == ON)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            QubitTarget_OFF = data.get_values("prob", "dimensionless")[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()

            amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
            duration = data.get_values("flux_pulse_duration", "ns")
            flux_pulse_duration = duration[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()
            flux_pulse_amplitude = amplitude[
                (data.df["ON_OFF"] == OFF)
                & (data.df["controlqubit"] == q_control)
                & (data.df["targetqubit"] == q_target)
                & (data.df["result_qubit"] == q_target)
                & (data.df["Mtype"] == m_type)
            ].to_numpy()

            amplitude_unique = flux_pulse_amplitude[
                flux_pulse_duration == flux_pulse_duration[0]
            ]
            duration_unique = flux_pulse_duration[
                flux_pulse_amplitude == flux_pulse_amplitude[0]
            ]

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"] = make_subplots(
                rows=2,
                cols=2,
                horizontal_spacing=0.1,
                vertical_spacing=0.2,
                subplot_titles=(
                    f"Qubit Target {q_target}",
                    f"Qubit Control {q_control}",
                ),
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=QubitTarget_ON,
                    name="QubitTarget ON",
                    colorbar=dict(len=0.46, y=0.75),
                ),
                row=1,
                col=1,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=QubitTarget_OFF,
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitTarget OFF",
                ),
                row=2,
                col=1,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=QubitControl_ON,
                    colorbar=dict(len=0.46, y=0.75),
                    name="QubitControl ON",
                ),
                row=1,
                col=2,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=QubitControl_OFF,
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitControl OFF",
                ),
                row=2,
                col=2,
            )

            figs[f"c{q_control}_t{q_target}_raw_{m_type}"].update_layout(
                xaxis_title="Pulse duration (ns)",
                yaxis_title="Amplitude (a.u.)",
                xaxis2_title="Pulse duration (ns)",
                yaxis2_title="Amplitude (a.u.)",
                title=f"Raw data",
            )

            # Normalizing between -1 and 1, and getting simple phase
            figs[f"c{q_control}_t{q_target}_phase"] = go.Figure()
            QubitTarget_ON_norm = QubitTarget_ON - 0.5
            QubitTarget_ON_norm = QubitTarget_ON_norm / np.max(
                np.abs(QubitTarget_ON_norm)
            )
            QubitTarget_OFF_norm = QubitTarget_OFF - 0.5
            QubitTarget_OFF_norm = QubitTarget_OFF_norm / np.max(
                np.abs(QubitTarget_OFF_norm)
            )

            figs[f"c{q_control}_t{q_target}_phase"].add_trace(
                go.Heatmap(
                    x=flux_pulse_duration,
                    y=flux_pulse_amplitude,
                    z=np.rad2deg(
                        np.arccos(QubitTarget_ON_norm) - np.arccos(QubitTarget_OFF_norm)
                    ),
                    colorbar=dict(len=0.46, y=0.25),
                    name="QubitControl OFF",
                ),
            )
            figs[f"c{q_control}_t{q_target}_phase"].update_layout(
                uirevision="0",
                yaxis_title="Amplitude (a.u.)",
                xaxis_title="t_p",
                title=f"Control {q_control}, Target {q_target}",
            )

    return figs
