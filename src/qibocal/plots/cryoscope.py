# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy
import scipy.signal as ss
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.signal import lfilter

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, rabi, ramsey


# For cryoscope
def cryoscope_raw(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    # Making figure
    figs = {}
    figs["raw"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    for amp in amplitude_unique:
        figs["raw"].add_trace(
            go.Scatter(
                x=flux_pulse_duration,
                y=data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy(),
                name=f"<X> | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )

        figs["raw"].add_trace(
            go.Scatter(
                x=flux_pulse_duration,
                y=data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy(),
                name=f"<Y> | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )
    figs["raw"].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="MSR (prob)",
        title=f"Raw data",
    )
    return figs["raw"]


# For cryoscope
def cryoscope_raw_heatmap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"

    # Extracting Data
    x = data.get_values("prob", "dimensionless")[
        data.df["component"] == MX_tag
    ].to_numpy()
    y = data.get_values("prob", "dimensionless")[
        data.df["component"] == MY_tag
    ].to_numpy()
    x = x[: len(y)]
    y = y[: len(y)]

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    # Making figure
    figs = {}

    figs["raw"] = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.3,
        subplot_titles=(
            "<X>",
            "<Y>",
        ),
    )

    figs["raw"].add_trace(
        go.Heatmap(
            x=flux_pulse_duration,
            y=flux_pulse_amplitude,
            z=x,
            colorbar=dict(len=0.46, y=0.75),
        ),
        row=1,
        col=1,
    )

    figs["raw"].add_trace(
        go.Heatmap(
            x=flux_pulse_duration,
            y=flux_pulse_amplitude,
            z=y,
            colorbar=dict(len=0.46, y=0.25),
        ),
        row=1,
        col=2,
    )
    figs["raw"].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (a.u.)",
        xaxis2_title="Pulse duration (ns)",
        yaxis2_title="Amplitude (a.u.)",
        title=f"Raw data",
    )
    return figs["raw"]

# For cryoscope
def cryoscope_norm(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    # Making figure
    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    for amp in amplitude_unique:
        x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

        x_norm = x_data - 0.5
        y_norm = y_data - 0.5

        x_norm = x_norm / max(abs(x_norm))
        y_norm = y_norm / max(abs(y_norm))

        figs["norm"].add_trace(
            go.Scatter(
                x=flux_pulse_duration,
                y=x_norm,
                name=f"<X> | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )

        figs["norm"].add_trace(
            go.Scatter(
                x=flux_pulse_duration,
                y=y_norm,
                name=f"<Y> | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )
    figs["norm"].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="MSR (prob)",
        title=f"Normalized data",
    )
    return figs["norm"]

# For cryoscope
def cryoscope_norm_fft(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    import numpy as np
    import scipy.signal as ss

    MX_tag = "MX"
    MY_tag = "MY"

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    # Making figure
    figs = {}
    figs["norm"] = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )

    mean_detuning = []
    mean_detuning2 = []
    mean_detuning3 = []

    for amp in amplitude_unique:
        x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

        x_norm = x_data - 0.5
        y_norm = y_data - 0.5
        x_norm = x_norm / max(abs(x_norm))
        y_norm = y_norm / max(abs(y_norm))

        norm_data = x_norm + 1j * y_norm
        
        sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
        derivative_window_length = 7 / sampling_rate
        derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
        derivative_window_size += (derivative_window_size + 1) % 2
        derivative_order = 2

        freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
        demod_freq = - \
                freq_guess * sampling_rate
        
        #y = amp*exp(2pi i f t + ph) + off.
        demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
        phase = np.arctan2(demod_data.imag, demod_data.real)
        phase_unwrapped = np.unwrap(phase)
        #phase_unwrapped_angle = np.unwrap(np.angle(demod_data))

        # use a savitzky golay filter: it take sliding window of length
        # `window_length`, fits a polynomial, returns derivative at
        # middle point
        # phase = phase_unwrapped_data

        # Ramiro method 
        # wl = len(duration_unique) 
        # if (wl % 2 == 0):
        #     wl = len(duration_unique) - 1
        # dt = duration_unique[1] - duration_unique[0]
        # detuning = ss.savgol_filter(
        #     phase_unwrapped, 
        #     window_length=wl, 
        #     polyorder=1, 
        #     deriv=1) / dt
        
        # real_detuning = get_real_detuning(detuning, demod_freq, sampling_rate)


        # Di Carlo method
        detuning2 = ss.savgol_filter(
            phase_unwrapped / (2 * np.pi),
            window_length = derivative_window_size,
            polyorder = derivative_order,
            deriv=1) * sampling_rate * 1e9
            
        real_detuning2 = get_real_detuning(detuning2, demod_freq, sampling_rate)

        # Maxime method
        dt = np.diff(duration_unique) * 1e-9
        dphi_dt_unwrap = np.abs(np.diff(phase_unwrapped) / dt)
        detuning3 = dphi_dt_unwrap / (2 * np.pi)

        #mean_detuning.append(np.mean(detuning))
        mean_detuning2.append(np.mean(real_detuning2))
        mean_detuning3.append(np.mean(detuning3))

        #demod_data circle after fft
        # global_title = "FFT demod data circle"
        # title_x = "<X>"
        # title_y = "<Y>"
        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=demod_data.real,
        #         y=demod_data.imag,
        #         name=f"A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        #<X> sin <Y> cos after FFT demod 
        # global_title = "<X> , <Y> after FFT demodulation"
        # title_x = "Flux Pulse duration"
        # title_y = "<X> , <Y>"
        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=demod_data.real,
        #         name=f"<X> fft demod | A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=demod_data.imag,
        #         name=f"<Y> fft demod | A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        #demod_data phase after fft
        # global_title = "Phase vs. Time"
        # title_x = "Flux Pulse duration"
        # title_y = "Phase"
        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=phase,
        #         name=f"Phase | A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        # demod_data phase unwrapped after fft
        global_title = "Phase unwrapped vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Phase unwrapped"
        figs["norm"].add_trace(
            go.Scatter(
                x=duration_unique,
                y=phase_unwrapped,
                name=f"phase unwrapped | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )

        #demod_data phase unwrapped angle after fft
        # title = "Phase unwrapped angle vs. Time"
        # title_x = "Flux Pulse duration"
        # title_y = "Phase unwraped angle" 
        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=phase_unwrapped_angle,
        #         name=f"phase unwrapped angle | A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        # global_title = "Detuning vs. Time"
        # title_x = "Flux Pulse duration"
        # title_y = "Detunning Ramiro" 
        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=detuning,
        #         name=f"A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        # global_title = "Detuning vs. Time"
        # title_x = "Flux Pulse duration"
        # title_y = "Detunning Di Carlo" 
        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=detuning2,
        #         name=f"A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        # global_title = "Detuning vs. Time"
        # title_x = "Flux Pulse duration"
        # title_y = "Detunning Maxime" 
        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=detuning3,
        #         name=f"A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        # global_title = "Detuning vs. Time"
        # title_x = "Flux Pulse duration"
        # title_y = "Real Detunning (Hz)" 
        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=real_detuning,
        #         name=f"A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )

        # figs["norm"].add_trace(
        #     go.Scatter(
        #         x=duration_unique,
        #         y=real_detuning2,
        #         name=f"detunning 2 | A = {amp:.3f}"
        #     ),
        #     row=1,
        #     col=1,
        # )


    # demod_data phase unwrapped angle after fft
    # global_title = "Mean Detuning vs. Amplitude"
    # title_x = "Amplitude (dimensionless)"
    # title_y = "Real Detunning mean (Hz)" 
    # figs["norm"].add_trace(
    #     go.Scatter(
    #         x=amplitude_unique,
    #         y=mean_detuning2,
    #         name=f"Detuning mean Di Carlo"
    #     ),
    #     row=1,
    #     col=1,
    # )

    # #demod_data phase unwrapped angle after fft
    # global_title = "Mean Detuning vs. Amplitude"
    # title_x = "Amplitude (dimensionless)"
    # title_y = "Real Detunning mean (Hz)" 
    # figs["norm"].add_trace(
    #     go.Scatter(
    #         x=amplitude_unique,
    #         y=mean_detuning3,
    #         name=f"Detuning mean Maxime"
    #     ),
    #     row=1,
    #     col=1,
    # )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]


# For cryoscope
def cryoscope_norm_heatmap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"

    # Extracting Data
    x = data.get_values("prob", "dimensionless")[
        data.df["component"] == MX_tag
    ].to_numpy()
    y = data.get_values("prob", "dimensionless")[
        data.df["component"] == MY_tag
    ].to_numpy()
    x = x[: len(y)]
    y = y[: len(y)]

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    x_norm, y_norm = normalize_data(x, y, amplitude_unique, duration_unique, flux_pulse_amplitude, flux_pulse_duration, amplitude)

    # Making figure
    figs = {}

    figs["norm"] = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.3,
        subplot_titles=(
            "<X>",
            "<Y>",
        ),
    )

    figs["norm"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=x_norm,
            colorbar=dict(len=0.46, y=0.75),
        ),
        row=1,
        col=1,
    )

    figs["norm"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=y_norm,
            colorbar=dict(len=0.46, y=0.25),
        ),
        row=1,
        col=2,
    )
    figs["norm"].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (a.u.)",
        xaxis2_title="Pulse duration (ns)",
        yaxis2_title="Amplitude (a.u.)",
        title=f"Normalized data",
    )
    return figs["norm"]


# For cryoscope phase vs. flux pulse duration
def cryoscope_phase_duration(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    # Making figure
    figs = {}
    figs["phase"] = make_subplots(
        rows=2,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.3,
        subplot_titles=(
            "Phase",
            "Phase Unwrapped",
        ),
    )
    for amp in amplitude_unique:

        # # Extracting Data
        x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

        x_norm = x_data - 0.5
        y_norm = y_data - 0.5
        x_norm = x_norm / max(abs(x_norm))
        y_norm = y_norm / max(abs(y_norm))

        phase = np.arctan2(x_norm, y_norm)
        
        # Plot the unwrapped along duration
        phase_unwrapped = np.unwrap(phase, axis=0)

        figs["phase"].add_trace(
            go.Scatter(
                x=flux_pulse_duration,
                y=phase,
                name=f"Phase | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )

        figs["phase"].add_trace(
            go.Scatter(
                x=flux_pulse_duration,
                y=phase_unwrapped,
                name=f"Unwrapped Phase | A = {amp:.3f}"

            ),
            row=2,
            col=1,
        )

    figs["phase"].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Phase (rad)",
        xaxis2_title="Pulse duration (ns)",
        yaxis2_title="Phase unwrapped(rad)",
    )
    return figs["phase"]


# For cryoscope phase vs. flux pulse duration
def cryoscope_phase_duration_heatmap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"

    # # Extracting Data
    x = data.get_values("prob", "dimensionless")[
        data.df["component"] == MX_tag
    ].to_numpy()
    y = data.get_values("prob", "dimensionless")[
        data.df["component"] == MY_tag
    ].to_numpy()
    x = x[: len(y)]
    y = y[: len(y)]

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    # Making figure
    figs = {}
    figs["phase"] = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.3,
        subplot_titles=(
            "Phase",
            "Phase Unwrapped along duration",
        ),
    )

    x_norm, y_norm = normalize_data(x, y, amplitude_unique, duration_unique, flux_pulse_amplitude, flux_pulse_duration, amplitude)
    phase = np.arctan2(x_norm, y_norm)
    
    # Plot the unwrapped along duration
    phase_unwrapped = np.unwrap(phase, axis=1)

    figs["phase"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phase,
        ),
        row=1,
        col=1,
    )

    figs["phase"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phase_unwrapped,
        ),
        row=1,
        col=2,
    )

    figs["phase"].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (dimensionless)",
        xaxis2_title="Pulse duration (ns)",
        yaxis2_title="Amplitude (dimensionless)",
    )
    return figs["phase"]


# For cryoscope phase vs. flux pulse duration
def cryoscope_phase_amplitude_heatmap(folder, routine, qubit, format):
    data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"

    # # Extracting Data
    x = data.get_values("prob", "dimensionless")[
        data.df["component"] == MX_tag
    ].to_numpy()
    y = data.get_values("prob", "dimensionless")[
        data.df["component"] == MY_tag
    ].to_numpy()
    x = x[: len(y)]
    y = y[: len(y)]

    amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
    duration = data.get_values("flux_pulse_duration", "ns")
    flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
    flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
    amplitude_unique = flux_pulse_amplitude[
        flux_pulse_duration == flux_pulse_duration[0]
    ]
    duration_unique = flux_pulse_duration[
        flux_pulse_amplitude == flux_pulse_amplitude[0]
    ]

    # Making figure
    figs = {}
    figs["phase"] = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.3,
        subplot_titles=(
            "Phase",
            "Phase Unwrapped along amplitude",
        ),
    )

    x_norm, y_norm = normalize_data(x, y, amplitude_unique, duration_unique, flux_pulse_amplitude, flux_pulse_duration, amplitude)
    phase = np.arctan2(x_norm, y_norm)
    
    # Plot the unwrapped along duration
    phase_unwrapped = np.unwrap(phase, axis=0)

    figs["phase"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phase,
        ),
        row=1,
        col=1,
    )

    figs["phase"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phase_unwrapped,
        ),
        row=1,
        col=2,
    )

    figs["phase"].update_layout(
        xaxis_title="Pulse duration (ns)",
        yaxis_title="Amplitude (dimensionless)",
        xaxis2_title="Pulse duration (ns)",
        yaxis2_title="Amplitude (dimensionless)",
    )
    return figs["phase"]














# def cryoscope_detunning(folder, routine, qubit, format):
#     data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
#     import numpy as np

#     MX_tag = "MX"
#     MY_tag = "MY"

#     amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")
#     duration = data.get_values("flux_pulse_duration", "ns")
#     flux_pulse_duration = duration[data.df["component"] == MY_tag].to_numpy()
#     flux_pulse_amplitude = amplitude[data.df["component"] == MY_tag].to_numpy()
#     amplitude_unique = flux_pulse_amplitude[
#         flux_pulse_duration == flux_pulse_duration[0]
#     ]
#     duration_unique = flux_pulse_duration[
#         flux_pulse_amplitude == flux_pulse_amplitude[0]
#     ]

#     # Making figure
#     figs = {}
#     figs["detunning"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )
#     for amp in amplitude_unique:

#         # # Extracting Data
#         x = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#         y = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
#         x = x[: len(y)]
#         y = y[: len(y)]

#         phase = np.arctan2(x, y)
#         phase_unwrapped = np.unwrap(phase)

#         wl = len(x) 
#         if (wl % 2 == 0):
#             wl = len(x) - 1
#         dt = flux_pulse_duration[1] - flux_pulse_duration[0]
#         detunning = ss.savgol_filter(phase_unwrapped, window_length=wl, polyorder=1, deriv=1) / dt

#         figs["detunning"].add_trace(
#             go.Scatter(
#                 x=flux_pulse_duration,
#                 y=detunning,
#                 name=f"phase A={amp:.3f}"
#             ),
#             row=1,
#             col=1,
#         )

#     figs["detunning"].update_layout(
#         xaxis_title="Pulse duration (ns)",
#         yaxis_title="Detunning",
#         title="Raw data",
#     )
#     return figs["detunning"]




def normalize_data(x, y, amplitude_unique, duration_unique, flux_pulse_amplitude, flux_pulse_duration, amplitude):
    x_norm = np.ones((len(amplitude_unique), len(duration_unique))) * np.nan
    y_norm = np.ones((len(amplitude_unique), len(duration_unique))) * np.nan

    for i, amp in enumerate(amplitude_unique):
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

    return x_norm, y_norm


def normalize_sincos(x, y, window_size_frac=61, window_size=None, do_envelope=True):

    if window_size is None:
        window_size = len(x) // window_size_frac

        # window size for savgol filter must be odd
        window_size -= (window_size + 1) % 2

    x = ss.savgol_filter(x, window_size, 0, 0)  ## why poly order 0
    y = ss.savgol_filter(y, window_size, 0, 0)

    return x, y  # this is returning just the noise,


def fft_based_freq_guess_complex(y):
    """
    guess the shape of a sinusoidal complex signal y (in multiples of
        sampling rate), by selecting the peak in the fft.
    return guess (f, ph, off, amp) for the model
        y = amp*exp(2pi i f t + ph) + off.
    """
    fft = np.fft.fft(y)[1:len(y)]
    freq_guess_idx = np.argmax(np.abs(fft))
    if freq_guess_idx >= len(y) // 2:
        freq_guess_idx -= len(y)
    freq_guess = 1 / len(y) * (freq_guess_idx + 1)

    phase_guess = np.angle(fft[freq_guess_idx]) + np.pi / 2
    amp_guess = np.absolute(fft[freq_guess_idx]) / len(y)
    offset_guess = np.mean(y)

    return freq_guess, phase_guess, offset_guess, amp_guess

def get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order=0):
    real_detuning = detuning - demod_freq + sampling_rate * nyquist_order
    return real_detuning