# -*- coding: utf-8 -*-
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy
import scipy.signal as ss
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

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
    figs += [go.Figure()]  # 6 for the reconstructed pulse

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
        )  # TODO: continue
        a_b_fits[i, :] = np.append(a_fit, b_fit)
        pulses[i, :] = awg_reconstructed_pulse

    figs[6].data[-1].visible = True
    steps = []
    for i in range(len(figs[6].data)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(figs[6].data)],
            label=str(amplitude_unique[i]),
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
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

    # best "a" and "b" parameters that explain how ideal_awg_pulse (what we sent) got distorted into awg_reconstructed_pulse (what we mesured)

    # freq_shift_median = np.median(freq_shift)
    # freq_shift_array = np.concatenate((freq_shift_array, freq_shift))
    # freq_shift_median_array = np.concatenate(
    #     (freq_shift_median_array, np.array([freq_shift_median]))
    # )  # This is the data sought: average freq shifts for a given flux pulse amplitude
    # flux_pulse_duration_array = np.concatenate(
    #     (flux_pulse_duration_array, flux_pulse_duration)
    # )
    # flux_pulse_amplitude_array = np.concatenate(
    #     (flux_pulse_amplitude_array, [amplitude] * len(freq_shift))
    # )

    # print(amplitudes, freq_shift_median_array)

    return figs


"""    ####################################################################################

    freq_shift_array = np.array([])
    freq_shift_median_array = np.array([])
    flux_pulse_duration_array = np.array([])
    flux_pulse_amplitude_array = np.array([])
    amplitudes = data.get_values("flux_pulse_amplitude", "dimensionless")[
        data.df["component"] == MY_tag
    ].unique()

    for amplitude in amplitudes:
        z = data.get_values("prob", "dimensionless")[data.df["component"] == MZ_tag][
            data.df["flux_pulse_amplitude"] == amplitude
        ].to_numpy()
        x = data.get_values("prob", "dimensionless")[data.df["component"] == MX_tag][
            data.df["flux_pulse_amplitude"] == amplitude
        ].to_numpy()
        y = data.get_values("prob", "dimensionless")[data.df["component"] == MY_tag][
            data.df["flux_pulse_amplitude"] == amplitude
        ].to_numpy()
        x = x[: len(z)]
        y = y[: len(z)]
        flux_pulse_duration = data.get_values("flux_pulse_duration", "ns")[
            data.df["component"] == MZ_tag
        ][data.df["flux_pulse_amplitude"] == amplitude].to_numpy()
        phi = np.arctan2((y - np.mean(z)), (x - np.mean(z)))
        phi = np.unwrap(phi)
        dt = np.diff(flux_pulse_duration)
        dphi_dt = np.diff(phi) / dt * 1e9
        freq_shift = dphi_dt / (2 * np.pi)
        flux_pulse_duration = flux_pulse_duration[1:]
        freq_shift_median = np.median(freq_shift)
        freq_shift_array = np.concatenate((freq_shift_array, freq_shift))
        freq_shift_median_array = np.concatenate(
            (freq_shift_median_array, np.array([freq_shift_median]))
        )  # This is the data sought: average freq shifts for a given flux pulse amplitude
        flux_pulse_duration_array = np.concatenate(
            (flux_pulse_duration_array, flux_pulse_duration)
        )
        flux_pulse_amplitude_array = np.concatenate(
            (flux_pulse_amplitude_array, [amplitude] * len(freq_shift))
        )

    print(amplitudes, freq_shift_median_array)

    fig.add_trace(
        go.Heatmap(
            x=flux_pulse_duration_array,
            y=flux_pulse_amplitude_array,
            z=freq_shift_array,
            colorbar=dict(len=0.46, y=0.5),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="flux pulse duration (ns)",
        yaxis_title="flux pulse amplitude (dimensionless)",
        xaxis2_title="flux pulse duration (ns)",
        yaxis2_title="flux pulse amplitude (dimensionless)",
    )

    ################################  smoothing  ###################################

    flux_pulse_duration = data.get_values('flux_pulse_duration', 'ns')[data.df['component'] == MZ_tag].to_numpy()
    flux_pulse_amplitude = data.get_values('flux_pulse_amplitude', 'dimensionless')[data.df['component'] == MZ_tag].to_numpy()

    freq_shift_array = np.array([])
    freq_shift_median_array = np.array([])
    flux_pulse_duration_array = np.array([])
    flux_pulse_amplitude_array = np.array([])
    amplitudes = data.get_values('flux_pulse_amplitude', 'dimensionless')[data.df['component'] == MY_tag].unique()
    for amplitude in amplitudes:
        z = data.get_values('MSR', 'uV')[data.df['component'] == MZ_tag][data.df['flux_pulse_amplitude'] == amplitude].to_numpy()
        x = data.get_values('MSR', 'uV')[data.df['component'] == MX_tag][data.df['flux_pulse_amplitude'] == amplitude].to_numpy()
        y = data.get_values('MSR', 'uV')[data.df['component'] == MY_tag][data.df['flux_pulse_amplitude'] == amplitude].to_numpy()
        x = x[:len(z)]
        y = y[:len(z)]
        flux_pulse_duration = data.get_values('flux_pulse_duration', 'ns')[data.df['component'] == MZ_tag][data.df['flux_pulse_amplitude'] == amplitude].to_numpy()
        phi = np.arctan2((y-np.mean(z)), (x-np.mean(z)))
        phi = np.unwrap(phi)

        wl = len(flux_pulse_duration)//4
        if (wl % 2 == 0):
            wl -= 1
        dt = np.diff(flux_pulse_duration)
        dphi_dt = ss.savgol_filter(phi, window_length=wl, polyorder=1, deriv=1)[1:] / dt * 1e9

        freq_shift = dphi_dt / (2 * np.pi)
        flux_pulse_duration = flux_pulse_duration[1:]
        freq_shift_median = np.median(freq_shift)

        freq_shift_array = np.concatenate((freq_shift_array, freq_shift))
        freq_shift_median_array = np.concatenate((freq_shift_median_array, np.array([freq_shift_median]))) # This is the data sought: average freq shifts for a given flux pulse amplitude
        flux_pulse_duration_array = np.concatenate((flux_pulse_duration_array, flux_pulse_duration))
        flux_pulse_amplitude_array = np.concatenate((flux_pulse_amplitude_array, [amplitude] * len(freq_shift)))

    print("Smoothing", amplitudes, freq_shift_median_array)

    return fig"""


"""
    ######################################################################
    # STEP 1: run cryosocpe analysis simulation for different awg_pulse amplitudes
    # and obtaine mesured detunning due to your experimental setup
    ######################################################################

    awg_duration_time = np.arange(300, 400, 0.1)
    detunning_res = []
    amplitude_values = []

    cte = 1
    awg_amplitudes = np.arange(0.1, 1, 0.1)
    ideal_detunning_values = cte * (2 * np.pi * awg_amplitudes**2)

    for A_pulso in awg_amplitudes:
        f = cte * (
            A_pulso**2
        )  # Only for simulation. In real data the modification of the awg amplitude produces the frequency changes in the qubit
        ideal_detunning = 2 * np.pi * f

        # Real model Data
        # data, awg_duration_range = ds.run_cryoscope(qubit)

        # Data model simulation
        data = sincos_model_real_imag(awg_duration_time, f)
        median_detunning, detuning_measured_for_filter = cryoscope_analysis(
            data, awg_duration_time, ideal_detunning, norm_window_size=61
        )
        detunning_res.append(median_detunning)
        amplitude_values.append(A_pulso)

    # Plot obtained detunning for each awg pulse amplitude
    plt.plot(amplitude_values, detunning_res, "o", color="C4")
    plt.plot(awg_amplitudes, ideal_detunning_values, "--", color="C4")
    plt.title("Detunning vs. Amplitude")
    plt.legend()
    plt.xlabel("Amplitude (a.u.)")
    plt.ylabel("Detunning")
    plt.show()

    ###############################################################################
    # STEP 2: run cryosocpe analysis simulation for obe awg_pulse with amplitude A
    # Expirementalist should choose best amplitude A from the iteration on
    # step 1
    ###############################################################################

    A = 0.5
    # Only for simulation. In real data the modification of the awg amplitude produces
    # the frequency changes in the qubit
    f = cte * (A**2)
    ideal_detunning = 2 * np.pi * f

    # Real model Data
    # data, awg_duration_range = ds.run_cryoscope(qubit)

    # Data model simulation
    data = sincos_model_real_imag(awg_duration_time, f)
    mean_detunning, detuning_measured_for_filter = cryoscope_analysis(
        data, awg_duration_time, ideal_detunning, norm_window_size=61
    )

    # Obtaining reconstructed pulse
    awg_reconstructed_pulse = np.interp(
        detuning_measured_for_filter, detunning_res, amplitude_values
    )

    # Ideal awg waveform sent to qubit
    ideal_awg_pulse = A * np.ones(awg_duration_time.shape)

    # best "a" and "b" parameters that explain how ideal_awg_pulse (what we sent) got distorted into awg_reconstructed_pulse (what we mesured)
    b_fit, a_fit = cryoscope_fit(
        awg_duration_time, ideal_awg_pulse, awg_reconstructed_pulse
    )

    print(awg_reconstructed_pulse)
    print(ideal_awg_pulse)
    print(b_fit)
    print(a_fit)

    plt.plot(
        awg_duration_time,
        awg_reconstructed_pulse,
        "o",
        label="Reconstructed AWG Flux Pulse",
        color="C4",
    )
    plt.plot(
        awg_duration_time,
        ideal_awg_pulse,
        "--",
        label="Ideal AWG Flux Pulse",
        color="C5",
    )
    # plt.ylim(0.5)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Awg flux pulses")
    plt.show()

"""


def normalize_sincos(x, y, window_size_frac=61, window_size=None, do_envelope=True):

    if window_size is None:
        window_size = len(x) // window_size_frac

        # window size for savgol filter must be odd
        window_size -= (window_size + 1) % 2

    x = ss.savgol_filter(x, window_size, 0, 0)  ## why poly order 0
    y = ss.savgol_filter(y, window_size, 0, 0)

    return x, y  # this is returning just the noise,


def cryoscope_analysis(data, duration_time, ideal_detunning, norm_window_size):
    # plot raw cryoscope results
    plt.title("Raw cryoscope data")
    plt.plot(duration_time, data.real, ".-", label="Re", color="C0")
    plt.plot(duration_time, data.imag, ".-", label="Im", color="C1")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude (a.u.)")
    plt.show()

    norm_data = normalize_sincos(data, window_size=norm_window_size)
    # norm_data = data

    plt.title("Normalized cryoscope data")
    plt.plot(duration_time, norm_data.real, ".-", label="Re", color="C0")
    plt.plot(duration_time, norm_data.imag, ".-", label="Im", color="C1")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude (a.u.)")
    plt.show()

    # plot phasor doing arctan2
    phase_data = np.arctan2(norm_data.imag, norm_data.real)
    plt.plot(duration_time, phase_data, ".-", label="Phase", color="C0")
    plt.title("Phase vs. Time")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Phase")
    plt.show()

    # plot unwrapped phase
    phase_unwrapped_data = np.unwrap(phase_data)
    plt.plot(duration_time, phase_unwrapped_data, ".-", label="Phase", color="C1")
    plt.title("Unwrapped Phase vs. Time")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Unwrapped Phase")
    plt.show()

    # use a savitzky golay filter: it take sliding window of length
    # `window_length`, fits a polynomial, returns derivative at
    # middle point
    phase = phase_unwrapped_data
    wl = len(duration_time)
    if wl % 2 == 0:
        wl = len(duration_time) - 1
    dt = duration_time[1] - duration_time[0]
    detunning = ss.savgol_filter(phase, window_length=wl, polyorder=1, deriv=1) / dt
    plt.plot(duration_time, detunning, "-o", label="Detunning", color="C2")
    plt.axhline(ideal_detunning, linestyle="--", label="Ideal Detunning", color="black")
    plt.title("Detunning vs. Time")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Detunning")
    plt.show()

    median_detunning = np.median(detunning)

    return median_detunning, detunning


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
    from scipy.signal import lfilter

    b_fit = p[:2]
    a_fit = p[2:]
    return lfilter(b_fit, a_fit, awg_reconstructed_pulse)


# For cryoscope
def cryoscope_raw_slider(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"
    MZ_tag = "MZ"

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

    flux_pulse_duration = data.get_values("flux_pulse_duration", "ns")[
        data.df["component"] == MZ_tag
    ].to_numpy()
    flux_pulse_amplitude_unique = data.get_values(
        "flux_pulse_amplitude", "dimensionless"
    )[data.df["component"] == MZ_tag].to_numpy()

    phi = np.arctan2(y, x)
    # phi = np.unwrap(phi)

    fig = go.Figure()

    probs = [
        data.get_values("prob", "dimensionless")[
            data.df["component"] == MZ_tag
        ].to_numpy(),
        data.get_values("prob", "dimensionless")[
            data.df["component"] == MX_tag
        ].to_numpy(),
        data.get_values("prob", "dimensionless")[
            data.df["component"] == MY_tag
        ].to_numpy(),
    ]

    flux_pulse_amplitude = [
        data.get_values("flux_pulse_amplitude", "dimensionless")[
            data.df["component"] == MZ_tag
        ].to_numpy(),
        data.get_values("flux_pulse_amplitude", "dimensionless")[
            data.df["component"] == MX_tag
        ].to_numpy(),
        data.get_values("flux_pulse_amplitude", "dimensionless")[
            data.df["component"] == MY_tag
        ].to_numpy(),
    ]

    for i in range(len(flux_pulse_amplitude_unique)):
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=flux_pulse_duration,
                y=probs[0][flux_pulse_amplitude[0] == flux_pulse_amplitude_unique[i]],
                name=f"z _ {flux_pulse_amplitude_unique[i]}",
            ),
        )
    for i in range(len(flux_pulse_amplitude_unique)):
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=flux_pulse_duration,
                y=probs[1][flux_pulse_amplitude[1] == flux_pulse_amplitude_unique[i]],
                name=f"x _ {flux_pulse_amplitude_unique[i]}",
            ),
        )
    for i in range(len(flux_pulse_amplitude_unique)):
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=flux_pulse_duration,
                y=probs[2][flux_pulse_amplitude[2] == flux_pulse_amplitude_unique[i]],
                name=f"y _ {flux_pulse_amplitude_unique[i]}",
            ),
        )
    # fig.data[-1].visible = True

    # Create and add slider
    steps = []
    for i in range(len(flux_pulse_amplitude_unique)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        step["args"][1][i + len(flux_pulse_amplitude_unique)] = True
        step["args"][1][i + 2 * len(flux_pulse_amplitude_unique)] = True
        steps.append(step)
    sliders = [dict(steps=steps)]
    fig.layout.update(sliders=sliders)

    return fig


def cryoscope_phase_slider(folder, routine, qubit, format):
    data = Dataset.load_data(folder, routine, format, f"data_q{qubit}")
    import numpy as np

    MX_tag = "MX"
    MY_tag = "MY"
    MZ_tag = "MZ"

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

    flux_pulse_duration = data.get_values("flux_pulse_duration", "ns")[
        data.df["component"] == MZ_tag
    ].to_numpy()
    flux_pulse_amplitude_unique = data.get_values(
        "flux_pulse_amplitude", "dimensionless"
    )[data.df["component"] == MZ_tag].to_numpy()

    phi = np.arctan2(y, x)
    # phi = np.unwrap(phi)

    fig = go.Figure()

    probs = [
        data.get_values("prob", "dimensionless")[
            data.df["component"] == MZ_tag
        ].to_numpy(),
        data.get_values("prob", "dimensionless")[
            data.df["component"] == MX_tag
        ].to_numpy(),
        data.get_values("prob", "dimensionless")[
            data.df["component"] == MY_tag
        ].to_numpy(),
    ]

    flux_pulse_amplitude = [
        data.get_values("flux_pulse_amplitude", "dimensionless")[
            data.df["component"] == MZ_tag
        ].to_numpy(),
        data.get_values("flux_pulse_amplitude", "dimensionless")[
            data.df["component"] == MX_tag
        ].to_numpy(),
        data.get_values("flux_pulse_amplitude", "dimensionless")[
            data.df["component"] == MY_tag
        ].to_numpy(),
    ]

    for i in range(len(flux_pulse_amplitude_unique)):
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=flux_pulse_duration,
                y=probs[0][flux_pulse_amplitude[0] == flux_pulse_amplitude_unique[i]],
                name=f"z _ {flux_pulse_amplitude_unique[i]}",
            ),
        )
    for i in range(len(flux_pulse_amplitude_unique)):
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=flux_pulse_duration,
                y=probs[1][flux_pulse_amplitude[1] == flux_pulse_amplitude_unique[i]],
                name=f"x _ {flux_pulse_amplitude_unique[i]}",
            ),
        )
    for i in range(len(flux_pulse_amplitude_unique)):
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=flux_pulse_duration,
                y=probs[2][flux_pulse_amplitude[2] == flux_pulse_amplitude_unique[i]],
                name=f"y _ {flux_pulse_amplitude_unique[i]}",
            ),
        )
    # fig.data[-1].visible = True

    # Create and add slider
    steps = []
    for i in range(len(flux_pulse_amplitude_unique)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
            label=str(i),
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        step["args"][1][i + len(flux_pulse_amplitude_unique)] = True
        step["args"][1][i + 2 * len(flux_pulse_amplitude_unique)] = True
        steps.append(step)
    sliders = [
        dict(steps=steps, active=10, currentvalue={"prefix": "Amplitude (a.u.)"})
    ]
    fig.update_layout(sliders=sliders)

    return fig
