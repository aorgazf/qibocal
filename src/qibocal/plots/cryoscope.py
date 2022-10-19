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

    MX_tag = "MX"
    MY_tag = "MY"
    MZ_tag = "MZ"

    z = data.get_values("MSR", "uV")[data.df["component"] == MZ_tag].to_numpy()
    x = data.get_values("MSR", "uV")[data.df["component"] == MX_tag].to_numpy()
    y = data.get_values("MSR", "uV")[data.df["component"] == MY_tag].to_numpy()
    x = x[: len(z)]
    y = y[: len(z)]

    flux_pulse_duration = data.get_values("flux_pulse_duration", "ns")[
        data.df["component"] == MZ_tag
    ].to_numpy()
    flux_pulse_amplitude = data.get_values("flux_pulse_amplitude", "dimensionless")[
        data.df["component"] == MZ_tag
    ].to_numpy()
    phi = np.arctan2((y - np.mean(z)), (x - np.mean(z)))
    # phi = np.unwrap(phi)

    fig = make_subplots(
        rows=3,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "z rotation (unwrapped) (rad)",
            "frequency shift (Hz)",
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

    fig.add_trace(
        go.Scatter(
            x=flux_pulse_duration,
            y=data.get_values("MSR", "uV")[data.df["component"] == MZ_tag][
                data.df["flux_pulse_amplitude"] == flux_pulse_amplitude[-1]
            ].to_numpy(),
            name="z",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=flux_pulse_duration,
            y=data.get_values("MSR", "uV")[data.df["component"] == MX_tag][
                data.df["flux_pulse_amplitude"] == flux_pulse_amplitude[-1]
            ].to_numpy(),
            name="x",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=flux_pulse_duration,
            y=data.get_values("MSR", "uV")[data.df["component"] == MY_tag][
                data.df["flux_pulse_amplitude"] == flux_pulse_amplitude[-1]
            ].to_numpy(),
            name="y",
        ),
        row=1,
        col=2,
    )

    ####################################################################################

    freq_shift_array = np.array([])
    freq_shift_median_array = np.array([])
    flux_pulse_duration_array = np.array([])
    flux_pulse_amplitude_array = np.array([])
    amplitudes = data.get_values("flux_pulse_amplitude", "dimensionless")[
        data.df["component"] == MY_tag
    ].unique()

    for amplitude in amplitudes:
        z = data.get_values("MSR", "uV")[data.df["component"] == MZ_tag][
            data.df["flux_pulse_amplitude"] == amplitude
        ].to_numpy()
        x = data.get_values("MSR", "uV")[data.df["component"] == MX_tag][
            data.df["flux_pulse_amplitude"] == amplitude
        ].to_numpy()
        y = data.get_values("MSR", "uV")[data.df["component"] == MY_tag][
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
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=flux_pulse_duration,
            y=freq_shift,
            name="frequency shift",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="flux pulse duration (ns)",
        yaxis_title="flux pulse amplitude (dimensionless)",
        xaxis2_title="flux pulse duration (ns)",
        yaxis2_title="flux pulse amplitude (dimensionless)",
    )

    #################################  smoothing  ###################################

    # flux_pulse_duration = data.get_values('flux_pulse_duration', 'ns')[data.df['component'] == MZ_tag].to_numpy()
    # flux_pulse_amplitude = data.get_values('flux_pulse_amplitude', 'dimensionless')[data.df['component'] == MZ_tag].to_numpy()

    # freq_shift_array = np.array([])
    # freq_shift_median_array = np.array([])
    # flux_pulse_duration_array = np.array([])
    # flux_pulse_amplitude_array = np.array([])
    # amplitudes = data.get_values('flux_pulse_amplitude', 'dimensionless')[data.df['component'] == MY_tag].unique()
    # for amplitude in amplitudes:
    #     z = data.get_values('MSR', 'uV')[data.df['component'] == MZ_tag][data.df['flux_pulse_amplitude'] == amplitude].to_numpy()
    #     x = data.get_values('MSR', 'uV')[data.df['component'] == MX_tag][data.df['flux_pulse_amplitude'] == amplitude].to_numpy()
    #     y = data.get_values('MSR', 'uV')[data.df['component'] == MY_tag][data.df['flux_pulse_amplitude'] == amplitude].to_numpy()
    #     x = x[:len(z)]
    #     y = y[:len(z)]
    #     flux_pulse_duration = data.get_values('flux_pulse_duration', 'ns')[data.df['component'] == MZ_tag][data.df['flux_pulse_amplitude'] == amplitude].to_numpy()
    #     phi = np.arctan2((y-np.mean(z)), (x-np.mean(z)))
    #     phi = np.unwrap(phi)

    #     wl = len(flux_pulse_duration)//4
    #     if (wl % 2 == 0):
    #         wl -= 1
    #     dt = np.diff(flux_pulse_duration)
    #     dphi_dt = ss.savgol_filter(phi, window_length=wl, polyorder=1, deriv=1)[1:] / dt * 1e9

    #     freq_shift = dphi_dt / (2 * np.pi)
    #     flux_pulse_duration = flux_pulse_duration[1:]
    #     freq_shift_median = np.median(freq_shift)

    #     freq_shift_array = np.concatenate((freq_shift_array, freq_shift))
    #     freq_shift_median_array = np.concatenate((freq_shift_median_array, np.array([freq_shift_median]))) # This is the data sought: average freq shifts for a given flux pulse amplitude
    #     flux_pulse_duration_array = np.concatenate((flux_pulse_duration_array, flux_pulse_duration))
    #     flux_pulse_amplitude_array = np.concatenate((flux_pulse_amplitude_array, [amplitude] * len(freq_shift)))

    # print("Smoothing", amplitudes, freq_shift_median_array)

    return fig

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
    plt.xlabel("Amplitude")
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


import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
from scipy.optimize import curve_fit


def normalize_sincos(data, window_size_frac=61, window_size=None, do_envelope=True):

    if window_size is None:
        window_size = len(data) // window_size_frac

        # window size for savgol filter must be odd
        window_size -= (window_size + 1) % 2

    mean_data_r = ss.savgol_filter(data.real, window_size, 0, 0)  ## why poly order 0
    mean_data_i = ss.savgol_filter(data.imag, window_size, 0, 0)

    mean_data = (
        mean_data_r + 1j * mean_data_i
    )  # should not this be mean_data = np.mean(mean_data_r + 1j * mean_data_i) ?

    if do_envelope:
        envelope = np.sqrt(
            ss.savgol_filter((np.abs(data - mean_data)) ** 2, window_size, 0, 0)
        )
    else:
        envelope = 1

    norm_data = (data - mean_data) / envelope
    return norm_data  # this is returning just the noise,


def cryoscope_analysis(data, duration_time, ideal_detunning, norm_window_size):
    # plot raw cryoscope results
    plt.title("Raw cryoscope data")
    plt.plot(duration_time, data.real, ".-", label="Re", color="C0")
    plt.plot(duration_time, data.imag, ".-", label="Im", color="C1")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

    norm_data = normalize_sincos(data, window_size=norm_window_size)
    # norm_data = data

    plt.title("Normalized cryoscope data")
    plt.plot(duration_time, norm_data.real, ".-", label="Re", color="C0")
    plt.plot(duration_time, norm_data.imag, ".-", label="Im", color="C1")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
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
    popt, pcov = curve_fit(
        lambda t, *p: cryoscope_step_response(t, p, awg_reconstructed_pulse),
        x_data,
        y_data,
        p0=pguess,
    )
    b_fit = popt[[0, 1]]
    a_fit = popt[[2, 3]]
    return b_fit, a_fit


def cryoscope_step_response(t, p, awg_reconstructed_pulse):
    # https://arxiv.org/pdf/1907.04818.pdf (page 11 - filter formula S22)
    # p = [b0 = 1−k +k ·α, b1 = −(1−k)·(1−α),a0 = 1 and a1 = −(1−α)]
    # p = [b0, b1, a0, a1]
    from scipy.signal import lfilter

    b_fit = p[:2]
    a_fit = p[2:]
    return lfilter(b_fit, a_fit, awg_reconstructed_pulse)
