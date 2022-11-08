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
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        # this_data = np.array(re_norm) + np.array(1j * im_norm)
        #print (len(complex_data))
        #ws = 20 # Needs to be adjusted for data analysis
        #norm_data = normalize_sincos(complex_data, window_size=ws)

        figs["norm"].add_trace(
            go.Scatter(
                x=flux_pulse_duration,
                y=re_norm,
                name=f"<X> | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )

        figs["norm"].add_trace(
            go.Scatter(
                x=flux_pulse_duration,
                y=im_norm,
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

def cryoscope_norm_heatmap(folder, routine, qubit, format):
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

    signal = []
    for amp in amplitude_unique:
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()

        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re) + np.array(1j*im) #raw data in complex form
        signal.append(this_data)

    signal = np.array(signal,dtype=np.complex128)
    global_title="Normalized Data"
    title_x = "Flux Pulse Duration"
    title_y = "Amplitudes"
    figs["norm"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=np.abs(signal),
            colorbar=dict(len=0.46, y=0.25),
        ),
        row=1,
        col=1,
    )
    
    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]

def cryoscope_fft(folder, routine, qubit, format):
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

    signal = []
    signal_fft = []
    fft_x = np.fft.fftfreq(n=duration_unique.shape[0], d=4e-9) # X axis

    for amp in amplitude_unique:
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re) + np.array(1j*im) #raw data in complex form
        #this_data = np.array(re_norm) + np.array(1j*im_norm) #normalized data in complex form         
        signal.append(this_data)
        signal_fft.append(np.fft.fft(this_data-np.mean(this_data)))

    signal = np.array(signal,dtype=np.complex128)
    signal_fft = np.array(signal_fft,dtype=np.complex128)

   # print(f"{signal_fft.shape}")
    # print(f"{fft_x.shape}")
    # print(f"{amplitude_unique.shape}")

    mask = np.argsort(fft_x)
    # fft_mat = signal_fft
    # for i in range(fft_mat.shape[0]):
    #     fft_mat[i,:] = fft_mat[i,mask]

    # demod_data circle
    global_title = "FFT data"
    title_x = "Frequency (GHz)"
    title_y = "Amplitudes"
    figs["norm"].add_trace(
        go.Heatmap(
            x=fft_x[mask],
            y=amplitude_unique,
            z=np.abs(signal_fft),
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]

def cryoscope_phase(folder, routine, qubit, format):
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
    signal = []
    for amp in amplitude_unique:      
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re) + np.array(1j*im) #raw data in complex form
        signal.append(this_data)

        #print (len(complex_data))
        # ws = 10 # Needs to be adjusted for data analysis
        # norm_data = normalize_sincos(complex_data, window_size=ws)

        # sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
        # derivative_window_length = 7 / sampling_rate
        # derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
        # derivative_window_size += (derivative_window_size + 1) % 2
        # derivative_order = 2
        # nyquist_order=0

        # freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(this_data)
        # demod_freq = - \
        #         freq_guess * sampling_rate
        
        #y = amp*exp(2pi i f t + ph) + off.
        #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
        # demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
        # Di Carlo has smooth here!!!

        # Unwrap phase        
        phase = np.arctan2(this_data.real, this_data.imag)

        # Phase vs. Time
        global_title = "Phase vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Phase"
        figs["norm"].add_trace(
            go.Scatter(
                x=duration_unique,
                y=phase,
                name=f"Phase | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]

def cryoscope_phase_heatmap(folder, routine, qubit, format):
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
    signal = []
    for amp in amplitude_unique:      
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re) + np.array(1j*im) #raw data in complex form
        signal.append(this_data)

    signal = np.array(signal,dtype=np.complex128)
    global_title = "Phase vs. Time"
    title_x = "Flux Pulse duration"
    title_y = "Phase"
    phase = np.arctan2(signal.real, signal.imag)
    figs["norm"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phase,
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]

def cryoscope_phase_unwrapped(folder, routine, qubit, format):
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
    signal = []
    for amp in amplitude_unique:      
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re_norm) + np.array(1j*im_norm) #raw data in complex form
        signal.append(this_data)

        # Unwrap phase        
        phase = np.arctan2(this_data.real, this_data.imag)
        phase_unwrapped = np.unwrap(phase)

        # Phase vs. Time
        global_title = "Phase Unwrapped vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Phase"
        figs["norm"].add_trace(
            go.Scatter(
                x=duration_unique,
                y=phase_unwrapped,
                name=f"Phase | A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]

def cryoscope_phase_unwrapped_heatmap(folder, routine, qubit, format):
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
    signal = []
    for amp in amplitude_unique:      
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re_norm) + np.array(1j*im_norm) #raw data in complex form
        signal.append(this_data)


    signal = np.array(signal,dtype=np.complex128)
    global_title = "Phase unwrapped along amplitude vs. Time"
    title_x = "Flux Pulse duration"
    title_y = "Phase"
    phase = np.arctan2(signal.real, signal.imag)
    phase_unwrapped = np.unwrap(phase, axis=0)
    figs["norm"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phase_unwrapped,
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]

def cryoscope_phase_amplitude_unwrapped_heatmap(folder, routine, qubit, format):
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
    signal = []
    for amp in amplitude_unique:      
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re_norm) + np.array(1j*im_norm) #raw data in complex form
        signal.append(this_data)


    signal = np.array(signal,dtype=np.complex128)
    global_title = "Phase unwrapped along duration vs. Time"
    title_x = "Flux Pulse duration"
    title_y = "Phase"
    phase = np.arctan2(signal.real, signal.imag)
    phase_unwrapped = np.unwrap(phase, axis=1)
    figs["norm"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phase_unwrapped,
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]

def cryoscope_fft_phase_unwrapped(folder, routine, qubit, format):
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

    signal = []
    signal_fft = []
    fft_x = np.fft.fftfreq(n=duration_unique.shape[0], d=4e-9) # X axis

    for amp in amplitude_unique:
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re) + np.array(1j*im) #raw data in complex form
        #this_data = np.array(re_norm) + np.array(1j*im_norm) #normalized data in complex form         
        signal.append(this_data)
        signal_fft.append(np.fft.fft(this_data-np.mean(this_data)))

    signal = np.array(signal,dtype=np.complex128)
    signal_fft = np.array(signal_fft,dtype=np.complex128)

    phase = np.arctan2(signal_fft.real, signal_fft.imag)
    phase_unwrapped = np.unwrap(phase)

    global_title = "Phase unwrapped FFT vs. Time"
    title_x = "Time"
    title_y = "Amplitudes"
    figs["norm"].add_trace(
        go.Heatmap(
            x=duration_unique,
            y=amplitude_unique,
            z=phase_unwrapped,
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]


def cryoscope_detuning_time(folder, routine, qubit, format):
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
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
    )


    sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
    derivative_window_length = 7 / sampling_rate
    derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
    derivative_window_size += (derivative_window_size + 1) % 2
    derivative_order = 2
    nyquist_order=0

    detuning_mean = []
    signal_fft = []
    for amp in amplitude_unique:      
        re = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        im = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        re_norm = re - 0.5
        im_norm = im - 0.5
        re_norm = np.array(re_norm / max(abs(re_norm)))
        im_norm = np.array(im_norm / max(abs(im_norm)))

        this_data = np.array(re_norm) + np.array(1j*im_norm) #raw data in complex form
        #signal.append(this_data)
        #signal_fft = np.fft.fft(this_data-np.mean(this_data))

        # Unwrap phase        
        phase = np.arctan2(this_data.real, this_data.imag)
        # phase = np.arctan2(signal_fft.real, signal_fft.imag)
        phase_unwrapped = np.unwrap(phase) #unwrapped fft signal
        phase_unwrapped = np.unwrap(phase, axis=0) #unwrapped phase along amplitude
        # phase_unwrapped = np.unwrap(phase, axis=1) #unwrapped phase along duration
        

        # use a savitzky golay filter: it take sliding window of length
        # `window_length`, fits a polynomial, returns derivative at
        # middle point
        # phase = phase_unwrapped_data

        # Di Carlo method
        # detuning = ss.savgol_filter(
        #     phase_unwrapped / (2 * np.pi),
        #     window_length = derivative_window_size,
        #     polyorder = derivative_order,
        #     deriv=1) * sampling_rate * 1e9

        # nyquist_order = 0    
        # detuning = get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order)

        # Maxime method
        dt = np.diff(duration_unique) * 1e-9
        dphi_dt_unwrap = np.abs(np.diff(phase_unwrapped) / dt)
        detuning = dphi_dt_unwrap / (2 * np.pi)
        detuning_mean.append(np.mean(detuning))

        # Detunning vs. Time
        global_title = "Detuning vs. Time"
        title_x = "Flux Pulse duration"
        title_y = "Detunning" 
        figs["norm"].add_trace(
            go.Scatter(
                x=duration_unique,
                y=detuning,
                name=f"A = {amp:.3f}"
            ),
            row=1,
            col=1,
        )
    
    # Detunning vs. Time
    global_title = "Detuning vs. amplitude"
    title_x2 = "Amplitude"
    title_y2 = "Detunning" 
    figs["norm"].add_trace(
        go.Scatter(
            x=amplitude_unique,
            y=detuning_mean,
            name=f"A = {amp:.3f}",
        ),
        row=1,
        col=2,
    )


    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        xaxis2_title=title_x2,
        yaxis2_title=title_y2,
        title=f"{global_title}",
    )
    return figs["norm"]

def cryoscope_detuning_amplitude(folder, routine, qubit, format):
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

    for amp in amplitude_unique:
        x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
        y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
        complex_data = x_data + 1j * y_data
        #print (len(complex_data))
        ws = 10 # Needs to be adjusted for data analysis
        norm_data = normalize_sincos(complex_data, window_size=ws)

        sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
        derivative_window_length = 7 / sampling_rate
        derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
        derivative_window_size += (derivative_window_size + 1) % 2
        derivative_order = 2
        nyquist_order=0

        freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
        demod_freq = - \
                freq_guess * sampling_rate
        
        #y = amp*exp(2pi i f t + ph) + off.
        #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
        demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
        # Di Carlo has smooth here!!!


        # Unwrap phase        
        phase = np.arctan2(demod_data.imag, demod_data.real)
        # phase_unwrapped = np.unwrap(phase)
        phase_unwrapped = np.unwrap(np.angle(demod_data))

        # use a savitzky golay filter: it take sliding window of length
        # `window_length`, fits a polynomial, returns derivative at
        # middle point
        # phase = phase_unwrapped_data

        # Di Carlo method
        detuning = ss.savgol_filter(
            phase_unwrapped / (2 * np.pi),
            window_length = derivative_window_size,
            polyorder = derivative_order,
            deriv=1) * sampling_rate * 1e9

        nyquist_order = 0    
        detuning = get_real_detuning(detuning, demod_freq, sampling_rate, nyquist_order)

        # Maxime method
        # dt = np.diff(duration_unique) * 1e-9
        # dphi_dt_unwrap = np.abs(np.diff(phase_unwrapped) / dt)
        # detuning = dphi_dt_unwrap / (2 * np.pi)


        mean_detuning.append(np.mean(detuning))
        
    # Mean detunning vs. amplitude
    global_title = "Mean Detuning vs. Amplitude"
    title_x = "Amplitude (dimensionless)"
    title_y = "Detunning mean (Hz)" 
    figs["norm"].add_trace(
        go.Scatter(
            x=amplitude_unique,
            y=mean_detuning,
            name=f"Detuning"
        ),
        row=1,
        col=1,
    )

    figs["norm"].update_layout(
        xaxis_title=title_x,
        yaxis_title=title_y,
        title=f"{global_title}",
    )
    return figs["norm"]



#Helper functions
def normalize_sincos(
        data,
        window_size_frac=500,
        window_size=None,
        do_envelope=True):

    if window_size is None:
        window_size = len(data) // window_size_frac

        # window size for savgol filter must be odd
        window_size -= (window_size + 1) % 2

    mean_data_r = ss.savgol_filter(data.real, window_size, 0, 0)
    mean_data_i = ss.savgol_filter(data.imag, window_size, 0, 0)

    mean_data = mean_data_r + 1j * mean_data_i

    if do_envelope:
        envelope = np.sqrt(
            ss.savgol_filter(
                (np.abs(
                    data -
                    mean_data))**2,
                window_size,
                0,
                0))
    else:
        envelope = 1
    norm_data = ((data - mean_data) / envelope)
    return norm_data

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

    # if do_envelope:
    #     envelope = np.sqrt(ss.savgol_filter((np.abs(data - mean_data))**2, window_size, 0, 0))
    # else:
    #     envelope = 1

    # norm_data = ((data - mean_data) / envelope)

    return x, y

# def cryoscope_demod_circle(folder, routine, qubit, format):
#     data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
#     import numpy as np
#     import scipy.signal as ss

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
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     for amp in amplitude_unique:
#         x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#         y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
#         complex_data = x_data + 1j * y_data
#         #print (len(complex_data))
#         ws = 10 # Needs to be adjusted for data analysis
#         norm_data = normalize_sincos(complex_data, window_size=ws)

#         sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
#         derivative_window_length = 7 / sampling_rate
#         derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#         derivative_window_size += (derivative_window_size + 1) % 2
#         derivative_order = 2
#         nyquist_order=0

#         freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
#         demod_freq = - \
#                 freq_guess * sampling_rate
        
#         #y = amp*exp(2pi i f t + ph) + off.
#         #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
#         demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
#         # Di Carlo has smooth here!!!

#         # demod_data circle
#         global_title = "FFT demod data circle"
#         title_x = "<X>"
#         title_y = "<Y>"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=demod_data.real,
#                 y=demod_data.imag,
#                 name=f"A = {amp:.3f}"
#             ),
#             row=1,
#             col=1,
#         )
    
#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     return figs["norm"]

# def cryoscope_demod_fft(folder, routine, qubit, format):
#     data = DataUnits.load_data(folder, "data", routine, format, f"data_q{qubit}")
#     import numpy as np
#     import scipy.signal as ss

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
#     figs["norm"] = make_subplots(
#         rows=1,
#         cols=1,
#         horizontal_spacing=0.1,
#         vertical_spacing=0.2,
#     )

#     for amp in amplitude_unique:
#         x_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MX_tag].to_numpy()
#         y_data = data.get_values("prob", "dimensionless")[data.df["flux_pulse_amplitude"] == amp][data.df["component"] == MY_tag].to_numpy()
        
#         complex_data = x_data + 1j * y_data
#         #print (len(complex_data))
#         ws = 10 # Needs to be adjusted for data analysis
#         norm_data = normalize_sincos(complex_data, window_size=ws)

#         sampling_rate = 1 / (duration_unique[1] - duration_unique[0])
#         derivative_window_length = 7 / sampling_rate
#         derivative_window_size = max(3, int(derivative_window_length * sampling_rate))
#         derivative_window_size += (derivative_window_size + 1) % 2
#         derivative_order = 2
#         nyquist_order=0

#         freq_guess, phase_guess, offset_guess, amp_guess = fft_based_freq_guess_complex(norm_data)
#         demod_freq = - \
#                 freq_guess * sampling_rate
        
#         #y = amp*exp(2pi i f t + ph) + off.
#         #demod_data = amp_guess * np.exp(2 * np.pi * 1j * demod_freq * duration_unique + phase_guess) + offset_guess
#         demod_data = np.exp(2 * np.pi * 1j * duration_unique * demod_freq) * norm_data
#         # Di Carlo has smooth here!!!

#         # <X> sin <Y> cos demod data 
#         global_title = "<X> , <Y> after FFT demodulation"
#         title_x = "Flux Pulse duration"
#         title_y = "<X> , <Y>"
#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=duration_unique,
#                 y=demod_data.real,
#                 name=f"<X> fft demod | A = {amp:.3f}"
#             ),
#             row=1,
#             col=1,
#         )

#         figs["norm"].add_trace(
#             go.Scatter(
#                 x=duration_unique,
#                 y=demod_data.imag,
#                 name=f"<Y> fft demod | A = {amp:.3f}"
#             ),
#             row=1,
#             col=1,
#         )
        
#     figs["norm"].update_layout(
#         xaxis_title=title_x,
#         yaxis_title=title_y,
#         title=f"{global_title}",
#     )
#     return figs["norm"]
