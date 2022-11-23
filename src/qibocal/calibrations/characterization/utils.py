import glob
import os
import pathlib

import numpy as np
from qibo.config import raise_error
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from sympy import Q

from qibocal.data import DataUnits


def variable_resolution_scanrange(
    lowres_width, lowres_step, highres_width, highres_step
):
    """Helper function for sweeps."""
    return np.concatenate(
        (
            np.arange(-lowres_width, -highres_width, lowres_step),
            np.arange(-highres_width, highres_width, highres_step),
            np.arange(highres_width, lowres_width, lowres_step),
        )
    )


def get_latest_datafolder(path=None):
    if path is None:
        cwd = pathlib.Path()
    else:
        cwd = path
    list_dir = sorted(
        glob.glob(os.path.join(cwd, "*/")), key=os.path.getmtime, reverse=True
    )
    for i in range(len(list_dir)):
        if os.path.isdir(cwd / list_dir[-i] / "data"):
            return cwd / list_dir[-i]


def get_fidelity(
    platform: AbstractPlatform,
    qubit,
    niter,
    param=None,
    save=True,
    amplitude_ro_pulse=0.9,
    amplitude_qd_pulse=None,
):
    """
    Returns the read-out fidelity for the measurement.
    Param:
    platform: Qibolab platform for QPU
    qubit: Qubit number under investigation
    niter: number of iterations
    param: name and units of the varied parameters to save the data in a dictionary format {"name[PintUnit]": vals, ...}
    save: bool to save the data or not
    optional parameters for designed routines: #FIXME: find a better way to do it, extracting pulses from sequence?
        amplitude_ro_pulse = 0.9 #Not 1 for a reason I forgot
    Returns:
    fidelity: float C [0,1]
    """
    if save:
        if param is None:
            raise_error(
                ValueError,
                "Please provide the varied parameters in a dict of QCVV type",
            )
        path = get_latest_datafolder() / "data_param"
        os.makedirs(path, exist_ok=True)

    platform.qrm[qubit].ports[
        "i1"
    ].hardware_demod_en = True  # binning only works with hardware demodulation enabled
    # create exc sequence
    exc_sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.duration)
    ro_pulse.amplitude = amplitude_ro_pulse
    if amplitude_qd_pulse is not None:
        RX_pulse.amplitude = amplitude_qd_pulse
    exc_sequence.add(RX_pulse)
    exc_sequence.add(ro_pulse)

    param_dict = {}
    for key in param:
        param_dict[key.split("[")[0]] = key.split("[")[1].replace("]", "")
    quantities = {"iteration": "dimensionless"}
    quantities.update(param_dict)
    data_exc = DataUnits(name=f"data_exc_{param}_q{qubit}", quantities=quantities)
    msr, phase, i, q = platform.execute_pulse_sequence(exc_sequence, nshots=niter)[
        "binned_integrated"
    ][ro_pulse.serial]

    for n in np.arange(niter):
        results = {
            "MSR[V]": msr[n],
            "i[V]": i[n],
            "q[V]": q[n],
            "phase[rad]": phase[n],
            "iteration[dimensionless]": n,
        }
        results.update(param)
        data_exc.add(results)
    if save:
        data_exc.to_csv(path)

    gnd_sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    ro_pulse.amplitude = amplitude_ro_pulse
    gnd_sequence.add(ro_pulse)

    data_gnd = DataUnits(name=f"data_gnd_{param}_q{qubit}", quantities=quantities)

    shots_results = platform.execute_pulse_sequence(exc_sequence, nshots=niter)[
        "binned_integrated"
    ][ro_pulse.serial]

    for n in np.arange(niter):
        msr, phase, i, q = shots_results[n]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "iteration[dimensionless]": n,
        }
        results.update(param)
        data_gnd.add(results)
    if save:
        data_gnd.to_csv(path)

    real_values_exc, real_values_gnd = rotate_to_distribution(data_exc, data_gnd)

    real_values_combined = real_values_exc + real_values_gnd
    real_values_combined.sort()

    cum_distribution_exc = [
        sum(map(lambda x: x.real >= real_value, real_values_exc))
        for real_value in real_values_combined
    ]
    cum_distribution_gnd = [
        sum(map(lambda x: x.real >= real_value, real_values_gnd))
        for real_value in real_values_combined
    ]
    cum_distribution_diff = np.abs(
        np.array(cum_distribution_exc) - np.array(cum_distribution_gnd)
    )
    argmax = np.argmax(cum_distribution_diff)
    threshold = real_values_combined[argmax]
    errors_exc = niter - cum_distribution_exc[argmax]
    errors_gnd = cum_distribution_gnd[argmax]
    fidelity = cum_distribution_diff[argmax] / niter
    return fidelity


def rotate_to_distribution(data_exc, data_gnd):
    iq_exc = (
        data_exc.get_values("i", "V").to_numpy()
        + 1j * data_exc.get_values("q", "V").to_numpy()
    )
    iq_gnd = (
        data_gnd.get_values("i", "V").to_numpy()
        + 1j * data_gnd.get_values("q", "V").to_numpy()
    )

    origin = np.mean(iq_gnd)
    iq_gnd = iq_gnd - origin
    iq_exc = iq_exc - origin
    angle = np.angle(np.mean(iq_exc))
    iq_exc = iq_exc * np.exp(-1j * angle) + origin
    iq_gnd = iq_gnd * np.exp(-1j * angle) + origin

    return np.real(iq_exc), np.real(iq_gnd)


def iq_to_prob(i, q, mean_gnd, mean_exc):
    state = i + 1j * q
    state = state - mean_gnd
    mean_exc = mean_exc - mean_gnd
    state = state * np.exp(-1j * np.angle(mean_exc))
    mean_exc = mean_exc * np.exp(-1j * np.angle(mean_exc))
    return np.real(state) / np.real(mean_exc)


def snr(signal, noise):
    """Signal to Noise Ratio to detect peaks and valleys."""
    return 20 * np.log(signal / noise)


def choose_freq(freq, span, resolution):
    """Choose a new frequency gaussianly distributed around initial one.

    Args:
        freq (float): frequency we sample around from.
        span (float): search space we sample from.
        resolution (int): number of points for search space resolution.

    Returns:
        freq+ff (float): new frequency sampled gaussianly around old value.

    """
    g = np.random.normal(0, span / 10, 1)
    f = np.linspace(-span / 2, span / 2, resolution)
    for ff in f:
        if g <= ff:
            break
    return freq + ff


def get_noise(background, platform, ro_pulse, qubit, sequence):
    """Measure the MSR for the background noise at different points and average the results.

    Args:
        background (list): frequencies where no feature should be found.
        platform ():
        ro_pulse (): Used in order to execute the pulse sequence with the right parameters in the right qubit.
        qubit (int): TODO: might be useful to make this parameters implicit and not given.
        sequence ():

    Returns:
        noise (float): Averaged MSR value for the different background frequencies.

    """
    noise = 0
    for b_freq in background:
        platform.ro_port[qubit].lo_frequency = b_freq - ro_pulse.frequency
        res = platform.execute_pulse_sequence(sequence, 1024)
        msr, phase, i, q = platform.execute_pulse_sequence(sequence)[ro_pulse.serial]
        noise += msr
    return noise / len(background)


def plot_flux(currs, freqs, qubit, fluxline):
    """Quick plot for the flux vs. current calibration.
    TODO: Move to plots with the new qq_live architecture.

    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    plt.plot(currs, freqs)
    plt.savefig(f"flux_q{qubit}_f{fluxline}.png", bbox_inches="tight")


def plot_punchout(atts, freqs, qubit):
    """Quick plot for the atts vs. freqs calibration.
    TODO: Move to plots with the new qq_live architecture.

    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    plt.plot(freqs, atts)
    plt.savefig(f"qubit_{qubit}_punchout.png", bbox_inches="tight")
