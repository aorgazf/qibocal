import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.calibrations.characterization.utils import variable_resolution_scanrange
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__fast_precision)
def resonator_spectroscopy(
    platform: AbstractPlatform,
    qubit: int,
    lowres_width,
    lowres_step,
    highres_width,
    highres_step,
    precision_width,
    precision_step,
    software_averages,
    points=10,
):

    platform.reload_settings()
    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        variable_resolution_scanrange(
            lowres_width, lowres_step, highres_width, highres_step
        )
        + resonator_frequency
    )
    fast_sweep_data = DataUnits(
        name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield fast_sweep_data
                yield lorentzian_fit(
                    fast_sweep_data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                )

            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            fast_sweep_data.add(results)
            count += 1
    yield fast_sweep_data

    if platform.resonator_type == "3D":
        resonator_frequency = fast_sweep_data.get_values("frequency", "Hz")[
            np.argmax(fast_sweep_data.get_values("MSR", "V"))
        ]
        avg_voltage = (
            np.mean(
                fast_sweep_data.get_values("MSR", "V")[: (lowres_width // lowres_step)]
            )
            * 1e6
        )
    else:
        resonator_frequency = fast_sweep_data.get_values("frequency", "Hz")[
            np.argmin(fast_sweep_data.get_values("MSR", "V"))
        ]
        avg_voltage = (
            np.mean(
                fast_sweep_data.get_values("MSR", "V")[: (lowres_width // lowres_step)]
            )
            * 1e6
        )

    precision_sweep__data = DataUnits(
        name=f"precision_sweep_q{qubit}", quantities={"frequency": "Hz"}
    )
    freqrange = (
        np.arange(-precision_width, precision_width, precision_step)
        + resonator_frequency
    )

    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0 and count > 0:
                yield precision_sweep__data
                yield lorentzian_fit(
                    fast_sweep_data + precision_sweep__data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                )

            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            precision_sweep__data.add(results)
            count += 1
    yield precision_sweep__data


@plot("Frequency vs Attenuation", plots.frequency_attenuation_msr_phase)
@plot("MSR vs Frequency", plots.frequency_attenuation_msr_phase__cut)
def resonator_punchout(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    min_att,
    max_att,
    step_att,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "attenuation": "dB"}
    )
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence = PulseSequence()
    sequence.add(ro_pulse)

    # TODO: move this explicit instruction to the platform
    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step)
        + resonator_frequency
        - (freq_width / 4)
    )
    attenuation_range = np.flip(np.arange(min_att, max_att, step_att))
    count = 0
    for _ in range(software_averages):
        for att in attenuation_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.ro_port[qubit].attenuation = att
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr * (np.exp(att / 10)),
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "attenuation[dB]": att,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data


@plot("MSR and Phase vs Flux Current", plots.frequency_flux_msr_phase)
def resonator_spectroscopy_flux(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    current_max,
    current_min,
    current_step,
    software_averages,
    fluxline=0,
    points=10,
):
    platform.reload_settings()

    if fluxline == "qubit":
        fluxline = qubit

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "current": "A"}
    )

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    qubit_biasing_current = platform.characterization["single_qubit"][qubit][
        "sweetspot"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )
    current_range = (
        np.arange(current_min, current_max, current_step) + qubit_biasing_current
    )

    count = 0
    for _ in range(software_averages):
        for curr in current_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.qf_port[fluxline].current = curr
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "current[A]": curr,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    platform.qf_port[fluxline].current = 0
    yield data
    # TODO: automatically extract the sweet spot current
    # TODO: add a method to generate the matrix


@plot("MSR row 1 and Phase row 2", plots.frequency_flux_msr_phase__matrix)
def resonator_spectroscopy_flux_matrix(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    current_min,
    current_max,
    current_step,
    fluxlines,
    software_averages,
    points=10,
):
    platform.reload_settings()

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )
    current_range = np.arange(current_min, current_max, current_step)

    count = 0

    if fluxlines == "diag":
        fluxlines = [qubit]
    elif fluxlines == "all":
        fluxlines = range(platform.nqubits)
    elif fluxlines == "outer":
        fluxlines = np.arange(platform.nqubits)
        fluxlines = fluxlines[fluxlines != qubit]

    for fluxline in fluxlines:
        data = DataUnits(
            name=f"data_q{qubit}_f{fluxline}",
            quantities={"frequency": "Hz", "current": "A"},
        )
        for _ in range(software_averages):
            for curr in current_range:
                for freq in frequency_range:
                    if count % points == 0:
                        yield data
                    platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                    platform.qf_port[fluxline].current = curr
                    msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                        ro_pulse.serial
                    ]
                    results = {
                        "MSR[V]": msr,
                        "i[V]": i,
                        "q[V]": q,
                        "phase[rad]": phase,
                        "frequency[Hz]": freq,
                        "current[A]": curr,
                    }
                    # TODO: implement normalization
                    data.add(results)
                    count += 1

    yield data


@plot("MSR and Phase vs Frequency", plots.dispersive_frequency_msr_phase)
def dispersive_shift(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    software_averages,
    points=10,
):

    platform.reload_settings()

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )

    data_spec = DataUnits(name=f"data_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield data_spec
                yield lorentzian_fit(
                    data_spec,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                )
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data_spec.add(results)
            count += 1
    yield data_spec

    # Shifted Spectroscopy
    sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX_pulse.finish)
    sequence.add(RX_pulse)
    sequence.add(ro_pulse)

    data_shifted = DataUnits(
        name=f"data_shifted_q{qubit}", quantities={"frequency": "Hz"}
    )
    count = 0
    for _ in range(software_averages):
        for freq in frequency_range:
            if count % points == 0 and count > 0:
                yield data_shifted
                yield lorentzian_fit(
                    data_shifted,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["resonator_freq", "peak_voltage"],
                    fit_file_name="fit_shifted",
                )
            platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data_shifted.add(results)
            count += 1
    yield data_shifted


@plot("MSR and Phase vs Frequency", plots.frequency_flux_offset_msr_phase)
def resonator_flux_offset(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    flux_amp_min,
    flux_amp_max,
    flux_amp_step,
    software_averages,
    points=10,
):
    platform.reload_settings()

    sequence = PulseSequence()
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]

    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step) + resonator_frequency
    )
    flux_range = np.arange(flux_amp_min, flux_amp_max, flux_amp_step)

    count = 0

    data = DataUnits(
        name=f"data_q{qubit}",
        quantities={"frequency": "Hz", "offset": "dimensionless"},
    )
    for _ in range(software_averages):
        for flux in flux_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.qb_port[qubit].offset = flux
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "offset[dimensionless]": flux,
                }
                data.add(results)
                count += 1

    yield data


@plot("MSR vs Frequency", plots.frequency_gain_msr_phase)
def resonator_gain(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    min_gain,
    max_gain,
    step_gain,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(
        name=f"data_q{qubit}", quantities={"frequency": "Hz", "gain": "dimensionless"}
    )
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence = PulseSequence()
    sequence.add(ro_pulse)

    # TODO: move this explicit instruction to the platform
    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step)
        + resonator_frequency
        - (freq_width / 4)
    )
    gain_range = np.flip(np.arange(min_gain, max_gain, step_gain))
    count = 0
    for _ in range(software_averages):
        for gain in gain_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                # TODO: move these explicit instructions to the platform
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                platform.ro_port[qubit].gain = gain
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr / gain,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "gain[dimensionless]": gain,
                }
                # TODO: implement normalization
                data.add(results)
                count += 1

    yield data


# Create a similar function like resonator_gain but using the ro_pulse.amplitude instead of the gain. So renaming everything to amplitude and amp
@plot("MSR vs Frequency", plots.frequency_amplitude_msr_phase)
def resonator_amplitude(
    platform: AbstractPlatform,
    qubit: int,
    freq_width,
    freq_step,
    min_amp,
    max_amp,
    step_amp,
    software_averages,
    points=10,
):
    platform.reload_settings()

    data = DataUnits(
        name=f"data_q{qubit}",
        quantities={"frequency": "Hz", "amplitude": "dimensionless"},
    )
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=0)
    sequence = PulseSequence()
    sequence.add(ro_pulse)

    resonator_frequency = platform.characterization["single_qubit"][qubit][
        "resonator_freq"
    ]
    frequency_range = (
        np.arange(-freq_width, freq_width, freq_step)
        + resonator_frequency
        - (freq_width / 4)
    )
    amp_range = np.flip(np.arange(min_amp, max_amp, step_amp))
    count = 0
    for _ in range(software_averages):
        for amp in amp_range:
            for freq in frequency_range:
                if count % points == 0:
                    yield data
                platform.ro_port[qubit].lo_frequency = freq - ro_pulse.frequency
                ro_pulse.amplitude = amp
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "frequency[Hz]": freq,
                    "amplitude[dimensionless]": amp,
                }
                data.add(results)
                count += 1
    yield data
