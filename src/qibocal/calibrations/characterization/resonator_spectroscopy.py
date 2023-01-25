import numpy as np
from qibo.config import log
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Sweeper

from qibocal import plots
from qibocal.config import raise_error
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Resonator Frequency", plots.frequency_msr_phase)
def resonator_spectroscopy(
    platform: AbstractPlatform,
    qubits: list,
    fast_width,
    fast_step,
    precision_width,
    precision_step,
    wait_time,
    nshots=1024,
    software_averages=1,
):
    r"""
    Perform spectroscopies on the qubits' readout resonators.
    This routine executes an initial scan around the expected resonator frequency indicated
    in the platform runcard. After that, a final sweep with more precision is executed centered in the new
    resonator frequency found.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        fast_width (int): Width frequency in HZ to perform the high resolution sweep
        fast_step (int): Step frequency in HZ for the high resolution sweep
        precision_width (int): Width frequency in HZ to perform the precision resolution sweep
        precision_step (int): Step frequency in HZ for the precission resolution sweep
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - Two DataUnits objects with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **readout_frequency**: frequency
            - **peak_voltage**: peak voltage
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
            - **qubit**: The qubit being tested
    """
    # reload instrument settings from runcard
    platform.reload_settings()
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range resonator frequency
    delta_frequency_range = np.arange(-fast_width // 2, fast_width // 2, fast_step)
    sweeper = Sweeper(
        "frequency",
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
        wait_time=wait_time,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency
    fast_sweep_data = DataUnits(
        name="fast_sweep_data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        results = platform.sweep(sequence, sweeper, nshots=nshots)

        while (
            any(result.in_progress for result in results.values())
            or len(fast_sweep_data) == 0
        ):
            # retrieve the results for every qubit
            for qubit in qubits:
                # average msr, phase, i and q over the number of shots defined in the runcard
                result = results[ro_pulses[qubit].serial]
                # store the results
                r = result.to_dict(average=False)
                r.update(
                    {
                        "frequency[Hz]": delta_frequency_range
                        + ro_pulses[qubit].frequency,
                        "qubit": len(delta_frequency_range) * [qubit],
                        "iteration": len(delta_frequency_range) * [iteration],
                    }
                )
                fast_sweep_data.add_data_from_dict(r)

            # save data as often as defined by points
            if result.in_progress:
                # save data
                yield fast_sweep_data
                # calculate and save fit
                yield lorentzian_fit(
                    fast_sweep_data,
                    x="frequency[Hz]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["resonator_freq", "peak_voltage"],
                )

    # finally, save the remaining data and fits
    yield fast_sweep_data
    yield lorentzian_fit(
        fast_sweep_data,
        x="frequency[Hz]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["readout_frequency", "peak_voltage"],
    )

    # store max/min peaks as new frequencies
    for qubit in qubits:
        qubit_data = (
            fast_sweep_data.df[fast_sweep_data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby("frequency", as_index=False)
            .mean()
        )
        if platform.resonator_type == "3D":
            qubits[qubit].readout_frequency = (
                qubit_data["frequency"][
                    np.argmax(qubit_data["MSR"].pint.to("V").pint.magnitude)
                ]
                .to("Hz")
                .magnitude
            )
        else:
            qubits[qubit].readout_frequency = (
                qubit_data["frequency"][
                    np.argmin(qubit_data["MSR"].pint.to("V").pint.magnitude)
                ]
                .to("Hz")
                .magnitude
            )

    # run a precision sweep around the newly detected frequencies

    # TODO: remove it
    # create a new sequence of pulses with adjusted frequencies
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    delta_frequency_range = np.arange(
        -precision_width // 2, precision_width // 2, precision_step
    )
    sweeper = Sweeper(
        "frequency",
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
        wait_time=wait_time,
    )

    # create a second DataUnits object to store the results,
    precision_sweep_data = DataUnits(
        name="precision_sweep_data",
        quantities={"frequency": "Hz"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        results = platform.sweep(sequence, sweeper, nshots=nshots)

        while (
            any(result.in_progress for result in results.values())
            or len(precision_sweep_data) == 0
        ):
            # retrieve the results for every qubit
            for qubit, ro_pulse in ro_pulses.items():
                # average msr, phase, i and q over the number of shots defined in the runcard
                result = results[ro_pulse.serial]
                r = result.to_dict(average=False)
                # store the results
                r.update(
                    {
                        "frequency[Hz]": delta_frequency_range + ro_pulse.frequency,
                        "qubit": len(delta_frequency_range) * [qubit],
                        "iteration": len(delta_frequency_range) * [iteration],
                    }
                )
                precision_sweep_data.add_data_from_dict(r)

            # save data as often as defined by points
            if result.in_progress:
                # save data
                yield precision_sweep_data
                # calculate and save fit
                yield lorentzian_fit(
                    precision_sweep_data,
                    x="frequency[Hz]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["resonator_freq", "peak_voltage"],
                )

    # finally, save the remaining data and fits
    yield precision_sweep_data
    yield lorentzian_fit(
        precision_sweep_data,
        x="frequency[Hz]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["readout_frequency", "peak_voltage"],
    )


@plot(
    "MSR and Phase vs Resonator Frequency and Attenuation",
    plots.frequency_attenuation_msr_phase,
)
@plot(
    "Cross section at half range attenuation", plots.frequency_attenuation_msr_phase_cut
)
def resonator_punchout(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    min_att,
    max_att,
    step_att,
    wait_time,
    nshots=1024,
    software_averages=1,
):
    r"""
    Perform spectroscopies on the qubits' readout resonators, decreasing the attenuation applied to
    the read-out pulse, producing an increment of the power sent to the resonator.
    That shows the two regimes of a given resonator, low and high-power regimes.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        freq_width (int): Width frequency in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequency in HZ for the spectroscopy sweep
        min_att (int): Minimum value in db for the attenuation sweep
        max_att (int): Minimum value in db for the attenuation sweep
        step_att (int): Step attenuation in db for the attenuation sweep
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz
            - **attenuation[dB]**: attenuation value in db applied to the flux line
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()

    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    # resonator frequency
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    freq_sweeper = Sweeper(
        "frequency",
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
        wait_time=wait_time,
    )

    # attenuation
    # TODO: Rename attenuation to amplitude?
    attenuation_range = np.arange(min_att, max_att, step_att)
    att_sweeper = Sweeper(
        "amplitude",
        attenuation_range,
        [ro_pulses[qubit] for qubit in qubits],
        wait_time=0,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and attenuation
    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "attenuation": "dB"},
        options=["qubit", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    atts = np.repeat(attenuation_range, len(delta_frequency_range))
    for iteration in range(software_averages):
        results = platform.sweep(sequence, att_sweeper, freq_sweeper, nshots=nshots)

        while any(result.in_progress for result in results.values()) or len(data) == 0:
            # retrieve the results for every qubit
            for qubit, ro_pulse in ro_pulses.items():
                # average msr, phase, i and q over the number of shots defined in the runcard
                result = results[ro_pulse.serial]
                # store the results
                freqs = np.array(
                    len(attenuation_range)
                    * list(delta_frequency_range + ro_pulse.frequency)
                ).flatten()
                r = result.to_dict()
                r.update(
                    {
                        "frequency[Hz]": freqs,
                        "attenuation[dB]": atts,
                        "qubit": len(freqs) * [qubit],
                        "iteration": len(freqs) * [iteration],
                    }
                )
                data.add_data_from_dict(r)

            # save data
            yield data
            # TODO: calculate and save fit

    # finally, save the remaining data and fits
    yield data


@plot(
    "MSR and Phase vs Resonator Frequency and Flux Current",
    plots.frequency_flux_msr_phase,
)
def resonator_spectroscopy_flux(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    bias_width,
    bias_step,
    fluxlines,
    wait_time,
    nshots=1024,
    software_averages=1,
):
    r"""
    Perform spectroscopy on the readout resonator modifying the current applied in the flux control line.
    This routine works for quantum devices flux controlled.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        freq_width (int): Width frequency in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequency in HZ for the spectroscopy sweep
        current_width (float): Width current in A for the flux current sweep
        current_step (float): Step current in A for the flux current sweep
        fluxlines (list): List of flux lines to use to perform the experiment. If it is set to "qubits", it uses each of
                        flux lines associated with the target qubits.
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz
            - **current[A]**: Current value in A applied to the flux line
            - **qubit**: The qubit being tested
            - **fluxline**: The fluxline being tested
            - **iteration**: The iteration number of the many determined by software_averages
    """
    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)
    freq_sweeper = Sweeper(
        "frequency",
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
        wait_time=wait_time,
    )

    # flux current
    if fluxlines == "qubits":
        fluxlines = qubits

    delta_bias_range = np.arange(-bias_width / 2, bias_width / 2, bias_step)
    bias_sweeper = Sweeper("offset", delta_bias_range, qubits=fluxlines, wait_time=0)

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and flux current
    data = DataUnits(
        name=f"data",
        quantities={"frequency": "Hz", "current": "A"},
        options=["qubit", "fluxline", "iteration"],
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        results = platform.sweep(sequence, bias_sweeper, freq_sweeper, nshots=nshots)

        while any(result.in_progress for result in results.values()) or len(data) == 0:
            # retrieve the results for every qubit
            for qubit, fluxline in zip(qubits, fluxlines):
                # TODO: Support more fluxlines

                result = results[ro_pulses[qubit].serial]

                biases = (
                    np.repeat(delta_bias_range, len(delta_frequency_range))
                    + platform.qubits[fluxline].flux.offset
                )
                freqs = np.array(
                    len(delta_bias_range)
                    * list(delta_frequency_range + ro_pulses[qubit].frequency)
                ).flatten()
                # store the results
                r = result.to_dict()
                r.update(
                    {
                        "frequency[Hz]": freqs,
                        "current[A]": biases,
                        "qubit": len(freqs) * [qubit],
                        "fluxline": len(freqs) * [fluxline],
                        "iteration": len(freqs) * [iteration],
                    }
                )
                data.add_data_from_dict(r)

            # save data
            yield data

    # finally, save the remaining data and fits
    yield data


@plot("MSR and Phase vs Resonator Frequency", plots.dispersive_frequency_msr_phase)
def dispersive_shift(
    platform: AbstractPlatform,
    qubits: list,
    freq_width,
    freq_step,
    software_averages=1,
    points=10,
):
    r"""
    Perform spectroscopy on the readout resonator, with the qubit in ground and excited state, showing
    the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        freq_width (int): Width frequency in HZ to perform the spectroscopy sweep
        freq_step (int): Step frequency in HZ for the spectroscopy sweep
        software_averages (int): Number of executions of the routine for averaging results
        fluxlines (list): List of flux control lines associated to different qubits to sweep current
        points (int): Save data results in a file every number of points

    Returns:
        A DataUnits object with the raw data obtained for the normal and shifted sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **frequency[Hz]**: Resonator frequency value in Hz
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create 2 sequences of pulses for the experiment:
    # sequence_0: I  - MZ
    # sequence_1: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence_0.add(ro_pulses[qubit])
        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(-freq_width // 2, freq_width // 2, freq_step)

    # create a DataUnits objects to store the results
    data_0 = DataUnits(
        name=f"data_0", quantities={"frequency": "Hz"}, options=["qubit", "iteration"]
    )
    data_1 = DataUnits(
        name=f"data_1", quantities={"frequency": "Hz"}, options=["qubit", "iteration"]
    )

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(software_averages):
        # sweep the parameter
        for delta_freq in delta_frequency_range:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data_0
                yield data_1
                # calculate and save fit
                yield lorentzian_fit(
                    data=data_0,
                    x="frequency[Hz]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["readout_frequency", "peak_voltage"],
                    fit_file_name="fit_data_0",
                )
                yield lorentzian_fit(
                    data=data_1,
                    x="frequency[Hz]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["readout_frequency_shifted", "peak_voltage"],
                    fit_file_name="fit_data_1",
                )

            # reconfigure the instruments based on the new resonator frequency
            # in this case setting the local oscillators
            # the pulse sequence does not need to be modified or recreated between executions
            for qubit in qubits:
                ro_pulses[qubit].frequency = (
                    delta_freq + qubits[qubit].readout_frequency
                )

            # execute the pulse sequences
            results_0 = platform.execute_pulse_sequence(sequence_0)
            results_1 = platform.execute_pulse_sequence(sequence_1)

            # retrieve the results for every qubit
            for data, results in list(zip([data_0, data_1], [results_0, results_1])):
                for ro_pulse in ro_pulses.values():
                    # average msr, phase, i and q over the number of shots defined in the runcard
                    r = results[ro_pulse.serial].to_dict()
                    # store the results
                    r.update(
                        {
                            "frequency[Hz]": ro_pulses[qubit].frequency,
                            "qubit": ro_pulse.qubit,
                            "iteration": iteration,
                        }
                    )
                    data.add(r)
            count += 1
    # finally, save the remaining data and fits
    yield data_0
    yield data_1
    yield lorentzian_fit(
        data=data_0,
        x="frequency[Hz]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["readout_frequency", "peak_voltage"],
        fit_file_name="fit_data_0",
    )
    yield lorentzian_fit(
        data=data_1,
        x="frequency[Hz]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["readout_frequency_shifted", "peak_voltage"],
        fit_file_name="fit_data_1",
    )
