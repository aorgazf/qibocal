import numpy as np
from qibolab.pulses import FluxPulse
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qm.qua import *


def baked_waveform(config, highfreq, waveform, durations):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in durations:
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op("flux_pulse", f"flux{highfreq}", wf)
            b.play("flux_pulse", f"flux{highfreq}")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


@plot("Chevron CZ", plots.duration_amplitude_msr_flux_pulse)
@plot("Chevron CZ - I", plots.duration_amplitude_I_flux_pulse)
@plot("Chevron CZ - Q", plots.duration_amplitude_Q_flux_pulse)
def tune_transition(
    platform,
    qubits: dict,
    flux_pulse_duration_start,
    flux_pulse_duration_end,
    flux_pulse_duration_step,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    dt=1,
    nshots=1024,
    relaxation_time=None,
):
    """Perform a Chevron-style plot for the flux pulse designed to apply a CZ (CPhase) gate.
    This experiment probes the |11> to i|02> transition by preparing the |11> state with
    pi-pulses, applying a flux pulse to the high frequency qubit to engage its 1 -> 2 transition
    with varying interaction duration and amplitude. We then measure both the high and low frequency qubit.

    We aim to find the spot where the transition goes from |11> -> i|02> -> -|11>.

    Args:
        platform: platform where the experiment is meant to be run.
        qubit (int): qubit that will interact with center qubit 2.
        flux_pulse_duration_start (int): minimum flux pulse duration in nanoseconds.
        flux_pulse_duration_end (int): maximum flux pulse duration in nanoseconds.
        flux_pulse_duration_step (int): step for the duration sweep in nanoseconds.
        flux_pulse_amplitude_start (float): minimum flux pulse amplitude.
        flux_pulse_amplitude_end (float): maximum flux pulse amplitude.
        flux_pulse_amplitude_step (float): step for the amplitude sweep.
        dt (int): time delay between the two flux pulses if enabled.

    Returns:
        data (DataSet): Measurement data for both the high and low frequency qubits.

    """
    if len(qubits) > 1:
        raise NotImplementedError

    qubit = list(qubits.keys())[0]

    platform.reload_settings()

    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    initialize_lowfreq = platform.create_RX_pulse(lowfreq, start=0, relative_phase=0)
    initialize_highfreq = platform.create_RX_pulse(highfreq, start=0, relative_phase=0)

    flux_sequence, _ = platform.create_CZ_pulse_sequence(
        (highfreq, lowfreq), start=initialize_lowfreq.finish
    )
    # find flux pulse that is targeting the ``highfreq`` qubit
    flux_pulse = next(
        iter(
            pulse
            for pulse in flux_sequence
            if isinstance(pulse, FluxPulse) and pulse.qubit == highfreq
        )
    )

    measure_lowfreq = platform.create_qubit_readout_pulse(
        lowfreq, start=flux_sequence.finish
    )
    measure_highfreq = platform.create_qubit_readout_pulse(
        highfreq, start=flux_sequence.finish
    )

    sequence = (
        initialize_lowfreq
        + initialize_highfreq
        + flux_sequence
        + measure_lowfreq
        + measure_highfreq
    )

    amplitudes = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    durations = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )
    data = DataUnits(
        name=f"data_q{lowfreq}{highfreq}",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
        options=["q_freq"],
    )

    opx = platform.design.instruments[0]
    qmsequence = opx.create_qmsequence(platform.qubits, sequence)

    qm_measure_lowfreq = qmsequence.pulse_to_qmpulse[measure_lowfreq.serial]
    qm_measure_highfreq = qmsequence.pulse_to_qmpulse[measure_highfreq.serial]

    # FLux pulse waveform generation
    flux_waveform = np.array([flux_pulse.amplitude] * durations[-1])
    # Baked flux pulse segments
    config = opx.config.__dict__
    square_pulse_segments = baked_waveform(config, highfreq, flux_waveform, durations)

    with program() as experiment:
        n = declare(int)
        a = declare(fixed)  # Flux pulse amplitude
        segment = declare(int)  # Flux pulse segment

        for qmpulse in qmsequence.ro_pulses:
            # threshold = platform.qubits[qmpulse.pulse.qubit].threshold
            # iq_angle = platform.qubits[qmpulse.pulse.qubit].iq_angle
            # qmpulse.declare_output(threshold, iq_angle)
            qmpulse.declare_output()

        with for_(n, 0, n < nshots, n + 1):
            with for_(*from_array(a, amplitudes)):
                with for_(*from_array(segment, durations)):
                    align()
                    # CZ 02-11 protocol
                    # Play pi on both qubits
                    play(initialize_lowfreq.serial, f"drive{lowfreq}")
                    play(initialize_highfreq.serial, f"drive{highfreq}")
                    # global align
                    # align()
                    wait(initialize_highfreq.duration // 4, f"flux{highfreq}")
                    # Wait some additional time to be sure that the pulses don't overlap, this can be calibrated
                    # wait(20)
                    # Play flux pulse with 1ns resolution
                    with switch_(segment):
                        for baked_segment, dur in zip(square_pulse_segments, durations):
                            with case_(dur):
                                baked_segment.run(amp_array=[(f"flux{highfreq}", a)])
                                wait(
                                    (initialize_lowfreq.duration + dur) // 4 + 1,
                                    f"readout{lowfreq}",
                                )
                                wait(
                                    (initialize_highfreq.duration + dur) // 4 + 1,
                                    f"readout{highfreq}",
                                )
                    # global align
                    # align()
                    # Wait some additional time to be sure that the pulses don't overlap, this can be calibrated
                    # wait(20)
                    # q0 state readout
                    for qmpulse in qmsequence.ro_pulses:
                        acquisition = qmpulse.acquisition
                        measure(
                            qmpulse.operation,
                            qmpulse.element,
                            None,
                            dual_demod.full(
                                "cos", "out1", "sin", "out2", acquisition.I
                            ),
                            dual_demod.full(
                                "minus_sin", "out1", "cos", "out2", acquisition.Q
                            ),
                        )
                        acquisition.classify_shots()
                    for qmpulse in qmsequence.ro_pulses:
                        qmpulse.acquisition.save()
                    # Cooldown to have the qubit in the ground state
                    if relaxation_time > 0:
                        wait(relaxation_time)

        with stream_processing():
            for qmpulse in qmsequence.ro_pulses:
                acquisition = qmpulse.acquisition
                serial = qmpulse.pulse.serial
                acquisition.I_stream.buffer(len(durations)).buffer(
                    len(amplitudes)
                ).average().save(f"{serial}_I")
                acquisition.Q_stream.buffer(len(durations)).buffer(
                    len(amplitudes)
                ).average().save(f"{serial}_Q")

    results = opx.execute_program(experiment)

    amps = np.repeat(amplitudes, len(durations))
    durs = np.array(len(amplitudes) * list(durations)).flatten()

    res_temp = results[measure_lowfreq.serial].raw
    res_temp.update(
        {
            "flux_pulse_duration[ns]": durs,
            "flux_pulse_amplitude[dimensionless]": amps,
            "q_freq": len(amps) * ["low"],
        }
    )
    data.add_data_from_dict(res_temp)

    res_temp = results[measure_highfreq.serial].raw
    res_temp.update(
        {
            "flux_pulse_duration[ns]": durs,
            "flux_pulse_amplitude[dimensionless]": amps,
            "q_freq": len(amps) * ["high"],
        }
    )
    data.add_data_from_dict(res_temp)
    yield data
