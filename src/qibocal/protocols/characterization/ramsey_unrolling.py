import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

from qibocal.auto.operation import Qubits, Routine

from .ramsey import RamseyData, RamseyParameters, _fit, _plot


def _acquisition(
    params: RamseyParameters,
    platform: Platform,
    qubits: Qubits,
) -> RamseyData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    sequences = []
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    freqs = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit,
            start=RX90_pulses1[qubit].finish,
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        freqs[qubit] = qubits[qubit].drive_frequency
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = RamseyData(
        n_osc=params.n_osc,
        t_max=params.delay_between_pulses_end,
        detuning_sign=+1,
        qubit_freqs=freqs,
    )

    # sweep the parameter
    for wait in waits:
        for qubit in qubits:
            RX90_pulses2[qubit].start = RX90_pulses1[qubit].finish + wait
            ro_pulses[qubit].start = RX90_pulses2[qubit].finish
            if params.n_osc != 0:
                RX90_pulses2[qubit].relative_phase = (
                    RX90_pulses2[qubit].start
                    * (-2 * np.pi)
                    * (params.n_osc)
                    / params.delay_between_pulses_end
                )
        sequences.append(sequence.copy())

    # execute the pulse sequence
    results = platform.execute_pulse_sequences(
        sequences,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
    )
    print(sequences, results)
    # FIXME: Multiplex
    for sequence, wait in zip(sequences, waits):
        print(sequence, sequence.ro_pulses)
        for qubit in qubits:
            print("PPPP ", results[sequence.ro_pulses[qubit]][0])
        for ro_pulse in sequence.ro_pulses:
            # for qubit in qubits:
            print(results)
            result = results[ro_pulse.serial][0]
            qubit = ro_pulse.qubit
            data.register_qubit(
                qubit, wait=wait, msr=result.magnitude, phase=result.phase
            )

    return data


ramsey_unrolling = Routine(_acquisition, _fit, _plot)
"""Ramsey Routine object."""
