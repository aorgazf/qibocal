from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine

from .resonator_flux_dependence import ResonatorFluxData
from .utils import flux_dependence_plot


# TODO: implement cross-talk
@dataclass
class QubitFluxParameters(Parameters):
    freq_width: int
    freq_step: int
    bias_width: float
    bias_step: float
    nshots: int
    relaxation_time: int
    drive_amplitude: float


@dataclass
class QubitFluxResults(Results):
    sweetspot: Dict[List[Tuple], str] = field(metadata=dict(update="sweetspot"))
    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="drive_frequency"))
    fitted_parameters: Dict[List[Tuple], List]


class QubitFluxData(ResonatorFluxData):
    ...


def _acquisition(
    params: QubitFluxParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> QubitFluxData:
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )
        qd_pulses[qubit].amplitude = params.drive_amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    bias_sweeper = Sweeper(
        Parameter.bias, delta_bias_range, qubits=list(qubits.values())
    )
    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and flux bias
    data = QubitFluxData()

    # repeat the experiment as many times as defined by software_averages
    results = platform.sweep(
        sequence,
        bias_sweeper,
        freq_sweeper,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]

        biases = np.repeat(delta_bias_range, len(delta_frequency_range))
        freqs = np.array(
            len(delta_bias_range)
            * list(delta_frequency_range + ro_pulses[qubit].frequency)
        ).flatten()
        # store the results
        r = {k: v.ravel() for k, v in result.raw.items()}
        r.update(
            {
                "frequency[Hz]": freqs,
                "bias[V]": biases,
                "qubit": len(freqs) * [qubit],
            }
        )
        data.add_data_from_dict(r)

    return data


def _fit(data: QubitFluxData) -> QubitFluxResults:
    return QubitFluxResults({}, {}, {})


def _plot(data: QubitFluxData, fit: QubitFluxResults, qubit):
    return flux_dependence_plot(data, fit, qubit)


qubit_flux = Routine(_acquisition, _fit, _plot)
