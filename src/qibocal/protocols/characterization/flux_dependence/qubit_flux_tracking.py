from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Qubits, Routine
from qibocal.config import raise_error

from ..qubit_spectroscopy_ef import DEFAULT_ANHARMONICITY
from . import qubit_flux_dependence, utils


@dataclass
class QubitFluxTrackParameters(qubit_flux_dependence.QubitFluxParameters):
    """QubitFluxTrack runcard inputs."""


@dataclass
class QubitFluxTrackResults(qubit_flux_dependence.QubitFluxParameters):
    """QubitFluxTrack outputs."""


@dataclass
class QubitFluxTrackData(qubit_flux_dependence.QubitFluxData):
    """QubitFluxTrack acquisition outputs."""

    def register_qubit_track(self, qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        # to be able to handle the 1D sweeper case
        size = len(freq)
        ar = np.empty(size, dtype=qubit_flux_dependence.QubitFluxType)
        ar["freq"] = freq
        ar["bias"] = [bias] * size
        ar["signal"] = signal
        ar["phase"] = phase
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: QubitFluxTrackResults,
    platform: Platform,
    qubits: Qubits,
) -> QubitFluxTrackData:
    """Data acquisition for QubitFlux Experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )

        if params.transition == "02":
            if qubits[qubit].anharmonicity != 0:
                qd_pulses[qubit].frequency -= qubits[qubit].anharmonicity / 2
            else:
                qd_pulses[qubit].frequency -= DEFAULT_ANHARMONICITY / 2

        if params.drive_amplitude is not None:
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

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )

    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    data = QubitFluxTrackData(resonator_type=platform.resonator_type)

    for bias in delta_bias_range:
        for qubit in qubits:
            try:
                freq_resonator = utils.transmon_readout_frequency_diagonal(
                    bias,
                    qubits[qubit].drive_frequency,
                    qubits[qubit].asymmetry,
                    qubits[qubit].crosstalk_matrix[qubit],
                    qubits[qubit].sweetspot,
                    qubits[qubit].bare_resonator_frequency,
                    qubits[qubit].g,
                )
                # modify qubit resonator frequency
                qubits[qubit].readout_frequency = freq_resonator
            except:
                raise_error
                (
                    RuntimeError,
                    "qubit_flux_track: Not enough parameters to estimate the resonator freq for the given bias. Please run resonator spectroscopy flux and update the runcard",
                )

            # modify qubit flux
            qubits[qubit].flux.offset = bias

            # execute pulse sequence sweeping only qubit resonator
            results = platform.sweep(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
                freq_sweeper,
            )

        # retrieve the results for every qubit
        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            data.register_qubit_track(
                qubit,
                signal=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + qd_pulses[qubit].frequency,
                bias=bias + qubits[qubit].sweetspot,
            )

    return data


qubit_flux_tracking = Routine(
    _acquisition,
    qubit_flux_dependence._fit,
    qubit_flux_dependence._plot,
    qubit_flux_dependence._update,
)
"""QubitFluxTrack Routine object."""
