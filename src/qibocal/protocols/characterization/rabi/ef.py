from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Qubits, Results, Routine

from . import amplitude, utils


@dataclass
class RabiAmplitudeEFParameters(amplitude.RabiAmplitudeParameters):
    """RabiAmplitudeEF runcard inputs."""


@dataclass
class RabiAmplitudeEFResults(Results):
    """RabiAmplitudeEF outputs."""

    amplitude: dict[QubitId, float]
    """Drive amplitude for each qubit."""
    length: dict[QubitId, float]
    """Drive pulse duration. Same for all qubits."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitted parameters."""


@dataclass
class RabiAmplitudeEFData(amplitude.RabiAmplitudeData):
    """RabiAmplitude data acquisition."""


def _acquisition(
    params: RabiAmplitudeEFParameters, platform: Platform, qubits: Qubits
) -> RabiAmplitudeEFData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    rx_pulses = {}
    durations = {}
    for qubit in qubits:
        rx_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        qd_pulses[qubit] = platform.create_RX_pulse(
            qubit, start=rx_pulses[qubit].finish
        )
        if params.pulse_length is not None:
            qd_pulses[qubit].duration = params.pulse_length

        durations[qubit] = qd_pulses[qubit].duration
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(rx_pulses[qubit])
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse amplitude
    qd_pulse_amplitude_range = np.arange(
        params.min_amp_factor,
        params.max_amp_factor,
        params.step_amp_factor,
    )
    sweeper = Sweeper(
        Parameter.amplitude,
        qd_pulse_amplitude_range,
        [qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.FACTOR,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = RabiAmplitudeEFData(durations=durations)

    # sweep the parameter
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper,
    )
    for qubit in qubits:
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit,
            amp=qd_pulses[qubit].amplitude * qd_pulse_amplitude_range,
            msr=result.magnitude,
            phase=result.phase,
        )
    return data


def _plot(data: RabiAmplitudeEFData, qubit, fit: RabiAmplitudeEFResults = None):
    """Plotting function for RabiAmplitude."""
    figures, report = utils.plot(data, qubit, fit)
    if report is not None:
        report = report.replace("Pi pulse", "Pi pulse 12")
    return figures, report


def _update(results: RabiAmplitudeEFResults, platform: Platform, qubit: QubitId):
    """Update RX2 amplitude"""
    update.drive_12_amplitude(results.amplitude[qubit], platform, qubit)


rabi_amplitude_ef = Routine(_acquisition, amplitude._fit, _plot, _update)
"""RabiAmplitudeEF Routine object."""
