from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.protocols.characterization.qubit_spectroscopy_ef import (
    DEFAULT_ANHARMONICITY,
)

from ..utils import GHZ_TO_HZ, HZ_TO_GHZ, table_dict, table_html
from . import utils


@dataclass
class QubitFluxParameters(Parameters):
    """QubitFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the qubit frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep [a.u.]."""
    drive_amplitude: Optional[float] = None
    """Drive amplitude (optional). If defined, same amplitude will be used in all qubits.
    Otherwise the default amplitude defined on the platform runcard will be used"""
    transition: Optional[str] = "01"
    """Flux spectroscopy transition type ("01" or "02"). Default value is 01"""
    drive_duration: int = 2000
    """Duration of the drive pulse."""


@dataclass
class QubitFluxResults(Results):
    """QubitFlux outputs."""

    sweetspot: dict[QubitId, float]
    """Sweetspot for each qubit."""
    frequency: dict[QubitId, float]
    """Drive frequency for each qubit."""
    asymmetry: dict[QubitId, float]
    """Asymmetry between junctions."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""
    matrix_element: dict[QubitId, float]
    """V_ii coefficient."""


QubitFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator flux dependence."""


@dataclass
class QubitFluxData(Data):
    """QubitFlux acquisition outputs."""

    resonator_type: str
    """Resonator type."""

    qubit_frequency: dict[QubitId, float] = field(default_factory=dict)
    """Qubit frequencies."""

    data: dict[QubitId, npt.NDArray[QubitFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, signal, phase):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, signal, phase, dtype=QubitFluxType
        )


def _acquisition(
    params: QubitFluxParameters,
    platform: Platform,
    qubits: Qubits,
) -> QubitFluxData:
    """Data acquisition for QubitFlux Experiment."""

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    qubit_frequency = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )
        qubit_frequency[qubit] = platform.qubits[qubit].drive_frequency

        if params.transition == "02":
            if qubits[qubit].anharmonicity:
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
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    bias_sweepers = [
        Sweeper(
            Parameter.bias,
            delta_bias_range,
            qubits=list(qubits.values()),
            type=SweeperType.OFFSET,
        )
    ]
    data = QubitFluxData(
        resonator_type=platform.resonator_type, qubit_frequency=qubit_frequency
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for bias_sweeper in bias_sweepers:
        results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)
        # retrieve the results for every qubit
        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            sweetspot = qubits[qubit].sweetspot
            data.register_qubit(
                qubit,
                signal=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + qd_pulses[qubit].frequency,
                bias=delta_bias_range + sweetspot,
            )
    return data


def _fit(data: QubitFluxData) -> Optional[QubitFluxResults]:
    """
    Post-processing for QubitFlux Experiment. See arxiv:0703002
    Fit frequency as a function of current for the flux qubit spectroscopy
    data (QubitFluxData): data object with information on the feature response at each current point.
    """

    qubits = data.qubits
    frequency = {}
    sweetspot = {}
    asymmetry = {}
    matrix_element = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]

        biases = qubit_data.bias
        frequencies = qubit_data.freq
        signal = qubit_data.signal

        if data.resonator_type == "3D":
            frequencies, biases = utils.extract_min_feature(
                frequencies,
                biases,
                signal,
            )
        else:
            frequencies, biases = utils.extract_max_feature(
                frequencies,
                biases,
                signal,
            )

        try:
            popt = curve_fit(
                utils.transmon_frequency,
                biases,
                frequencies * HZ_TO_GHZ,
                bounds=utils.qubit_flux_dependence_fit_bounds(
                    data.qubit_frequency[qubit], qubit_data.bias
                ),
                maxfev=100000,
            )[0]
            fitted_parameters[qubit] = popt.tolist()
            frequency[qubit] = popt[0] * GHZ_TO_HZ
            asymmetry[qubit] = popt[1]
            sweetspot[qubit] = popt[3]
            matrix_element[qubit] = popt[2]
        except ValueError as e:
            log.error(
                f"Error in qubit_flux protocol fit: {e} "
                "The threshold for the SNR mask is probably too high. "
                "Lowering the value of `threshold` in `extract_*_feature`"
                "should fix the problem."
            )
            return None

    return QubitFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        asymmetry=asymmetry,
        matrix_element=matrix_element,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: QubitFluxData, fit: QubitFluxResults, qubit):
    """Plotting function for QubitFlux Experiment."""
    figures = utils.flux_dependence_plot(
        data, fit, qubit, fit_function=utils.transmon_frequency
    )
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    "Sweetspot [V]",
                    "Qubit Frequency at Sweetspot [Hz]",
                    "Asymmetry d",
                    "V_ii [V]",
                ],
                [
                    np.round(fit.sweetspot[qubit], 4),
                    np.round(fit.frequency[qubit], 4),
                    np.round(fit.asymmetry[qubit], 4),
                    np.round(fit.matrix_element[qubit], 4),
                ],
            )
        )
        return figures, fitting_report
    return figures, ""


def _update(results: QubitFluxResults, platform: Platform, qubit: QubitId):
    update.drive_frequency(results.frequency[qubit], platform, qubit)
    update.sweetspot(results.sweetspot[qubit], platform, qubit)
    update.asymmetry(results.asymmetry[qubit], platform, qubit)


qubit_flux = Routine(_acquisition, _fit, _plot, _update)
"""QubitFlux Routine object."""
