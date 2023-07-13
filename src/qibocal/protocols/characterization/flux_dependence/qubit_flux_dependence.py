from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log

from . import utils
from ..utils import HZ_TO_GHZ


# TODO: implement cross-talk
@dataclass
class QubitFluxParameters(Parameters):
    """QubitFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the qubit frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep (V)."""
    drive_amplitude: float
    """Drive pulse amplitude. Same for all qubits."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    transition: Optional[str] = "0->1"
    """Flux spectroscopy transition type ("0->1" or "0->2")."""


@dataclass
class QubitFluxResults(Results):
    """QubitFlux outputs."""

    sweetspot: dict[QubitId, float] = field(metadata=dict(update="sweetspot"))
    """Sweetspot for each qubit."""
    frequency: dict[QubitId, float] = field(metadata=dict(update="drive_frequency"))
    """Drive frequency for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


QubitFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator flux dependence."""


@dataclass
class QubitFluxData(Data):
    """QubitFlux acquisition outputs."""

    """Resonator type."""
    resonator_type: str

    """ResonatorFlux acquisition outputs."""
    Ec: dict[QubitId, int] = field(default_factory=dict)
    """Qubit Ec provided by the user."""

    Ej: dict[QubitId, int] = field(default_factory=dict)
    """Qubit Ej provided by the user."""

    data: dict[QubitId, npt.NDArray[QubitFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, freq, bias, msr, phase):
        """Store output for single qubit."""
        size = len(freq) * len(bias)
        ar = np.empty(size, dtype=QubitFluxType)
        frequency, biases = np.meshgrid(freq, bias)
        ar["freq"] = frequency.ravel()
        ar["bias"] = biases.ravel()
        ar["msr"] = msr.ravel()
        ar["phase"] = phase.ravel()
        self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: QubitFluxParameters,
    platform: Platform,
    qubits: Qubits,
) -> QubitFluxData:
    """Data acquisition for QubitFlux Experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    Ec = {}
    Ej = {}
    for qubit in qubits:
        Ec[qubit] = qubits[qubit].Ec
        Ej[qubit] = qubits[qubit].Ej

        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=2000
        )

        if params.transition == "0->2":
            qd_pulses[qubit].frequency -= (
                qubits[qubit].anharmonicity / 2
            )  # TODO: add anharmonicity to platform runcard - single qubit gates settings

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
    bias_sweeper = Sweeper(
        Parameter.bias,
        delta_bias_range,
        qubits=list(qubits.values()),
        type=SweeperType.ABSOLUTE,
    )
    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and flux bias
    data = QubitFluxData(resonator_type=platform.resonator_type, Ec=Ec, Ej=Ej)

    # repeat the experiment as many times as defined by software_averages
    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        bias_sweeper,
        freq_sweeper,
    )

    # retrieve the results for every qubit
    for qubit in qubits:
        result = results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit,
            msr=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + qd_pulses[qubit].frequency,
            bias=delta_bias_range,
        )

    return data


def _fit(data: QubitFluxData) -> QubitFluxResults:
    """
    Post-processing for QubitFlux Experiment.
    Fit frequency as a function of current for the flux qubit spectroscopy
    data (QubitFluxData): data object with information on the feature response at each current point.
    """

    qubits = data.qubits
    frequency = {}
    sweetspot = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        Ec = data.Ec[qubit]
        Ej = data.Ej[qubit]

        frequency[qubit] = 0
        sweetspot[qubit] = 0
        fitted_parameters[qubit] = {
            "Xi": 0,
            "d": 0,
            "Ec": 0,
            "Ej": 0,
            "f_q_offset": 0,
            "C_ii": 0,
        }

        biases = qubit_data.bias
        frequencies = qubit_data.freq
        msr = qubit_data.msr

        frequencies, biases = utils.image_to_curve(frequencies, biases, msr)
        scaler = 10**9
        try:
            max_c = biases[np.argmax(frequencies)]
            min_c = biases[np.argmin(frequencies)]
            xi = 1 / (2 * abs(max_c - min_c))  # Convert bias to flux.

            # First order approximation: Ec and Ej NOT provided
            if (Ec and Ej) == 0:
                f_q_0 = np.max(
                    frequencies
                )  # Initial estimation for qubit frequency at sweet spot.
                popt = curve_fit(
                    utils.freq_q_transmon,
                    biases,
                    frequencies / scaler,
                    # p0=[max_c, xi, 0, f_q_0],
                    p0=[max_c, xi, 0, f_q_0 / scaler],
                    bounds=((-np.inf, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf)),
                    maxfev=2000000
                )[0]
                popt[3] *= scaler
                f_qs = popt[3]  # Qubit frequency at sweet spot.
                f_q_offset = utils.freq_q_transmon(
                    0, *popt
                )  # Qubit frequenct at zero current.
                C_ii = (f_qs - f_q_offset) / popt[
                    0
                ]  # Corresponding flux matrix element.

                frequency[qubit] = f_qs * HZ_TO_GHZ
                sweetspot[qubit] = popt[0]
                # fitted_parameters = xi, d, f_q_offset, C_ii
                fitted_parameters[qubit] = {
                    "Xi": popt[1],
                    "d": abs(popt[2]),
                    "f_q_offset": f_q_offset,
                    "C_ii": C_ii,
                }

            # Second order approximation: Ec and Ej provided
            elif (Ec and Ej) != 0:
                freq_q_mathieu1 = partial(utils.freq_q_mathieu, p7=0.4999)
                popt = curve_fit(
                    freq_q_mathieu1,
                    biases,
                    frequencies / scaler,
                    # p0=[max_c, xi, 0, Ec, Ej],
                    p0=[max_c, xi, 0, Ec / scaler, Ej / scaler],
                    bounds=((-np.inf, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf)),
                    maxfev=2000000
                )[0]
                popt[3] *= scaler
                popt[4] *= scaler
                f_qs = utils.freq_q_mathieu(
                    popt[0], *popt
                )  # Qubit frequency at sweet spot.
                f_q_offset = utils.freq_q_mathieu(
                    0, *popt
                )  # Qubit frequenct at zero current.
                C_ii = (f_qs - f_q_offset) / popt[
                    0
                ]  # Corresponding flux matrix element.

                frequency[qubit] = f_qs * HZ_TO_GHZ
                sweetspot[qubit] = popt[0]
                # fitted_parameters = xi, d, Ec, Ej, f_q_offset, C_ii
                fitted_parameters[qubit] = {
                    "Xi": popt[1],
                    "d": abs(popt[2]),
                    "Ec": popt[3],
                    "Ej": popt[4],
                    "f_q_offset": f_q_offset,
                    "C_ii": C_ii,
                }

            else:
                log.warning(
                    "qubit_flux_fit: the fitting was not succesful. Not enought guess parameters provided"
                )

        except:
            log.warning("qubit_flux_fit: the fitting was not succesful")

    return QubitFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: QubitFluxData, fit: QubitFluxResults, qubit):
    """Plotting function for QubitFlux Experiment."""
    return utils.flux_dependence_plot(data, fit, qubit)


qubit_flux = Routine(_acquisition, _fit, _plot)
"""QubitFlux Routine object."""
