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

from ..utils import GHZ_TO_HZ, HZ_TO_GHZ
from . import utils


@dataclass
class ResonatorFluxParameters(Parameters):
    """ResonatorFlux runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep [Hz]."""
    bias_width: float
    """Width for bias sweep [V]."""
    bias_step: float
    """Bias step for sweep (V)."""
    flux_qubits: Optional[list[QubitId]] = None
    """IDs of the qubits that we will sweep the flux on.
    If ``None`` flux will be swept on all qubits that we are running the routine on in a multiplex fashion.
    If given flux will be swept on the given qubits in a sequential fashion (n qubits will result to n different executions).
    Multiple qubits may be measured in each execution as specified by the ``qubits`` option in the runcard.
    """
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


@dataclass
class ResonatorFluxResults(Results):
    """ResonatoFlux outputs."""

    frequency: dict[QubitId, float] = field(metadata=dict(update="readout_frequency"))
    """Readout frequency for each qubit."""
    sweetspot: dict[QubitId, float] = field(metadata=dict(update="sweetspot"))
    """Sweetspot for each qubit."""
    flux_to_bias: dict[QubitId, float] = field(metadata=dict(update="flux_to_bias"))
    """flux_to_bias for each qubit."""
    asymmetry: dict[QubitId, float] = field(metadata=dict(update="asymmetry"))
    """asymmetry for each qubit."""
    Gs: dict[QubitId, float] = field(metadata=dict(update="readout_coupling"))
    """readout_coupling for each qubit."""
    brf: dict[QubitId, float] = field(metadata=dict(update="brf"))
    """bare_resonator_frequency for each qubit."""
    ssf_brf: dict[QubitId, float] = field(metadata=dict(update="ssf_brf"))
    """sweetspot_qubit_frequency/bare_resonator_frequency for each qubit."""
    ecs: dict[QubitId, float] = field(metadata=dict(update="ec"))
    """ec for each qubit."""
    ejs: dict[QubitId, float] = field(metadata=dict(update="ej"))
    """ej for each qubit."""

    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


@dataclass
class FluxCrosstalkResults(Results):
    """Empty fitting outputs for cross talk because fitting is not implemented in this case."""


ResFluxType = np.dtype(
    [
        ("freq", np.float64),
        ("bias", np.float64),
        ("msr", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for resonator flux dependence."""


@dataclass
class ResonatorFluxData(Data):
    """Resonator type."""

    resonator_type: str

    """ResonatorFlux acquisition outputs."""
    ec: dict[QubitId, float] = field(default_factory=dict)
    """Qubit ec provided by the user."""

    ej: dict[QubitId, float] = field(default_factory=dict)
    """Qubit ej provided by the user."""

    g: dict[QubitId, float] = field(default_factory=dict)
    """Qubit g provided by the user."""

    bare_resonator_frequency: dict[QubitId, int] = field(default_factory=dict)
    """Qubit bare resonator frequency power provided by the user."""

    data: dict[QubitId, npt.NDArray[ResFluxType]] = field(default_factory=dict)
    """Raw data acquired."""

    def register_qubit(self, qubit, flux_qubit, freq, bias, msr, phase):
        """Store output for single qubit."""
        self.data[qubit] = utils.create_data_array(
            freq, bias, msr, phase, dtype=ResFluxType
        )


@dataclass
class FluxCrosstalkData(ResonatorFluxData):
    """QubitFlux acquisition outputs when ``flux_qubits`` are given."""

    data: dict[QubitId, dict[QubitId, npt.NDArray[ResFluxType]]] = field(
        default_factory=dict
    )
    """Raw data acquired for (qubit, qubit_flux) pairs saved in nested dictionaries."""

    def register_qubit(self, qubit, flux_qubit, freq, bias, msr, phase):
        """Store output for single qubit."""
        ar = utils.create_data_array(freq, bias, msr, phase, dtype=ResFluxType)
        if qubit in self.data:
            self.data[qubit][flux_qubit] = ar
        else:
            self.data[qubit] = {flux_qubit: ar}


def _acquisition(
    params: ResonatorFluxParameters, platform: Platform, qubits: Qubits
) -> ResonatorFluxData:
    """Data acquisition for ResonatorFlux experiment."""
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    ec = {}
    ej = {}
    g = {}
    bare_resonator_frequency = {}
    for qubit in qubits:
        ec[qubit] = qubits[qubit].ec
        ej[qubit] = qubits[qubit].ej
        g[qubit] = qubits[qubit].g
        bare_resonator_frequency[qubit] = qubits[qubit].bare_resonator_frequency

        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
        type=SweeperType.OFFSET,
    )

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    if params.flux_qubits is None:
        flux_qubits = [None]
        bias_sweepers = [
            Sweeper(
                Parameter.bias,
                delta_bias_range,
                qubits=list(qubits.values()),
                type=SweeperType.OFFSET,
            )
        ]
        data_cls = ResonatorFluxData

    else:
        flux_qubits = params.flux_qubits
        bias_sweepers = [
            Sweeper(
                Parameter.bias,
                delta_bias_range,
                qubits=[platform.qubits[flux_qubit]],
                type=SweeperType.OFFSET,
            )
            for flux_qubit in flux_qubits
        ]
        data_cls = FluxCrosstalkData

    data = data_cls(
        resonator_type=platform.resonator_type,
        ec=ec,
        ej=ej,
        g=g,
        bare_resonator_frequency=bare_resonator_frequency,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for flux_qubit, bias_sweeper in zip(flux_qubits, bias_sweepers):
        results = platform.sweep(sequence, options, bias_sweeper, freq_sweeper)
        # retrieve the results for every qubit
        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            if flux_qubit is None:
                sweetspot = qubits[qubit].sweetspot
            else:
                sweetspot = platform.qubits[flux_qubit].sweetspot
            data.register_qubit(
                qubit,
                flux_qubit,
                msr=result.magnitude,
                phase=result.phase,
                freq=delta_frequency_range + ro_pulses[qubit].frequency,
                bias=delta_bias_range + sweetspot,
            )

    return data


def _fit(data: ResonatorFluxData) -> ResonatorFluxResults:
    """
    Post-processing for QubitFlux Experiment. See arxiv:0703002
    Fit frequency as a function of current for the flux qubit spectroscopy
    data (QubitFluxData): data object with information on the feature response at each current point.
    """
    if isinstance(data, FluxCrosstalkData):
        return FluxCrosstalkResults()

    qubits = data.qubits
    frequency = {}
    sweetspot = {}
    flux_to_bias = {}
    asymmetry = {}
    Gs = {}
    brf = {}
    ssf_brf = {}
    ecs = {}
    ejs = {}

    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        ec = data.ec[qubit]
        ej = data.ej[qubit]

        frequency[qubit] = 0
        sweetspot[qubit] = 0
        flux_to_bias[qubit] = 0
        asymmetry[qubit] = 0
        Gs[qubit] = 0
        brf[qubit] = 0
        ssf_brf[qubit] = 0
        ecs[qubit] = 0
        ejs[qubit] = 0

        fitted_parameters[qubit] = {
            "Xi": 0,
            "d": 0,
            "g": 0,
            "ec": 0,
            "ej": 0,
            "bare_resonator_frequency": 0,
            "f_qs": 0,
            "f_r_offset": 0,
            "C_ii": 0,
        }

        biases = qubit_data.bias
        frequencies = qubit_data.freq
        msr = qubit_data.msr

        if data.resonator_type == "3D":
            msr = -msr

        frequencies, biases = utils.image_to_curve(
            frequencies, biases, msr, msr_mask=0.5
        )

        # scaler = 10**9
        bare_resonator_frequency = data.bare_resonator_frequency[
            qubit
        ]  # Resonator frequency at high power.
        g = data.g[qubit]  # Readout coupling.
        max_c = biases[np.argmax(frequencies)]
        min_c = biases[np.argmin(frequencies)]
        xi = 1 / (2 * abs(max_c - min_c))  # Convert bias to flux.

        # First order approximation: bare_resonator_frequency, g provided
        if (ec == 0 and ej == 0) and (bare_resonator_frequency != 0 and g != 0):
            try:
                # Initial estimation for resonator frequency at sweet spot.
                f_r_0 = np.max(frequencies)
                # Initial estimation for qubit frequency at sweet spot.
                f_q_0 = bare_resonator_frequency - g**2 / (
                    f_r_0 - bare_resonator_frequency
                )
                popt = curve_fit(
                    utils.freq_r_transmon,
                    biases,
                    frequencies / GHZ_TO_HZ,
                    p0=[
                        max_c,
                        xi,
                        0,
                        f_q_0 / bare_resonator_frequency,
                        g / GHZ_TO_HZ,
                        bare_resonator_frequency / GHZ_TO_HZ,
                    ],
                    bounds=(
                        (-np.inf, 0, 0, 0, 0, 0),
                        (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
                    ),
                    maxfev=2000000,
                )[0]

                sweetspot[qubit] = popt[0]
                flux_to_bias[qubit] = popt[1]
                asymmetry[qubit] = popt[2]
                Gs[qubit] = popt[4]
                brf[qubit] = popt[5]
                ssf_brf[qubit] = popt[3]

                popt[4] *= GHZ_TO_HZ
                popt[5] *= GHZ_TO_HZ
                f_qs = popt[3] * popt[5]  # Qubit frequency at sweet spot.
                f_rs = utils.freq_r_transmon(
                    popt[0], *popt
                )  # Resonator frequency at sweet spot.
                f_r_offset = utils.freq_r_transmon(
                    0, *popt
                )  # Resonator frequency at zero current.
                C_ii = (f_rs - f_r_offset) / popt[
                    0
                ]  # Corresponding flux matrix element.

                frequency[qubit] = f_rs * HZ_TO_GHZ
                fitted_parameters[qubit] = {
                    "Xi": popt[1],
                    "d": abs(popt[2]),
                    "g": popt[4],
                    "bare_resonator_frequency": popt[5],
                    "f_qs": f_qs,
                    "f_r_offset": f_r_offset,
                    "C_ii": C_ii,
                }

            except:
                log.warning(
                    "resonator_flux_fit: First order approximation fitting was not succesful"
                )

        # Second order approximation: bare_resonator_frequency, g, ec, ej provided
        elif ec != 0 and ej != 0 and bare_resonator_frequency != 0 and g != 0:
            try:
                freq_r_mathieu1 = partial(utils.freq_r_mathieu, p7=0.4999)
                popt = curve_fit(
                    freq_r_mathieu1,
                    biases,
                    frequencies / GHZ_TO_HZ,
                    p0=[
                        bare_resonator_frequency / GHZ_TO_HZ,
                        g / GHZ_TO_HZ,
                        max_c,
                        xi,
                        0,
                        ec / GHZ_TO_HZ,
                        ej / GHZ_TO_HZ,
                    ],
                    bounds=(
                        (0, 0, -np.inf, 0, 0, 0, 0),
                        (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
                    ),
                    maxfev=2000000,
                )[0]

                sweetspot[qubit] = popt[2]
                flux_to_bias[qubit] = popt[3]
                asymmetry[qubit] = popt[4]
                Gs[qubit] = popt[1]
                brf[qubit] = popt[0]
                ECs[qubit] = popt[5]
                EJs[qubit] = popt[6]

                popt[0] *= GHZ_TO_HZ
                popt[1] *= GHZ_TO_HZ
                popt[5] *= GHZ_TO_HZ
                popt[6] *= GHZ_TO_HZ
                f_qs = utils.freq_q_mathieu(
                    popt[2], *popt[2::]
                )  # Qubit frequency at sweet spot.
                f_rs = utils.freq_r_mathieu(
                    popt[2], *popt
                )  # Resonator frequency at sweet spot.
                f_r_offset = utils.freq_r_mathieu(
                    0, *popt
                )  # Resonator frequenct at zero current.
                C_ii = (f_rs - f_r_offset) / popt[
                    2
                ]  # Corresponding flux matrix element.

                frequency[qubit] = f_rs * HZ_TO_GHZ
                sweetspot[qubit] = popt[2]
                fitted_parameters[qubit] = {
                    "Xi": popt[3],
                    "d": abs(popt[4]),
                    "g": popt[1],
                    "ec": popt[5],
                    "ej": popt[6],
                    "bare_resonator_frequency": popt[0],
                    "f_qs": f_qs,
                    "f_r_offset": f_r_offset,
                    "C_ii": C_ii,
                }
            except:
                log.warning(
                    "resonator_flux_fit: Second order approximation fitting was not succesful"
                )

        else:
            log.warning("resonator_flux_fit: Not enought guess parameters provided")

    return ResonatorFluxResults(
        frequency=frequency,
        sweetspot=sweetspot,
        flux_to_bias=flux_to_bias,
        asymmetry=asymmetry,
        Gs=Gs,
        brf=brf,
        ssf_brf=ssf_brf,
        ECs=ECs,
        EJs=EJs,
        fitted_parameters=fitted_parameters,
    )


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, qubit):
    """Plotting function for ResonatorFlux Experiment."""
    if isinstance(data[qubit], dict):
        return utils.flux_crosstalk_plot(data, fit, qubit)
    return utils.flux_dependence_plot(data, fit, qubit)


resonator_flux = Routine(_acquisition, _fit, _plot)
"""ResonatorFlux Routine object."""
