from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from ..utils import GHZ_TO_HZ, chi2_reduced, table_dict, table_html
from .utils import PERR_EXCEPTION, POPT_EXCEPTION, fitting, ramsey_fit, ramsey_sequence

COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"


@dataclass
class RamseyParameters(Parameters):
    """Ramsey runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    detuning: Optional[int] = 0
    """Frequency detuning [Hz] (optional).
        If 0 standard Ramsey experiment is performed."""
    unrolling: bool = False
    """If ``True`` it uses sequence unrolling to deploy multiple sequences in a single instrument call.
    Defaults to ``False``."""


@dataclass
class RamseyResults(Results):
    """Ramsey outputs."""

    frequency: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [Hz] for each qubit."""
    t2: dict[QubitId, tuple[float, Optional[float]]]
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [Hz] correction for each qubit."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""
    chi2: dict[QubitId, tuple[float, Optional[float]]]
    """Chi squared estimate mean value and error. """


RamseyType = np.dtype(
    [("wait", np.float64), ("prob", np.float64), ("errors", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class RamseyData(Data):
    """Ramsey acquisition outputs."""

    detuning: int
    """Frequency detuning [Hz]."""
    qubit_freqs: dict[QubitId, float] = field(default_factory=dict)
    """Qubit freqs for each qubit."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def waits(self):
        """
        Return a list with the waiting times without repetitions.
        """
        qubit = next(iter(self.data))
        return np.unique(self.data[qubit].wait)


def _acquisition(
    params: RamseyParameters,
    platform: Platform,
    qubits: Qubits,
) -> RamseyData:
    """Data acquisition for Ramsey Experiment (detuned).

    The protocol consists in applying the following pulse sequence
    RX90 - wait - RX90 - MZ
    for different waiting times `wait`.
    The range of waiting times is defined through the attributes
    `delay_between_pulses_*` available in `RamseyParameters`. The final range
    will be constructed using `np.arange`.
    It is possible to detune the drive frequency using the parameter `detuning` in
    RamseyParameters which will increment the drive frequency accordingly.
    Currently when `detuning==0` it will be performed a sweep over the waiting values
    if `detuning` is not zero, all sequences with different waiting value will be
    executed sequentially. By providing the option `unrolling=True` in RamseyParameters
    the sequences will be unrolled when the frequency is detuned.
    The following protocol will display on the y-axis the probability of finding the ground
    state, therefore it is advise to execute it only after having performed the single
    shot classification. Error bars are provided as binomial distribution error.
    """

    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    options = ExecutionParameters(
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    sequence = PulseSequence()

    data = RamseyData(
        detuning=params.detuning,
        qubit_freqs={
            qubit: platform.qubits[qubit].native_gates.RX.frequency for qubit in qubits
        },
    )

    if params.detuning == 0:
        sequence = PulseSequence()
        for qubit in qubits:
            sequence += ramsey_sequence(platform=platform, qubit=qubit)

        sweeper = Sweeper(
            Parameter.start,
            waits,
            [
                sequence.get_qubit_pulses(qubit).qd_pulses[-1] for qubit in qubits
            ],  # TODO: check if it is correct
            type=SweeperType.ABSOLUTE,
        )

        # execute the sweep
        results = platform.sweep(
            sequence,
            options,
            sweeper,
        )
        for qubit in qubits:
            probs = results[qubit].probability()
            # The probability errors are the standard errors of the binomial distribution
            errors = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs]
            data.register_qubit(
                RamseyType,
                (qubit),
                dict(
                    wait=waits,
                    prob=probs,
                    errors=errors,
                ),
            )

    if params.detuning != 0:
        sequences, all_ro_pulses = [], []
        for wait in waits:
            sequence = PulseSequence()
            for qubit in qubits:
                sequence += ramsey_sequence(
                    platform=platform, qubit=qubit, wait=wait, detuning=params.detuning
                )

            sequences.append(sequence)
            all_ro_pulses.append(sequence.ro_pulses)

        if params.unrolling:
            results = platform.execute_pulse_sequences(sequences, options)

        elif not params.unrolling:
            results = [
                platform.execute_pulse_sequence(sequence, options)
                for sequence in sequences
            ]

        # We dont need ig as every serial is different
        for ig, (wait, ro_pulses) in enumerate(zip(waits, all_ro_pulses)):
            for qubit in qubits:
                serial = ro_pulses[qubit].serial
                if params.unrolling:
                    result = results[serial][0]
                else:
                    result = results[ig][serial]
                prob = result.probability()
                error = np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    RamseyType,
                    (qubit),
                    dict(
                        wait=np.array([wait]),
                        prob=np.array([prob]),
                        errors=np.array([error]),
                    ),
                )

    return data


def _fit(data: RamseyData) -> RamseyResults:
    r"""
    Fitting routine for Ramsey experiment. The used model is
    .. math::
        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x p_4}.
    """
    qubits = data.qubits
    waits = data.waits
    popts = {}
    freq_measure = {}
    t2_measure = {}
    delta_phys_measure = {}
    chi2 = {}
    for qubit in qubits:
        qubit_data = data[qubit]
        qubit_freq = data.qubit_freqs[qubit]
        probs = qubit_data["prob"]
        try:
            popt, perr = fitting(waits, probs, qubit_data.errors)
        except:
            popt = POPT_EXCEPTION
            perr = PERR_EXCEPTION

        delta_fitting = popt[2] / (2 * np.pi)
        # TODO: check sign
        delta_phys = int(delta_fitting * GHZ_TO_HZ - data.detuning)
        corrected_qubit_frequency = int(qubit_freq - delta_phys)
        t2 = 1 / popt[4]
        # TODO: check error formula
        freq_measure[qubit] = (
            corrected_qubit_frequency,
            perr[2] * GHZ_TO_HZ / (2 * np.pi),
        )
        t2_measure[qubit] = (t2, perr[4] * (t2**2))
        popts[qubit] = popt
        # TODO: check error formula
        delta_phys_measure[qubit] = (
            delta_phys,
            popt[2] * GHZ_TO_HZ / (2 * np.pi),
        )
        chi2[qubit] = (
            chi2_reduced(
                probs,
                ramsey_fit(waits, *popts[qubit]),
                qubit_data.errors,
            ),
            np.sqrt(2 / len(probs)),
        )
    return RamseyResults(freq_measure, t2_measure, delta_phys_measure, popts, chi2)


def _plot(data: RamseyData, qubit, fit: RamseyResults = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.data[qubit]
    waits = data.waits
    probs = qubit_data["prob"]
    error_bars = qubit_data["errors"]
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs,
                opacity=1,
                name="Probability of State 0",
                showlegend=True,
                legendgroup="Probability of State 0",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((waits, waits[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=waits,
                y=ramsey_fit(
                    waits,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                    float(fit.fitted_parameters[qubit][4]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            )
        )
        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    "Delta Frequency [Hz]",
                    "Drive Frequency [Hz]",
                    "T2* [ns]",
                    "chi2 reduced",
                ],
                [
                    fit.delta_phys[qubit],
                    fit.frequency[qubit],
                    fit.t2[qubit],
                    fit.chi2[qubit],
                ],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Ground state probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: RamseyResults, platform: Platform, qubit: QubitId):
    update.drive_frequency(results.frequency[qubit][0], platform, qubit)


ramsey = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""
