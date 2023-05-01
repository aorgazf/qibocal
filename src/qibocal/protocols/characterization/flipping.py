from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color


@dataclass
class FlippingParameters(Parameters):
    nflips_max: int
    nflips_step: int


@dataclass
class FlippingResults(Results):
    amplitudes: Dict[List[Tuple], str] = field(metadata=dict(update="drive_amplitude"))
    amplitude_factors: Dict[List[Tuple], str]
    fitted_parameters: Dict[List[Tuple], List]


class FlippngData(DataUnits):
    def __init__(self, resonator_type):
        super().__init__(
            name="data",
            quantities={"flips": "dimensionless"},
            options=["qubit"],
        )

        self._resonator_type = resonator_type

    @property
    def resonator_type(self):
        return self._resonator_type


def _acquisition(
    params: FlippingParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> FlippngData:
    r"""
    The flipping experiment correct the delta amplitude in the qubit drive pulse. We measure a qubit after applying
    a Rx(pi/2) and N flips (Rx(pi) rotations). After fitting we can obtain the delta amplitude to refine pi pulses.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        nflips_max (int): Maximum number of flips introduced in each sequence
        nflips_step (int): Scan range step for the number of flippings
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **flips[dimensionless]**: Number of flips applied in the current execution
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **amplitude_correction_factor**: Pi pulse correction factor
            - **corrected_amplitude**: Corrected pi pulse amplitude
            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **popt3**: p3
            - **qubit**: The qubit being tested
    """

    # create a DataUnits object to store MSR, phase, i, q and the number of flips
    data = FlippngData(platform.resonator_type)

    # sweep the parameter
    for flips in range(0, params.nflips_max, params.nflips_step):
        # create a sequence of pulses for the experiment
        sequence = PulseSequence()
        ro_pulses = {}
        for qubit in qubits:
            RX90_pulse = platform.create_RX90_pulse(qubit, start=0)
            sequence.add(RX90_pulse)
            # execute sequence RX(pi/2) - [RX(pi) - RX(pi)] from 0...flips times - RO
            start1 = RX90_pulse.duration
            for j in range(flips):
                RX_pulse1 = platform.create_RX_pulse(qubit, start=start1)
                start2 = start1 + RX_pulse1.duration
                RX_pulse2 = platform.create_RX_pulse(qubit, start=start2)
                sequence.add(RX_pulse1)
                sequence.add(RX_pulse2)
                start1 = start2 + RX_pulse2.duration

            # add ro pulse at the end of the sequence
            ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=start1)
            sequence.add(ro_pulses[qubit])
        # execute the pulse sequence
        results = platform.execute_pulse_sequence(sequence)
        for ro_pulse in ro_pulses.values():
            # average msr, phase, i and q over the number of shots defined in the runcard

            r = results[ro_pulse.serial].average.raw
            r.update(
                {
                    "flips[dimensionless]": flips,
                    "qubit": ro_pulse.qubit,
                    "pi_pulse_amplitude": qubits[ro_pulse.qubit].pi_pulse_amplitude,
                }
            )

            data.add(r)

    return data


def flipping_fit(x, p0, p1, p2, p3):
    # A fit to Flipping Qubit oscillation
    # Epsilon?? should be Amplitude : p[0]
    # Offset                        : p[1]
    # Period of oscillation         : p[2]
    # phase for the first point corresponding to pi/2 rotation   : p[3]
    return np.sin(x * 2 * np.pi / p2 + p3) * p0 + p1


# FIXME: not working
def _fit(data: FlippngData) -> FlippingResults:
    r"""
    Fitting routine for T1 experiment. The used model is

    .. math::

        y = p_0 sin\Big(\frac{2 \pi x}{p_2} + p_3\Big).

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the flipping model
        y (str): name of the output values for the flipping model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        niter(int): Number of times of the flipping sequence applied to the qubit
        pi_pulse_amplitudes(list): list of corrected pi pulse amplitude
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **popt3**: p3
            - **labels[0]**: delta amplitude
            - **labels[1]**: corrected amplitude


    """
    qubits = data.df["qubit"].unique()
    corrected_amplitudes = {}
    fitted_parameters = {}
    amplitude_correction_factors = {}
    for qubit in qubits:
        qubit_data_df = data.df[data.df["qubit"] == qubit]
        pi_pulse_amplitude = qubit_data_df["pi_pulse_amplitude"].unique()
        voltages = qubit_data_df["MSR"].pint.to("uV").pint.magnitude
        flips = qubit_data_df["flips"].pint.magnitude

        if data.resonator_type == "3D":
            pguess = [
                pi_pulse_amplitude / 2,
                np.mean(voltages),
                -40,
                0,
            ]  # epsilon guess parameter
        else:
            pguess = [
                pi_pulse_amplitude / 2,
                np.mean(voltages),
                40,
                0,
            ]  # epsilon guess parameter

        try:
            popt, _ = curve_fit(flipping, flips, voltages, p0=pguess, maxfev=2000000)

        except:
            log.warning("flipping_fit: the fitting was not succesful")
            popt = [0, 0, 2, 0]

        eps = -1 / popt[2]
        amplitude_correction_factor = eps / (eps - 1)
        corrected_amplitude = amplitude_correction_factor * pi_pulse_amplitude

        corrected_amplitudes[qubit] = corrected_amplitude
        fitted_parameters[qubit] = popt
        amplitude_correction_factors[qubit] = amplitude_correction_factor

    return FlippingResults(
        corrected_amplitudes, amplitude_correction_factors, fitted_parameters
    )


def _plot(data: FlippngData, fit: FlippingResults, qubit):
    figures = []

    fig = go.Figure()

    fitting_report = ""
    qubit_data = data.df[data.df["qubit"] == qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data["flips"].pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            marker_color=get_color(0),
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        ),
    )

    # add fitting trace
    if len(data) > 0:
        flips_range = np.linspace(
            min(data.df["flips"]),
            max(data.df["flips"]),
            2 * len(data),
        )

        fig.add_trace(
            go.Scatter(
                x=flips_range,
                y=flipping_fit(
                    flips_range,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(1),
            ),
        )
        fitting_report = fitting_report + (
            f"{qubit}| Amplitude Correction Factor: {fit.amplitudes[qubit][0]:.4f}<br>"
            + f"{qubit} | Corrected Amplitude: {fit.amplitude_factors[qubit]:.4f}<br><br>"
        )

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Flips (dimensionless)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


flipping = Routine(_acquisition, _fit, _plot)
