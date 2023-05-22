from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.platforms.platform import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.calibrations.niGSC.basics.fitting import fit_exp2_func
from qibocal.calibrations.niGSC.XIdrb import ModuleReport
from qibocal.config import log
from qibocal.data import DataUnits

from . import utils

"""
For info check
https://forest-benchmarking.readthedocs.io/en/latest/examples/randomized_benchmarking.html
"""


@dataclass
class XRBParameters(Parameters):
    """Standard RB runcard inputs."""

    min_depth: int
    """Minimum depth."""
    max_depth: int
    """Minimum amplitude multiplicative factor."""
    step_depth: int
    """Minimum amplitude multiplicative factor."""
    runs: int
    """Number of random sequences per depth"""
    nshots: int
    """Number of shots."""
    relaxation_time: float
    """Relaxation time (ns)."""


@dataclass
class XRBResults(Results):
    """Standard RB outputs."""

    fidelities: Dict[List[float], List]
    """Probabilities obtained for each sequence"""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitted parameters."""


class XRBData(DataUnits):
    """RabiAmplitude data acquisition."""

    def __init__(self):
        super().__init__(
            "data",
            {
                "sequence": "dimensionless",
                "length": "dimensionless",
                "probabilities": "dimensionless",
            },
            options=["qubit"],
        )


def _acquisition(
    params: XRBParameters, platform: AbstractPlatform, qubits: Qubits
) -> XRBData:
    r"""
    Data acquisition for Rabi experiment sweeping amplitude.
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.
    """

    # create sequences of pulses for the experiment
    depths = list(range(params.min_depth, params.max_depth, params.step_depth))

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = XRBData()

    # sweep the parameter
    for qubit in qubits.values():
        # Generate random I-X circuits and corresponding pulse sequences
        sequences = defaultdict(list)
        circuits = defaultdict(list)
        for depth in depths:
            for run in range(params.runs):
                circuit = list(np.random.randint(0, 2, depth))
                sequence = PulseSequence()
                for _ in range(sum(circuit)):
                    sequence.add(
                        platform.create_RX_pulse(
                            qubit,
                            start=sequence.finish,
                        )
                    )
                sequences[f"{depth}_{run}"].append(sequence)
                circuits[f"{depth}_{run}"].append(circuit)

        # Execute pulse sequences
        for sequence, circuit in zip(sequences.values(), circuits.values()):
            results = platform.execute_pulse_sequence(
                sequence[0],
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                ),
            )

            ro_pulses = sequence[0].ro_pulses
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[0].serial]
            r = result.serialize

            print(circuit[0])

            r.update(
                {
                    # "sequence[dimensionless]": [int(x) for x in circuit[0]],
                    "sequence[dimensionless]": 0,  # TODO: Store sequences
                    "length[dimensionless]": len(circuit[0]),
                    "probabilities[dimensionless]": r["state_0"][0],
                    "qubit": qubit.name,
                }
            )
            data.add_data_from_dict(r)

    return data


def _fit(data: XRBData) -> XRBResults:
    """Post-processing for Standard RB."""

    qubits = data.df["qubit"].unique()

    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit]

        sequence_length = qubit_data["length"].pint.to("dimensionless").pint.magnitude
        probabilities = (
            qubit_data["probabilities"].pint.to("dimensionless").pint.magnitude
        )

        # TODO: Translate no
        x = sequence_length.values
        y = probabilities.values

        popt, pcov = fit_exp2_func(x, y)

        fitted_parameters[qubit] = popt

    return XRBResults(fitted_parameters)


def _plot(data: XRBData, fit: XRBResults, qubit):
    """Plotting function for Standard RB."""
    # Initiate a report object.
    report = ModuleReport()
    # Add general information to the table.
    report.info_dict["Number of qubits"] = 1
    report.info_dict["Number of shots"] = len(experiment.data[0]["samples"])
    report.info_dict["runs"] = experiment.extract("samples", "depth", "count")[1][0]
    dfrow = df_aggr.loc["filter"]
    popt_pairs = list(dfrow["popt"].items())[::2] + list(dfrow["popt"].items())[1::2]
    report.info_dict["Fit"] = "".join(
        [f"{key}={number_to_str(value)} " for key, value in popt_pairs]
    )
    perr_pairs = list(dfrow["perr"].items())[::2] + list(dfrow["perr"].items())[1::2]
    report.info_dict["Fitting deviations"] = "".join(
        [f"{key}={number_to_str(value)} " for key, value in perr_pairs]
    )
    # Use the predefined ``scatter_fit_fig`` function from ``basics.utils`` to build the wanted
    # plotly figure with the scattered filtered data along with the mean for
    # each depth and the exponential fit for the means.
    report.all_figures.append(scatter_fit_fig(experiment, df_aggr, "depth", "filter"))

    # Return the figure of the report object and the corresponding table.
    return report.build()


XRB = Routine(_acquisition, _fit, _plot)
"""Standard RB Routine object."""
