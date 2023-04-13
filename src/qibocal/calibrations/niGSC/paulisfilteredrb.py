from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from qibo import gates, matrices
from qibo.models import Circuit
from qibo.noise import NoiseModel
from qibo.quantum_info import comp_basis_to_pauli

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.circuitfactory import CircuitFactory
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.plot import Report, scatter_fit_fig
from qibocal.config import raise_error

# matrices = QiboMatrices()
pauli_list = []
for p in [matrices.I, matrices.X, matrices.Y, matrices.Z]:
    pauli_list.append(gates.Unitary(p, 0))
    pauli_list.append(gates.Unitary(-p, 0))
    pauli_list.append(gates.Unitary(1j * p, 0))
    pauli_list.append(gates.Unitary(-1j * p, 0))


def is_xy(g):
    """Check if a Unitary gate is not diagonal"""
    return not np.allclose(np.abs(g.parameters), np.eye(2))


# Define the circuit factory class for this specific module.
class ModuleFactory(CircuitFactory):
    def __init__(self, nqubits: int, depths: list, qubits: list = []) -> None:
        super().__init__(nqubits, depths, qubits)
        if len(self.qubits) != 1:
            raise_error(
                ValueError,
                f"This class is written for gates acting on only one qubit, not {len(self.qubits)} qubits.",
            )
        self.name = "PauliGroup"

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(1, density_matrix=True)
        # Draw sequence of random Pauli gates.
        gate_lists = np.random.choice(pauli_list, size=depth)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(0))
        return circuit


# Define the experiment class for this specific module.
class ModuleExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        data: Iterable | None = None,
        nshots: int | None = None,
        noise_model: NoiseModel = None,
    ) -> None:
        super().__init__(circuitfactory, data, nshots, noise_model)
        self.prebuild()
        self.name = "paulisfilteredrb"

    def execute(self, circuit: Circuit, datarow: dict) -> dict:
        datadict = super().execute(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1
        # Find the number of X and Y gates in the circuit
        countXY = circuit.gate_types["x"] + circuit.gate_types["y"]
        for gate in circuit.gates_of_type("Unitary"):
            countXY += int(is_xy(gate[-1]))
        datadict["countXY"] = countXY
        return datadict


# Define the result class for this specific module.
class ModuleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Pauli Filtered Randomized Benchmarking"


# The filter functions/post processing functions always dependent on circuit and data row!
# It is executed row by row when used on an experiment object.
def filter_irrep(circuit: Circuit, datarow: dict) -> dict:
    """Calculates the filtered signal for the gate group :math:`\\{i^k\\sigma_j\\}_\\{j, k=0\\}^\\{3\\}`.

    :math:`n_X`, :math:`n_Y` denote the amount of :math:`X` and :math:`Y` gates in the circuit with gates
    :math:`g` and :math`i` the outcome which is either ground state :math:`0`
    or exited state :math:`1`.

    .. math::
        f_{\\lambda}(i,g)
        = (-1)^{m_x+m_y + i}/2


    Args:
        circuit (Circuit): Not needed here.
        datarow (dict): The dictionary with the samples from executed circuits and :math:`\\sum k_j`
                        of all the gates :math:`R_x(k_j\\pi/2)` in the executed circuit.

    Returns:
        dict: _description_
    """
    samples = datarow["samples"]
    countXY = datarow["countXY"]
    filtervalue = 0
    for s in samples:
        filtervalue += np.conj(((-1) ** (countXY % 2 + s[0])) / 2.0)

    datarow["filter"] = np.real(filtervalue / len(samples))
    return datarow


# All the predefined sequential postprocessing / filter functions are bundled together here.
def post_processing_sequential(experiment: Experiment):
    """Perform sequential tasks needed to analyze the experiment results.

    The data is added/changed in the experiment, nothing has to be returned.

    Args:
        experiment (Experiment): Experiment object after execution of the experiment itself.
    """

    # Compute and add the ground state probabilities row by row.
    experiment.perform(filter_irrep)


# After the row by row execution of tasks comes the aggregational task. Something like calculation
# of means, deviations, fitting data, tasks where the whole data as to be looked at, and not just
# one instance of circuit + other information.
def get_aggregational_data(experiment: Experiment, ndecays: int = None) -> pd.DataFrame:
    """Computes aggregational tasks, fits data and stores the results in a data frame.

    No data is manipulated in the ``experiment`` object.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.
        ndecays (int): Number of decay parameters to fit. Default is 1.

    Returns:
        pd.DataFrame: The summarized data.
    """
    ndecays = ndecays if ndecays is not None else 1
    # Has to fit the column describtion from ``filter_irrep``.
    depths, ydata = experiment.extract("filter", "depth", "mean")
    _, ydata_std = experiment.extract("filter", "depth", "std")
    # Fit the filtered signal for each depth, there could be two overlaying exponential functions.
    if ndecays == 1:
        popt, perr = fitting_methods.fit_exp1_func(depths, ydata)
        popt_dict = {"A": popt[0], "p": popt[1]}
        perr_dict = {"A_err": perr[0], "p_err": perr[1]}
        fit_func_label = "exp1_func"
    else:
        fitting_methods.fit_expn_func(depths, ydata, ndecays)
        popt_keys = [f"A{k+1}" for k in range(ndecays)]
        popt_keys += [f"p{k+1}" for k in range(ndecays)]
        popt_dict = dict(zip(popt_keys, popt))
        perr_keys = [f"A{k+1}_err" for k in range(ndecays)]
        perr_keys += [f"p{k+1}_err" for k in range(ndecays)]
        perr_dict = dict(zip(perr_keys, perr))
        fit_func_label = "expn_func"

    # Build a list of dictionaries with the aggregational information.
    data = [
        {
            "depth": depths,  # The x-axis.
            "data": ydata,  # The filtred signal.
            "2sigma": 2 * ydata_std,  # The 2 * standard deviation error for each depth.
            "fit_func": fit_func_label,  # Which function was used to fit.
            "popt": popt_dict,  # The fitting parameters.
            "perr": perr_dict,  # The estimated errors.
        }
    ]

    df = pd.DataFrame(data, index=["filter"])
    return df


def gate_group(nqubits=1):
    """
    Pauli gate group
    """
    return pauli_list


def irrep_info(nqubits=1):
    """
    Infromation corresponding to the irreducible representation of the Pauli group.
    Returns:
        tuple: (basis, index, size, multiplicity) of the z irrep
    """
    basis_c2p_1q = comp_basis_to_pauli(1, normalize=True)
    return (basis_c2p_1q, 3, 1, 1)


# This is highly individual. The only important thing for the qq module is that a plotly figure is
# returned, if qq is not used any type of figure can be build.
def build_report(experiment: Experiment, df_aggr: pd.DataFrame) -> Figure:
    """Use data and information from ``experiment`` and the aggregated data data frame to
    build a report as plotly figure.

    Args:
        experiment (Experiment): After sequential postprocessing of the experiment data.
        df_aggr (pd.DataFrame): Normally build with ``get_aggregational_data`` function.

    Returns:
        (Figure): A plotly.graphical_object.Figure object.
    """

    # Initiate a report object.
    report = ModuleReport()
    # Add general information to the object.
    report.info_dict["Number of qubits"] = len(experiment.data[0]["samples"][0])
    report.info_dict["Number of shots"] = len(experiment.data[0]["samples"])
    report.info_dict["runs"] = experiment.extract("samples", "depth", "count")[1][0]
    report.info_dict["Fitting daviations"] = "".join(
        [
            "{}:{:.3f} ".format(key, df_aggr.loc["filter"]["perr"][key])
            for key in df_aggr.loc["filter"]["perr"]
        ]
    )
    # Use the predefined ``scatter_fit_fig`` function from ``basics.utils`` to build the wanted
    # plotly figure with the scattered filtered data along with the mean for
    # each depth and the exponential fit for the means.
    report.all_figures.append(scatter_fit_fig(experiment, df_aggr, "depth", "filter"))
    # Return the figure the report object builds out of all figures added to the report.
    return report.build()
