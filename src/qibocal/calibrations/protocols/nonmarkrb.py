from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo import gates, models
from qibo.noise import NoiseModel, PauliError
from qibolab.platforms.abstract import AbstractPlatform
from qibo.models import Circuit

from qibocal.calibrations.protocols.abstract import Circuitfactory, Experiment, Result
from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.fitting.rb_methods import fit_exp1_func, fit_exp2_func

from scipy.linalg import expm 

# Define the circuit factory class for this specific module.
class NonMarkovianFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits, density_matrix=True)

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = models.Circuit(len(self.qubits), density_matrix=self.density_matrix)
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 2, size=depth)
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = nonmark_gate(random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        # Measure only the first qubit
        circuit.add(gates.M(0))
        return circuit

# Define the circuit factory class for this specific module.
class NonMarkovianFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None, circuit_map: qibo.gates.Gate = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits, density_matrix=True)
        self.circuit_map = circuit_map

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = models.Circuit(len(self.qubits), density_matrix=self.density_matrix)
        # Add gates to circuit.
        if self.circuit_map:
            circuit.add(circuit_map)
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


# Define the experiment class for this specific module.
class NonMarkovianExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
        init_state = None,
    ) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel, init_state)

    def single_task(self, circuit: models.Circuit, datarow: dict) -> dict:
        datadict = super().single_task(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1 if circuit.ngates > 1 else 0
        datadict["countZ"] = circuit.gate_types['z']
        return datadict


# Define the result class for this specific module.
class NonMarkovianResult(Result):
    def __init__(self, dataframe: pd.DataFrame, fitting_func, title="", ndecays=1) -> None:
        super().__init__(dataframe)
        self.fitting_func = fitting_func
        self.title = title
        self.ndecays = ndecays

    def single_fig(self):
        xdata_scatter = self.df["depth"].to_numpy()
        ydata_scatter = self.df["filters"].to_numpy()
        xdata, ydata = self.extract("depth", "filters", "mean")
        self.scatter_fit_fig(xdata_scatter, ydata_scatter, xdata, ydata, ndecays=self.ndecays)


def filter_func(experiment: Experiment):
    filtersing_list = []
    state = experiment.init_state if experiment.init_state is not None else np.array([[1, 0], [0, 0]])
    for datarow in experiment.data:
        samples = datarow["samples"]
        countZ = datarow["countz"]
        filtersign = 0
        for s in samples:
            filtersign += (-1) ** (countZ % 2 + s[0]) * (2 * state[0][0] - 1) / 2.0
        filtersing_list.append(filtersign / len(samples))
    experiment._append_data("filters", filtersing_list)


def analyze(
    experiment: Experiment, noisemodel: NoiseModel = None, **kwargs
) -> go._figure.Figure:
    experiment.apply_task(filter_func)
    fitting_func = fit_exp1_func if kwargs.get("ndecays") == 1 else fit_exp2_func
    result = XFilterResult(experiment.dataframe, fitting_func, title=kwargs.get("title"), ndecays=kwargs.get("ndecays"))
    result.single_fig()
    report = result.report()
    return report


# Make perform take a whole noisemodel already.
def perform(
    nqubits: int,
    depths: list,
    runs: int,
    nshots: int,
    qubits: list = None,
    noise: NoiseModel = None,
    title: str = "X-gate Filtered RB",
    ndecays: int = 1,
    init_state = None,
    is_kraus = False
):
    # if init_state is None:
    #     init_state = np.array([[1, 0], [0, 0]])
    # Initiate the circuit factory and the faulty Experiment object.
    factory = XFilterFactory(nqubits, depths, runs, qubits=qubits) if not is_kraus else XFilterKrausFactory(nqubits, depths, runs, qubits=qubits)
    experiment = XFilterExperiment(factory, nshots, noisemodel=noise, init_state=init_state)
    # Execute the experiment.
    experiment.execute()
    report = analyze(experiment, noisemodel=noise, title=title, ndecays=ndecays)
    report.show()


def nonmark_gate(xi_inds=[0]):
    omega0 = 1
    omega1 = 1
    J = 0.1

    gates_list = []

    for k in xi_inds:
        hamiltonian = np.zeros((4, 4))
        if k:
            hamiltonian += (omega0 / 2) * np.array([[0, 0, 1, 0],
                                                    [0, 0, 0, 1],
                                                    [1, 0, 0, 0],
                                                    [0, 1, 0, 0]])

            hamiltonian += (omega1 / 2) * np.array([[1, 0, 0, 0],
                                                    [0, -1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, -1]])
        hamiltonian += J * np.array([[0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, 0]])
        unitary_matrix = expm(-1j * hamiltonian * np.pi)
        gates_list.append(gates.Unitary(unitary_matrix, 0, 1))
    return gates_list

c = models.Circuit(2, density_matrix=True)
# noisy_x = nonmark_gate(1)
# noisy_i = nonmark_gate(0)
random_ints = [0, 1, 1, 0]
gates_lists = nonmark_gate(random_ints)
c.add(gates_lists)
c.add(gates.M(0))
print(c.draw())
ex = c(nshots=10)
print(ex.samples())
print(ex.probabilities())
