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
from qibocal.fitting.rb_methods import fit_exp1_func, fit_exp2_func, fit_expn_func

from scipy.linalg import expm
from copy import deepcopy


def ham_to_unitary_matr(k=0, dt=0.1, ok=1, jk=1, g1=0, g2=0):
    omega0 = np.pi / dt
    omega1 = omega0 * ok
    J = 1 / (dt * jk)

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

    dissipation = np.array([[0, g2, g1, g1+g2],
                            [g2, 0, g1+g2, g1],
                            [g1, g1+g2, 0, g2],
                            [g1+g2, g1, g2, 0],])
    res_unitary = expm((-1j * hamiltonian + dissipation) * dt)
    return res_unitary


# Define the circuit factory class for this specific module.
class NonMarkovianFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None, dt=0.1, ok=1, jk=1, g1=0, g2=0
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits, density_matrix=True)
        self.dt = dt
        self.ok = ok
        self.jk = jk
        self.g1 = g1
        self.g2 = g2
    
    def __next__(self) -> (Circuit, int):
        if self.n >= self.runs * len(self.depths):
            raise StopIteration
        else:
            circuit, countX = self.build_circuit(self.depths[self.n % len(self.depths)])
            circuit_init_kwargs = circuit.init_kwargs
            del circuit_init_kwargs['nqubits']
            self.n += 1
            # Distribute the circuit onto the given support.
            bigcircuit = Circuit(self.nqubits, **circuit_init_kwargs)
            bigcircuit.add(circuit.on_qubits(*self.qubits))
            return bigcircuit, countX

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = models.Circuit(len(self.qubits), density_matrix=self.density_matrix)
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 2, size=depth)
        # random_ints = [1 for _ in range(depth)]
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = self.nonmark_gates(random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        # Measure only the first qubit
        circuit.add(gates.M(0))
        return circuit, sum(random_ints)
    
    def nonmark_gates(self, xi_inds=[0]):
        omega0 = np.pi / self.dt
        omega1 = omega0 * self.ok
        J = 1 / (self.dt * self.jk)

        gates_list = []

        for k in xi_inds:
            unitary_matrix = ham_to_unitary_matr(k, self.dt, self.ok, self.jk, self.g1, self.g2)
            gates_list.append(gates.Unitary(unitary_matrix, 0, 1))
        
        return gates_list


# Define the circuit factory class for this specific module.
class XFilterFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits, density_matrix=True)

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = models.Circuit(len(self.qubits), density_matrix=self.density_matrix)
        # There are only two gates to choose from.
        a = [gates.X(0), gates.X(0)]
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 2, size=depth)
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = np.take(a, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


def jc_kraus_channel(time=1, omega=2 * np.pi/3):
        # Define Kraus operators
        K0 = np.array([[1, 0], [0, np.cos(omega * time / 2)]])
        K1 = -1j * np.array([[0, np.sin(omega * time / 2)], [0, 0]])

        return gates.KrausChannel([((0,), K0), ((0,), K1)])


class XFilterKrausFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None, kraus_i = None, kraus_x = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits, density_matrix=True)
        self.kraus_i = kraus_i
        self.kraus_x = kraus_x

    def __next__(self) -> (Circuit, int):
        if self.n >= self.runs * len(self.depths):
            raise StopIteration
        else:
            circuit, countX = self.build_circuit(self.depths[self.n % len(self.depths)])
            self.n += 1
            # Distribute the circuit onto the given support.
            bigcircuit = Circuit(self.nqubits, density_matrix=self.density_matrix)
            bigcircuit.add(circuit.queue)
            
            return bigcircuit, countX


    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = models.Circuit(len(self.qubits), density_matrix=self.density_matrix)
        # There are only two gates to choose from.
        a = [self.kraus_i, self.kraus_x]
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 2, size=depth)
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = np.take(a, random_ints)

        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(0))
        return circuit, sum(random_ints)


class XFilterRandMeasFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits, density_matrix=True)

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = models.Circuit(len(self.qubits), density_matrix=self.density_matrix)
        # There are only two gates to choose from.
        a = [gates.I(0), gates.X(0)]
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 2, size=depth)
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = np.take(a, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)

        rand_meas_int = np.random.randint(0, 3)
        if rand_meas_int == 1:
            circuit.add(gates.S(0).dagger())
        if rand_meas_int > 0:
            circuit.add(gates.H(0))
        
        # For Y-basis we need Sdag*H <- before measurement
        # For X-basis we need H before measurement
        # if X: circuit.add(gates.H)
        # if Y: circuit.add(gates.S.dag(), gates.H)
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
    
    def execute(self) -> None:
        """Calls method ``single_task`` while iterating over attribute
        ``circuitfactory```.

        Collects data given the already set data and overwrites
        attribute ``data``.
        """
        if self.circuitfactory is None:
            raise_error(NotImplementedError, "There are no circuits to execute.")
        newdata = []
        for circuit, countX in self.circuitfactory:
            try:
                datarow = next(self.data)
            except TypeError:
                datarow = {}
            datadict = self.single_task(deepcopy(circuit), datarow)
            datadict["countX"] = countX
            newdata.append(datadict)
        self.data = newdata

    def single_task(self, circuit: models.Circuit, datarow: dict) -> dict:
        datadict = super().single_task(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1 if circuit.ngates > 1 else 0
        return datadict

# Define the experiment class for this specific module.
class XFilterExperiment(Experiment):
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
        datadict["countX"] = circuit.gate_types['x']
        return datadict


# Define the result class for this specific module.
class XFilterResult(Result):
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


def filter_sign(experiment: Experiment):
    filtersing_list = []
    state = experiment.init_state if experiment.init_state is not None else np.array([[1, 0], [0, 0]])
    for datarow in experiment.data:
        samples = datarow["samples"]
        countX = datarow["countX"]
        filtersign = 0
        for s in samples:
            # For {X}:
            # k = (countX + s[0]) % 2
            # filtersign += state[k][k]
            # For {I, X}:
            filtersign += (-1) ** (countX % 2 + s[0]) * (2 * state[0][0] - 1) / 2.0
        filtersing_list.append(filtersign / len(samples))
    experiment._append_data("filters", filtersing_list)


def analyze(
    experiment: Experiment, noisemodel: NoiseModel = None, **kwargs
) -> go._figure.Figure:
    experiment.apply_task(filter_sign)
    if kwargs.get("ndecays") == 1:
        fitting_func = fit_exp1_func 
    elif kwargs.get("ndecays") == 2:
        fitting_func = fit_exp2_func 
    else: 
        fitting_func = fit_expn_func
    result = XFilterResult(experiment.dataframe, fitting_func, title=kwargs.get("title"), ndecays=kwargs.get("ndecays"))
    result.single_fig()
    report = result.report()
    return report


# General Markovian perform
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
    is_nonmark = False
):
    # if init_state is None:
    #     init_state = np.array([[1, 0], [0, 0]])
    # Initiate the circuit factory and the faulty Experiment object.
    factory = XFilterFactory(nqubits, depths, runs, qubits=qubits) if not is_nonmark else NonMarkovianFactory(nqubits+1, depths, runs)
    experiment = XFilterExperiment(factory, nshots, noisemodel=noise, init_state=init_state)
    # Execute the experiment.
    experiment.execute()
    report = analyze(experiment, noisemodel=noise, title=title, ndecays=ndecays)
    report.show()

# Non-markovian model 1 
def perform_nonm(
    nqubits: int,
    depths: list,
    runs: int,
    nshots: int,
    qubits: list = None,
    noise: NoiseModel = None,
    ndecays: int = 1,
    init_state = None,
    dt = 0.1,
    ok = 1,
    jk = 1,
    g1 = 0,
    g2 = 0
):
    # if init_state is None:
    #     init_state = np.array([[1, 0], [0, 0]])
    # Initiate the circuit factory and the faulty Experiment object.
    factory = NonMarkovianFactory(nqubits+1, depths, runs, dt=dt, ok=ok, jk=jk, g1=g1, g2=g2)
    experiment = NonMarkovianExperiment(factory, nshots, noisemodel=noise, init_state=init_state)
    # Execute the experiment.
    experiment.execute()
    title = f"$dt={dt}, \Omega_0=π/\Delta t, \Omega_1=\Omega_0\cdot{ok}, J=1/(dt\cdot{jk})$"
    report = analyze(experiment, title=title, ndecays=ndecays)
    report.show()

# Non-Markovian model 2
def perform_kraus(
    nqubits: int,
    depths: list,
    runs: int,
    nshots: int,
    qubits: list = None,
    ndecays: int = 1,
    kraus_i = None,
    kraus_x = None,
    init_state = None,
    noise: NoiseModel = None,
    title:str = None
):
    
    factory = XFilterKrausFactory(nqubits+1, depths, runs, kraus_i=kraus_i, kraus_x=kraus_x)
    experiment = NonMarkovianExperiment(factory, nshots, noisemodel=noise, init_state=init_state)
    # Execute the experiment.
    experiment.execute()
    report = analyze(experiment, title=title, ndecays=ndecays)
    report.show()

