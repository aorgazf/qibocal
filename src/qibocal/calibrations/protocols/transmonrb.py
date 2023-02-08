from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from qibo import gates, matrices, models
from qibo.models import Circuit
from qibo.noise import NoiseModel, PauliError
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.calibrations.protocols.abstract import (
    Experiment,
    Result,
    SingleCliffordsFactory,
    Circuitfactory
)
# Temporary (2):
from copy import deepcopy
from qibocal.calibrations.protocols.utils import ONEQUBIT_CLIFFORD_PARAMS

from qibocal.calibrations.protocols.utils import effective_depol
from qibocal.config import raise_error
from qibocal.data import Data
from qibocal.decorators import plot
from qibocal.fitting.rb_methods import fit_exp1_func, fit_exp1B_func, fit_exp2_func, fit_expn_func
from qibocal.plots.rb import standardrb_plot

import qibo
qibo.set_backend("numpy") # for KrausChannel to work (when it's not invertible) 

from qibo.quantum_info.basis import vectorization, unvectorization

from odeintw import odeintw

# Constants
FREQ0 = 4 * 1e9 # Rotation frequency of the 1st qubit
GATE_TIME = 40 * 1e-9 # Gate time

noisy_x_dict = {}

def s(t):
    '''
    s(t)=π/t constant (for now)
    '''
    return np.pi / (GATE_TIME) # * 1e-9)

def standard_hamiltonian(t, omega_d, phi):
    '''
    H = -omega0/2 sigma_z + π/t sin(omega_d t + phi) sigma_y
    '''
    # Constants
    big_omega = 1
    v_0 = 1
    # Hamiltonian
    hamiltonian = np.zeros((2, 2), dtype=complex)
    hamiltonian += (-FREQ0 / 2) * matrices.Z
    hamiltonian += big_omega * v_0 * s(t) * np.sin((omega_d * t) + phi) * matrices.Y
    return hamiltonian

def get_gate(omega, phi, k=0):
    global noisy_x_dict
    if k in noisy_x_dict.keys():
        return noisy_x_dict[k]

    def tdse(a, t, omega_d, phi):
        h = standard_hamiltonian(t, omega_d, phi + k * FREQ0 * GATE_TIME)
        return -1j * (h @ a)

    u0 = np.eye(2, dtype=complex)
    t = np.linspace(0, GATE_TIME, 1000)
    sol = odeintw(tdse, u0, t, args=(omega, phi))
    noisy_x_dict[k] = gates.Unitary(sol[-1], 0)
    # print(np.linalg.norm(sol[-1] @ np.transpose(np.conj(sol[-1])) - np.eye(2)))
    # print(sol[-1].round(3))
    return gates.Unitary(sol[-1], 0)
    

class TransmonXIdFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None, factory_id: str = None, gate_set: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits)
        self.gate_set = gate_set
        self.factory_id = factory_id


    def build_ideal_circuit(self, random_ints: list = []):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(1, density_matrix=True)
        # There are only two gates to choose from.
        a = [gates.I(0), gates.X(0)]
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = np.take(a, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(0))
        return circuit
    

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(len(self.qubits), density_matrix=True)
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, 2, size=depth)
        # Create a gate sequence that accounts for the z-rotation
        for k in range(depth):
            if random_ints[k]:
                circuit.add(get_gate(FREQ0, k * FREQ0 * GATE_TIME))
            else:
                circuit.add(gates.I(0))

        circuit.add(gates.M(0))
        # Create the same circuit but without noise
        ideal_circuit = self.build_ideal_circuit(random_ints)
        # Return noisy and ideal circuits
        return circuit, ideal_circuit
        

    def __next__(self) -> Circuit:
        if self.n >= self.runs * len(self.depths):
            raise StopIteration
        else:
            circuit, ideal_circuit = self.build_circuit(self.depths[self.n % len(self.depths)])
            circuit_init_kwargs = circuit.init_kwargs
            del circuit_init_kwargs['nqubits']
            self.n += 1
            # Distribute the circuit onto the given support.
            bigcircuit = Circuit(self.nqubits, **circuit_init_kwargs)
            bigcircuit.add(circuit.queue)
            return bigcircuit, ideal_circuit

# Define the experiment class for this specific module.
class TransmonRBExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
        init_state: np.ndarray = np.array([[1, 0], [0, 0]]),
        irrep_basis: str = "",
        factory_id: str = "xid"
    ) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel, init_state)
        self.irrep_basis = irrep_basis
        self.s_matr = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.factory_id = factory_id

    
    def fill_datadict(self, circuit: models.Circuit, ideal_circuit: models.Circuit, datadict: dict):
        # FIXME allow change of initial state

        init_state = np.array([[1, 0], [0, 0]], dtype=complex) if self.init_state.shape[0] > 2 else self.init_state

        # state_ad = np.copy(init_state)
        # # Projection onto XYZ part of the state
        # if len(self.irrep_basis) == 3:
        #     state_ad -= np.eye(2, dtype=complex) / 2
        # if len(self.irrep_basis) > 0:
        #     state_ad = np.zeros(init_state.shape, dtype=complex)
        #     for c in self.irrep_basis:
        #         px = paulis_dict[c].reshape(1, -1).conj() @ init_state.reshape(-1, 1) / 2 # px = (Pi, rho) / d
        #         state_ad += px * paulis_dict[c]

        # FIXME MAKE THIS CODE NORMAL (change irrep_basis to indices)
        
    
        # ps_state = density_to_pauli(init_state, norm=False).reshape(-1, 1) / len(init_state) # coeffs for non-norm. paulis / dim
        # ps_state[0][0] = 0
        # ps_state[1][0] = 0

        ps_state = np.array([[0], [0], [0], [0.5]])
        
        # FIXME make S^+ P_\\lam \\rho more general (not just for 1 qubit)
        ps_state = self.s_matr @ ps_state.reshape(-1, 1)
        
        comp_state = np.zeros((2, 2), dtype=complex)
        ps = [matrices.I, matrices.X, matrices.Y, matrices.Z]
        for i in range(4):
            comp_state += ps_state[i][0] * ps[i] 
            
        # for i in range(4):
        #     comp_state += ps_state[i][0] * list(paulis_dict.values())[i]

        # state_vec = s_matr @ vectorization(state_ad, order="system") # .reshape(-1, 1)
        # state_ad = unvectorization(state_vec, order="system") 
        # for i in range(2):
        #     for j in range(2):
        #         state_ad[i][j] = state_vec[2 * i + j]

        # Ideal final state for this circuit
        ideal_exec = ideal_circuit.execute(comp_state)
        ideal_state = ideal_exec.state()

        # Store filter functions for outcomes 0 and 1
        datadict["filter0"] = ideal_state[0][0]
        datadict["filter1"] = ideal_state[1][1]

        return datadict


    def markovian_single_task(self, circuit: models.Circuit, datarow: dict) -> dict:
        # Create an 'ideal' circuit for the target qubit
        ideal_circuit = circuit.copy(deep=True) 
        # Store samples of a noisy circuit and its depth 
        datadict = super().single_task(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1 if circuit.ngates > 1 else 0

        datadict = self.fill_datadict(circuit, ideal_circuit, datadict)

        return datadict


    def single_task(self, circuits, datarow) -> dict:
        # Check for Markovian case
        if isinstance(circuits, models.Circuit):
            return self.markovian_single_task(circuits, datarow)

        # Non-Markovian case, when the factory returns pair of circuits: ideal and noisy
        circuit, ideal_circuit = circuits
        # Store samples of a noisy circuit and its depth 
        datadict = super().single_task(circuit, datarow)
        datadict["depth"] = circuit.ngates - 1 if circuit.ngates > 1 else 0

        datadict = self.fill_datadict(circuit, ideal_circuit, datadict)
        return datadict


# Define the result class for this specific module.
class TransmonRBResult(Result):
    def __init__(self, dataframe: pd.DataFrame, fitting_func, title, ndecays=1, coeffs=[], decays=[]) -> None:
        super().__init__(dataframe)
        self.fitting_func = fitting_func
        self.title = title
        self.ndecays = ndecays
        self.coeffs = coeffs
        self.decays = decays

    def single_fig(self):
        xdata_scatter = self.df["depth"].to_numpy()
        ydata_scatter = self.df["filters"].to_numpy()
        xdata, ydata = self.extract("depth", "filters", "mean")
        myparams = self.my_params()
        self.scatter_fit_fig(xdata_scatter, ydata_scatter, xdata, ydata, ndecays=self.ndecays, myparams=myparams)

    def my_params(self):
        xs = list(np.linspace(1, 30, 100))
        ys = []
        for x in xs:
            y = 0
            for i in range(len(self.coeffs)):
                y += np.real(self.coeffs[i] * (self.decays[i] ** x))
            ys.append(y)
        return [xs, ys]
    
    def return_decay_parameters(self):
        xdata, ydata = self.extract("depth", "filters", "mean")
        popt, pcov, _, _ = self.fitting_func(xdata, ydata)

        return popt, pcov


def single_qubit_filter(experiment: Experiment):
    filters_list = []
    for datarow in experiment.data:
        samples = datarow["samples"]
        filter_value = 0
        for s in samples:
            if s[0]:
                filter_value += datarow["filter1"]
            else: 
                filter_value += datarow["filter0"]
        filters_list.append(np.real(filter_value / len(samples)))
    experiment._append_data("filters", filters_list)


def analyze(
    experiment: Experiment, noisemodel: NoiseModel = None, **kwargs
) -> go._figure.Figure:
    experiment.apply_task(single_qubit_filter)
    result = TransmonRBResult(experiment.dataframe, kwargs.get("fitting_func"), kwargs.get("title"), kwargs.get("ndecays"), kwargs.get("coeffs"), kwargs.get("decays"))
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
    noise = None,
    init_state: np.ndarray = np.array([[1, 0], [0, 0]], dtype=complex),
    title: str = "Filtered Randomized Benchmarking",
    coeffs: list = [],
    decays: list = []
):
    fitting_func = fit_exp2_func
    ndecays = 2
    if len(init_state) != 2 ** nqubits:
        while len(init_state) < 2 ** nqubits:
            init_state = np.kron(init_state, init_state)
    # Initiate the circuit factory and the faulty Experiment object.

    factory = TransmonXIdFactory(nqubits, depths, runs, qubits)
    experiment = TransmonRBExperiment(factory, nshots, noisemodel=noise, init_state=init_state)
    # Execute the experiment.
    experiment.execute()
    analyze(experiment, noisemodel=noise, fitting_func=fitting_func, title=title, ndecays=ndecays, coeffs=coeffs, decays=decays).show()

perform(1, range(1, 31), 100, 1024)
