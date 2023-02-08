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

from qibo.quantum_info.basis import *
from qibo.quantum_info.superoperator_transformations import vectorization, unvectorization

from fourier_info import get_s, density_to_pauli


class NonMarkovianFactory(Circuitfactory):
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
        a = gates_dict[self.factory_id]
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
        random_ints = np.random.randint(0, len(self.gate_set), size=depth)
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = np.take(self.gate_set, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(*range(len(self.qubits))))
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


class CustomFactory(Circuitfactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None, gate_set: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits)
        self.gate_set = gate_set

    def build_circuit(self, depth: int):
        # Initiate the empty circuit from qibo with 'self.nqubits'
        # many qubits.
        circuit = Circuit(len(self.qubits), density_matrix=True)
        # There are only two gates to choose from.
        a = self.gate_set
        # Draw sequence length many zeros and ones.
        random_ints = np.random.randint(0, len(self.gate_set), size=depth)
        # Get the Xs and Ids with random_ints as indices.
        gate_lists = np.take(a, random_ints)
        # Add gates to circuit.
        circuit.add(gate_lists)
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


class SingleCliffordsFilterFactory(SingleCliffordsFactory):
    def __init__(
        self, nqubits: int, depths: list, runs: int, qubits: list = None
    ) -> None:
        super().__init__(nqubits, depths, runs, qubits)

    def build_circuit(self, depth: int):
        circuit = Circuit(len(self.qubits), density_matrix=True)
        for _ in range(depth):
            circuit.add(self.gates())
        
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit


def get_single_clifford_gates(qubit_ind=0):
    '''Generate a list of single qubit cliffords''' 
    res_list = []
    factory = SingleCliffordsFactory(1, [1], 1)
    for el in ONEQUBIT_CLIFFORD_PARAMS:
        unitary_gate = gates.Unitary(factory.clifford_unitary(*el), qubit_ind)
        res_list.append(unitary_gate)
    return res_list


gates_keys = ["cliffords", "xid", "three", "four", "paulis", "x", "id"]

paulis_list = [
    np.eye(2),
    np.array([[0,1],[1,0]]),
    np.array([[0,-1j],[1j,0]]),
    np.array([[1,0],[0,-1]])
]

pauli_group = [gates.I(0), gates.X(0), gates.Y(0), gates.Z(0)]
for s in paulis_list:
    pauli_group.append(gates.Unitary(-s, 0))
    pauli_group.append(gates.Unitary(1j * s, 0))
    pauli_group.append(gates.Unitary(-1j * s, 0))


gates_dict = {
    "cliffords": get_single_clifford_gates(0),
    "xid": [gates.I(0), gates.X(0)],
    "three": [gates.I(0), gates.RX(0, 2 * np.pi / 3), gates.X(0), gates.RX(0, 4 * np.pi / 3)],
    "four": [gates.I(0), gates.RX(0, np.pi / 2), gates.X(0), gates.RX(0, 3 * np.pi / 2)],
    "paulis": pauli_group, 
    "x": [gates.X(0)],
    "id": [gates.I(0)],
}

noisy_gates_dict = {
    "cliffords": [],
    "xid": [gates.X],
    "three": [gates.RX, gates.X],
    "four": [gates.RX, gates.X],
    "paulis": [gates.X, gates.Y, gates.Unitary],
    "x": [gates.X],
    "id": [gates.I],
}

irrep_dict = {
    "cliffords": 'xyz',
    "xid":'yz',
    "three": 'z',
    "four": 'z',
    "paulis": 'z',
    "x": 'yz',
    "id": ''
}

paulis_dict = {
    "i": matrices.I,
    "x": matrices.X,
    "y": matrices.Y,
    "z": matrices.Z
}

multiplicities_dict = {
    "cliffords": 1,
    "xid": 2,
    "three": 2,
    "four": 2,
    "paulis": 1,
    "x": 2,
    "id": 4
}


# Define the experiment class for this specific module.
class FilterRBExperiment(Experiment):
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
        self.s_matr = get_s(gates_key=factory_id)
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
        
        if self.factory_id == 'four':
            ps_state = density_to_pauli(init_state, norm=True).reshape(-1, 1)
            basis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1j/np.sqrt(2), -1j/np.sqrt(2)], [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]]) 
            sqrt_state = basis.T.conj() @ ps_state
            ps_state = np.zeros(sqrt_state.shape, dtype=complex)
            ps_state[-1][0] = sqrt_state[-1][0]
            self.s_matr = basis.T.conj() @ self.s_matr @ basis
            ps_state = self.s_matr @ ps_state
            ps_state = basis @ ps_state
            ps_state /= np.sqrt(2)
        elif self.factory_id == 'three':
            ps_state = density_to_pauli(init_state, norm=True).reshape(-1, 1)
            basis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/np.sqrt(2), -1j/np.sqrt(2)], [0, 0, 1j/np.sqrt(2), -1/np.sqrt(2)]])
            sqrt_state = basis.T.conj() @ ps_state
            ps_state = np.zeros(sqrt_state.shape, dtype=complex)
            ps_state[-1][0] = sqrt_state[-1][0]
            s_matr = basis.T.conj() @ self.s_matr @ basis
            ps_state = s_matr @ ps_state
            ps_state = basis @ ps_state
            ps_state /= np.sqrt(2)
        else:
            ps_state = density_to_pauli(init_state, norm=False).reshape(-1, 1) / len(init_state) # coeffs for non-norm. paulis / dim
            if 'i' not in self.irrep_basis:
                ps_state[0][0] = 0
            if 'x' not in self.irrep_basis:
                ps_state[1][0] = 0
            if 'y' not in self.irrep_basis:
                ps_state[2][0] = 0
            if 'z' not in self.irrep_basis:
                ps_state[3][0] = 0

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
class FilterRBResult(Result):
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
        import pdb
        pdb.set_trace()
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
    result = FilterRBResult(experiment.dataframe, kwargs.get("fitting_func"), kwargs.get("title"), kwargs.get("ndecays"), kwargs.get("coeffs"), kwargs.get("decays"))
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
    factory_id: str = gates_keys[0],
    gates_list: list = [],
    coeffs: list = [],
    decays: list = []
):
    if multiplicities_dict[factory_id] == 1:
        fitting_func = fit_exp1_func 
        ndecays = 1
    elif multiplicities_dict[factory_id] == 2:
        fitting_func = fit_exp2_func
        ndecays = 2
    else: 
        fitting_func = fit_expn_func
        ndecays = 4
    if len(init_state) != 2 ** nqubits:
        while len(init_state) < 2 ** nqubits:
            init_state = np.kron(init_state, init_state)
    # Initiate the circuit factory and the faulty Experiment object.
    if len(gates_list) != 0:
        fitting_func = fit_exp2_func
        ndecays = 2
        factory = NonMarkovianFactory(nqubits, depths, runs, qubits=qubits, factory_id=factory_id, gate_set=gates_list)
    elif factory_id == gates_keys[0]: # Cliffords
        factory = SingleCliffordsFilterFactory(nqubits, depths, runs, qubits=qubits)
    else: 
        gate_set = gates_dict[factory_id] if factory_id in gates_keys else [gates.I(0)]
        factory = CustomFactory(nqubits, depths, runs, qubits=qubits, gate_set=gate_set)

    # SingleCliffordsFilterFactory(nqubits, depths, runs, qubits=qubits)
    irrep_basis = irrep_dict[factory_id]
    experiment = FilterRBExperiment(factory, nshots, noisemodel=noise, init_state=init_state, irrep_basis=irrep_basis, factory_id=factory_id)
    # Execute the experiment.
    experiment.execute()
    analyze(experiment, noisemodel=noise, fitting_func=fitting_func, title=title, ndecays=ndecays, coeffs=coeffs, decays=decays).show()




### Compute ideal Fourier transform \hat\omega[\tau_{ad}]
# factory = SingleCliffordsFactory(1, [1], 1)


# from sympy import *
# # Create a list of 1-qubit paulis without Id
# s0 = np.eye(2)
# s1 = np.array([[0,1],[1,0]])
# s2 = np.array([[0,-1j],[1j,0]])
# s3 = np.array([[1,0],[0,-1]])
# paulis = np.array([s0, s1, s2, s3]) 

# def to_pauli_op(u=np.eye(2), dtype="complex"):
#     ''' Transfrom a unitary operator to Pauli-Liouville superoperator'''
#     dim = 2
#     nq_paulis = deepcopy(paulis)
    
#     pauli_dim = len(nq_paulis)
#     res = np.zeros((pauli_dim, pauli_dim), dtype=dtype)
#     for i in range(pauli_dim):
#         Pi = (nq_paulis[i])
#         for j in range(pauli_dim):
#             Pj = (nq_paulis[j])
#             # m_el = Pi * (u * Pj * u.T.conj()) 
#             # trace_el = 0
#             # for row in range(m_el.shape[0]):
#             #     trace_el = trace_el + m_el[row,row]
#             res[i][j] = np.trace(Pi @ (u @ Pj @ u.T.conj()) ) / dim
            
#     return res


# def get_fourier():
#     res = np.zeros((12, 12), dtype=complex)
#     for el in ONEQUBIT_CLIFFORD_PARAMS:
#         print(el, '\n')
#         clifford = factory.clifford_unitary(*el)
#         print(clifford.round(3), '\n')
#         pauli_repr = to_pauli_op(clifford)
#         irrep_ad = pauli_repr[1:, 1:]
#         # print(pauli_repr.real.round(3))
#         res += np.kron(irrep_ad.T.conj(), pauli_repr) / len(ONEQUBIT_CLIFFORD_PARAMS)
#     return res

# fourier_pauli = get_fourier()
# for row in fourier_pauli:
#     for el in row:
#         if np.abs(el) < 1e-4:
#             print("0", end="\t")
#         elif np.abs(np.imag(el)) < 1e-4:
#             print(np.round(el.real, 4), end="\t")
#         else:
#             print(np.round(el, 3), end="\t")
#     print()
# print(fourier_pauli.round(3))
