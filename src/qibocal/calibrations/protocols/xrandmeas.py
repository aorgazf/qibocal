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


# Define the circuit factory class for this specific module.
class XRandMeasFactory(Circuitfactory):
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

        # rand_meas_int = np.random.randint(0, 2)
        # if rand_meas_int == 1:
        #     circuit.add(gates.S(0).dagger())
        # if rand_meas_int > 0:
        #     circuit.add(gates.H(0))
        
        # For Y-basis we need Sdag*H <- before measurement
        # For X-basis we need H before measurement
        # if X: circuit.add(gates.H)
        # if Y: circuit.add(gates.S.dag(), gates.H)
        circuit.add(gates.M(*range(len(self.qubits))))
        return circuit, 0


# Define the experiment class for this specific module.
class XRandMeasExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable,
        nshots: int = None,
        data: list = None,
        noisemodel: NoiseModel = None,
        init_state: np.ndarray = None,
    ) -> None:
        super().__init__(circuitfactory, nshots, data, noisemodel, init_state)

    def single_task(self, cm_tuple: (models.Circuit, int), datarow: dict) -> dict:
        circuit, meas_int = cm_tuple
        datadict = super().single_task(circuit, datarow)
        circuit.add(gates.S(0).dagger())
        circuit.add(gates.H(0))
        samples_y = circuit_y(
            self.init_state, nshots=self.nshots
        ).samples()
        datadict["samplesy"] = samples_y
        datadict["depth"] = circuit.ngates - 1 if circuit.ngates > 1 else 0
        datadict["countX"] = circuit.gate_types['x']
        datadict["meas"] = meas_int
        return datadict


# Define the result class for this specific module.
class XRandMeasResult(Result):
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
    a = 1
    b = 0
    if experiment.init_state is not None:
        a = experiment.init_state[0][0]
        b = experiment.init_state[0][1]

    filtersing_list = []
    for datarow in experiment.data:
        samples = datarow["samples"]
        countX = datarow["countX"]
        meas = datarow["meas"]
        filtersign = 0
        for i in range(len(samples) // 2):
            # if meas == 0:
            #     filtersign += (-1) ** (countX % 2) * (1 - 3 * a) if s[0] else (-1) ** (countX % 2) * (3 * a - 2) 
            # elif meas == 1:
            #     filtersign += (-1) ** (countX % 2) * (((-1) ** s[0]) * 6 * np.imag(b) - 1) / 2.0
            # else:
            #     filtersign += (-1) ** (countX % 2) * (((-1) ** (s[0] + 1)) * 6 * np.real(b) - 1) / 2.0
            filtersign += (-1) ** (countX % 2 + samples[i][0]) * (2 * a - 1) / 4.0
            filtersign += (-1) ** (countX % 2 + samples[len(samples) // 2 + i][0]) * (b - np.conj(b)) * 1j / 4.0
        # import pdb
        # pdb.set_trace()
        filtersing_list.append(filtersign / len(samples))
    experiment._append_data("filters", filtersing_list)    


def analyze(
    experiment: Experiment, noisemodel: NoiseModel = None, **kwargs
) -> go._figure.Figure:
    experiment.apply_task(filter_sign)
    fitting_func = fit_exp1_func if kwargs.get("ndecays") == 1 else fit_exp2_func
    result = XRandMeasResult(experiment.dataframe, fitting_func, title=kwargs.get("title"), ndecays=kwargs.get("ndecays"))
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
):
    # Initiate the circuit factory and the faulty Experiment object.
    factory = XRandMeasFactory(nqubits, depths, runs, qubits=qubits)
    experiment = XRandMeasExperiment(factory, nshots, noisemodel=noise,
    init_state = np.array([[3 / 4, 3 / 8 - np.sqrt(3) * 1j / 8], [3 / 8 + np.sqrt(3) * 1j / 8, 1 / 4]]))
    # Execute the experiment.
    experiment.execute()
    analyze(experiment, noisemodel=noise, title=title, ndecays=ndecays).show()


nqubits = 1
depths = range(1, 31, 1)
runs = 30
nshots = 1024

def pauli_noise_model(px=0, py=0, pz=0):

    # Define the noise model
    paulinoise = PauliError(px, py, pz)
    noise = NoiseModel()
    noise.add(paulinoise, gates.X)

    # Calculate the expexted decay 1-px-py
    exp_decay = (1 - px - pz, 1 - px - py)
    print(exp_decay)

    # Run the protocol
    perform(nqubits, depths, runs, nshots, noise=noise, 
    title=f"Pauli Noise Error: {px}, {py}, {pz}. Expected decay: {exp_decay}", ndecays=2) 

pauli_noise_model(0.05, 0.01, 0.02)