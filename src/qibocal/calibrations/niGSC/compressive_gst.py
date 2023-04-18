""" Here the standard randomized benchmarking is implemented using the
niGSC (non-interactive gate set characterization) architecture.
"""


from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import pandas as pd
import qibo
from plotly.graph_objects import Figure
from qibo import gates
from qibo.models import Circuit
from qibo.noise import NoiseModel
from qibo.states import CircuitResult

from qibocal.calibrations.niGSC.basics.experiment import (
    Experiment,
    experiment_directory,
)
from qibocal.calibrations.niGSC.basics.plot import Report
from qibocal.calibrations.niGSC.basics.utils import probabilities
from qibocal.config import raise_error

int_to_gate = {
    0: lambda q: gates.I(*q),
    1: lambda q: gates.RX(q[0], np.pi / 2),
    2: lambda q: gates.RY(q[0], np.pi / 2),
    3: lambda q: gates.RX(q[1], np.pi / 2),
    4: lambda q: gates.RY(q[1], np.pi / 2),
    5: lambda q: gates.CZ(*q),
}


class ModuleFactory:
    """Iterator object, when called a random circuit with wanted gate
    distribution is created.
    """

    def __init__(
        self, nqubits: int, depths: list | np.ndarray | int, qubits: list = []
    ) -> None:
        self.nqubits = nqubits if nqubits is not None else len(qubits)
        self.qubits = qubits if qubits else list(range(nqubits))
        if len(self.qubits) != 2:
            raise_error(
                ValueError, f"len(qubits) has to be 2. Got {len(self.qubits)} instead."
            )
        if isinstance(depths, int):
            depths = [depths]
        self.depths = depths
        self.name = "CompressiveGSTFactory"

    def __len__(self):
        return len(self.depths)

    def __iter__(self) -> list:
        self.n = 0
        return self

    def __next__(self) -> list:
        # Check if the stop critarion is met.
        if self.n >= len(self.depths):
            raise StopIteration
        else:
            circuit = self.build_circuit(self.depths[self.n])
            self.n += 1
            return circuit

    def build_circuit(self, depth: int):
        return list(np.random.randint(0, len(int_to_gate), size=depth))


class ModuleExperiment(Experiment):
    def __init__(
        self,
        circuitfactory: Iterable | None,
        data: Iterable | None = None,
        nshots: int | None = 128,
        noise_model: NoiseModel | None = None,
        qubits: list = [0, 1],
    ) -> None:
        super().__init__(circuitfactory, data, nshots, noise_model)
        from qibo.backends import GlobalBackend

        self.nqubits = circuitfactory.nqubits
        self.qubits = circuitfactory.qubits
        self.circuitfactory = list(circuitfactory)
        self.backend = GlobalBackend()

        if isinstance(self.backend.platform, str):
            qibo.set_backend("qibolab", platform="dummy")
            self.backend = GlobalBackend()

        self.platform = self.backend.platform

    def perform(self, sequential_task: Callable[[list, dict], dict]) -> None:
        newdata = []
        for circuit in self.circuitfactory:
            newdata.append(sequential_task(deepcopy(circuit), {}))
        self.data = newdata

    def execute(self, circuit: list, datarow: dict) -> dict:
        # Create pulse sequence
        sequence, circuit_qibo = self.circuit_to_sequence(circuit)

        # Execute the pulse sequence on the platform
        if not self.platform.is_connected:
            self.platform.connect()
            self.platform.setup()

        self.platform.start()
        readout = self.platform.execute_pulse_sequence(sequence, self.nshots)
        self.platform.stop()

        result = CircuitResult(self.backend, circuit_qibo, readout, self.nshots)

        # Register measurement outcomes
        if isinstance(readout, dict):
            gate = circuit_qibo.queue[-1]
            samples = []
            for serial in gate.pulses:
                shots = readout[serial].shots
                if shots is not None:
                    samples.append(shots)
            gate.result.backend = self.backend
            gate.result.register_samples(np.array(samples).T)

        return {"circuit": circuit, "probabilities": probabilities(result.samples())}

    def circuit_to_sequence(self, circuit):
        from qibolab.pulses import PulseSequence

        # Create qibo circuit
        circuit_qibo = Circuit(self.nqubits)
        circuit_qibo.add([int_to_gate[i](self.qubits) for i in circuit])

        # Define PulseSequence
        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)

        for index in circuit:
            # Single qubits gates except Id
            if 0 < index < 5:
                qubit = 0 if index < 3 else 1
                phase = 0 if index % 2 else -np.pi / 2
                sequence.add(
                    self.platform.create_RX90_pulse(
                        qubit,
                        start=max(
                            sequence.get_qubit_pulses(*self.qubits).finish,
                            sequence.get_qubit_pulses(*self.qubits).finish,
                        ),
                        relative_phase=virtual_z_phases[qubit]+phase,
                    )
                )
            # CZ gate
            elif index == 5:
                # create CZ pulse sequence with start time = 0
                (
                    cz_sequence,
                    cz_virtual_z_phases,
                ) = self.platform.create_CZ_pulse_sequence(self.qubits)

                # determine the right start time based on the availability of the qubits involved
                cz_qubits = {*cz_sequence.qubits, *self.qubits}
                cz_start = max(sequence.get_qubit_pulses(*cz_qubits).finish, 0)

                # shift the pulses
                for pulse in cz_sequence.pulses:
                    pulse.start += cz_start

                # add pulses to the sequence
                sequence.add(cz_sequence)

                # update z_phases registers
                for qubit in cz_virtual_z_phases:
                    virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

        # Add measurement pulse
        measurement_start = max(
            sequence.get_qubit_pulses(*self.qubits).finish,
            0,
        )
        m_gate = gates.M(*self.qubits)
        m_gate.pulses = ()

        for qubit in self.qubits:
            MZ_pulse = self.platform.create_MZ_pulse(qubit, start=measurement_start)
            sequence.add(MZ_pulse)
            m_gate.pulses = (*m_gate.pulses, MZ_pulse.serial)

        circuit_qibo.add(m_gate)
        return sequence, circuit_qibo

    def save(self, path: str | None = None) -> str:
        """Creates a path if None given and pickles relevant data from ``self.data``
        and if ``self.circuitfactory`` is a list that one too.

        Returns:
            (str): The path of stored experiment.
        """

        # Check if path to store is given, if not create one. If yes check if the last character
        # is a /, if not add it.
        if path is None:
            self.path = experiment_directory("rb")
        else:
            self.path = path if path[-1] == "/" else f"{path}/"

        self.dataframe.to_csv(f"{self.path}/experiment_data.csv")
        # It is convenient to know the path after storing, so return it.
        return self.path


class moduleReport(Report):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Compressive GST"


def post_processing_sequential(experiment: Experiment):
    pass


def get_aggregational_data(experiment: Experiment) -> pd.DataFrame:
    return pd.DataFrame({})


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
    report = moduleReport()
    # Return the figure the report object builds out of all figures added to the report.
    return report.build()
