import itertools

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibocal.data import DataUnits

ROTATIONS = ["I", "RX", "RX90", "RY90"]
TWO_QUBIT_ROTATIONS = list(itertools.product(ROTATIONS, ROTATIONS))
TWO_QUBIT_ROTATIONS.remove(("RX", "RX"))

BASIS_GATES = {
    "I": None,
    "RX": lambda q: gates.RX(q, theta=np.pi),
    "RX90": lambda q: gates.RX(q, theta=np.pi / 2),
    "RY90": lambda q: gates.RY(q, theta=np.pi / 2),
}

Z = np.array([[1, 0], [0, -1]])
ZZ = np.kron(Z, Z)


def experiment_frequencies(folder, routine):
    """Load experiment data and calculate frequencies from individual shots."""
    data = DataUnits.load_data(folder, "data", routine, "csv", "data").df
    qubits = np.unique(data["qubit"])
    qubit1, qubit2 = min(qubits), max(qubits)
    frequencies = {}
    for rotation1, rotation2 in TWO_QUBIT_ROTATIONS:
        condition = (data["rotation1"] == rotation1) & (data["rotation2"] == rotation2)
        shots1 = np.array(data[condition & (data["qubit"] == qubit1)]["shots"])
        shots2 = np.array(data[condition & (data["qubit"] == qubit2)]["shots"])
        shots = np.stack([shots1, shots2]).T
        values, counts = np.unique(shots, axis=0, return_counts=True)
        frequencies[(rotation1, rotation2)] = {
            f"{v1}{v2}": c for (v1, v2), c in zip(values, counts)
        }
    return frequencies


def circuit_from_sequence(folder, routine):
    """Create a qibo circuit that simulates the tomography procedure."""
    with open(f"{folder}/runcard.yml") as file:
        action_runcard = yaml.safe_load(file)

    experiment_qubits = action_runcard["qubits"]
    sequence = action_runcard["actions"][routine]["sequence"]
    nshots = action_runcard["actions"][routine]["nshots"]

    circuit = Circuit(2)
    for moment in sequence:
        for pulse_description in moment:
            pulse_type, qubit = pulse_description[:2]
            if pulse_type == "FluxPulse":
                # FIXME: FluxPulse is not always one-to-one to CZ gate
                circuit.add(gates.CZ(0, 1))
            else:
                circuit.add(BASIS_GATES[pulse_type](experiment_qubits.index(qubit)))
    return circuit, nshots


def simulate_frequencies(circuit, nshots):
    """Simulate the tomography procedure and obtain the expected shot frequency values."""
    frequencies = {}
    for rotation1, rotation2 in TWO_QUBIT_ROTATIONS:
        c = Circuit(2)
        c.add(circuit.queue)
        gate1 = BASIS_GATES[rotation1]
        gate2 = BASIS_GATES[rotation2]
        if gate1 is not None:
            c.add(gate1(0))
        if gate2 is not None:
            c.add(gate2(1))
        c.add(gates.M(0, 1))
        probs = NumpyBackend().execute_circuit(c).probabilities()
        frequencies[(rotation1, rotation2)] = (nshots * probs).astype(int)
    return frequencies


def shot_frequencies_bar_chart(folder, routine, qubit, format):
    fitting_report = "No fitting data"

    circuit, nshots = circuit_from_sequence(folder, routine)
    simulation_frequencies = simulate_frequencies(circuit, nshots)

    frequencies = experiment_frequencies(folder, routine)

    labels = ["00", "01", "10", "11"]
    titles = [f"({r1}, {r2})" for r1, r2 in TWO_QUBIT_ROTATIONS]
    fig = make_subplots(rows=3, cols=5, subplot_titles=titles)
    row, col = 1, 1
    color1 = "rgba(0.1, 0.34, 0.7, 0.8)"
    color2 = "rgba(0.7, 0.4, 0.1, 0.6)"
    for rotation1, rotation2 in TWO_QUBIT_ROTATIONS:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=simulation_frequencies[(rotation1, rotation2)],
                name="simulation",
                width=0.5,
                marker_color=color1,
                legendgroup="simulation",
                showlegend=row == 1 and col == 1,
            ),
            row=row,
            col=col,
        )

        freqs = frequencies[(rotation1, rotation2)]
        exp_values = np.array([freqs.get(l, 0) for l in labels])
        fig.add_trace(
            go.Bar(
                x=labels,
                y=exp_values,
                name="experiment",
                width=0.5,
                marker_color=color2,
                legendgroup="experiment",
                showlegend=row == 1 and col == 1,
            ),
            row=row,
            col=col,
        )

        col += 1
        if col > 5:
            row += 1
            col = 1

    # TODO: The following annotation doesn't work
    # We need a way to show the preperation sequence (or circuit.draw())
    # in the report
    # fig.add_annotation(
    #    dict(
    #        font=dict(color="black", size=12),
    #        x=0,
    #        y=1.2,
    #        showarrow=False,
    #        text="Preperation Circuit",
    #        font_family="Arial",
    #        font_size=20,
    #        textangle=0,
    #        xanchor="left",
    #        xref="paper",
    #        yref="paper",
    #        font_color="#5e9af1",
    #        hovertext=circuit.draw(),
    #    )
    # )
    fig.update_layout(barmode="overlay", height=1200)

    return [fig], fitting_report
