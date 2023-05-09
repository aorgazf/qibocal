import json
import shutil
import time
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from qibo import gates
from qibo.config import log
from qibo.models import Circuit
from qibolab import Platform
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal.calibrations.niGSC.basics.fitting import exp1B_func, fit_exp1B_func
from qibocal.calibrations.niGSC.basics.utils import gate_fidelity


def scatter(data):
    if isinstance(data, str):
        with open(data) as json_file:
            data = json.load(json_file)

    fig = go.Figure()
    # fig_traces = []
    runs_data = [
        data[d][r]["hardware_probabilities"]
        for d in data["depths"]
        for r in range(data["runs"])
    ]
    fig.add_trace(
        go.Scatter(
            x=data["depths"] * data["runs"],
            y=runs_data,
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["depths"],
            y=data["groundstate probabilities"],
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    x_fit = np.linspace(
        min(data["depths"]), max(data["depths"]), len(data["depths"]) * 20
    )

    y_fit = np.real(exp1B_func(x_fit, *data["popt"].values()))
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name="".join([f"{key}:{data['popt'][key]:.3f} " for key in data["popt"]]),
            line=go.scatter.Line(dash="dot"),
        )
    )
    fig.update_layout(
        {
            "title": f"Gate fidelity: {data['Gate fidelity']}. Gate fidelity primitive: {data['Gate fidelity primitive']}."
        }
    )
    return fig


int_to_gate = {
    # Virtual gates
    0: lambda q: gates.I(q),
    1: lambda q: gates.Z(q),
    2: lambda q: gates.RZ(q, np.pi / 2),
    3: lambda q: gates.RZ(q, -np.pi / 2),
    # pi rotations
    4: lambda q: gates.X(q),
    5: lambda q: gates.Y(q),
    # pi/2 rotations
    6: lambda q: gates.RX(q, np.pi / 2),
    7: lambda q: gates.RX(q, -np.pi / 2),
    8: lambda q: gates.RY(q, np.pi / 2),
    9: lambda q: gates.RY(q, -np.pi / 2),
    # 2pi/3 rotations
    10: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, 0),  # Rx(pi/2)Ry(pi/2)
    11: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, np.pi),  # Rx(pi/2)Ry(-pi/2)
    12: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, 0),  # Rx(-pi/2)Ry(pi/2)
    13: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, -np.pi),  # Rx(-pi/2)Ry(-pi/2)
    14: lambda q: gates.U3(q, np.pi / 2, 0, np.pi / 2),  # Ry(pi/2)Rx(pi/2)
    15: lambda q: gates.U3(q, np.pi / 2, 0, -np.pi / 2),  # Ry(pi/2)Rx(-pi/2)
    16: lambda q: gates.U3(q, np.pi / 2, -np.pi, np.pi / 2),  # Ry(-pi/2)Rx(pi/2)
    17: lambda q: gates.U3(q, np.pi / 2, np.pi, -np.pi / 2),  # Ry(-pi/2)Rx(-pi/2)
    # Hadamard-like
    18: lambda q: gates.U3(q, np.pi / 2, -np.pi, 0),  # X Ry(pi/2)
    19: lambda q: gates.U3(q, np.pi / 2, 0, np.pi),  # X Ry(-pi/2)
    20: lambda q: gates.U3(q, np.pi / 2, np.pi / 2, np.pi / 2),  # Y Rx(pi/2)
    21: lambda q: gates.U3(q, np.pi / 2, -np.pi / 2, -np.pi / 2),  # Y Rx(pi/2)
    22: lambda q: gates.U3(q, np.pi, -np.pi / 4, np.pi / 4),  # Rx(pi/2)Ry(pi/2)Rx(pi/2)
    23: lambda q: gates.U3(
        q, np.pi, np.pi / 4, -np.pi / 4
    ),  # Rx(-pi/2)Ry(pi/2)Rx(-pi/2)
}


def circuit_to_sequence(platform: AbstractPlatform, nqubits, qubit, circuit):
    # Define PulseSequence
    sequence = PulseSequence()
    virtual_z_phases = defaultdict(int)

    next_pulse_start = 0
    for index in circuit:
        if index == 0:
            continue
        gate = int_to_gate[index](qubit)
        # Virtual gates
        if isinstance(gate, gates.Z):
            virtual_z_phases[qubit] += np.pi
        if isinstance(gate, gates.RZ):
            virtual_z_phases[qubit] += gate.parameters[0]
        # U3 pulses
        if isinstance(gate, gates.U3):
            theta, phi, lam = gate.parameters
            virtual_z_phases[qubit] += lam
            sequence.add(
                platform.create_RX90_pulse(
                    qubit,
                    start=next_pulse_start,
                    relative_phase=virtual_z_phases[qubit],
                )
            )
            virtual_z_phases[qubit] += theta
            sequence.add(
                platform.create_RX90_pulse(
                    qubit,
                    start=next_pulse_start,
                    relative_phase=virtual_z_phases[qubit] - np.pi,
                )
            )
            virtual_z_phases[qubit] += phi
        # RX, RY
        if isinstance(gate, (gates.X, gates.Y)):
            phase = 0 if isinstance(gate, gates.X) else np.pi / 2
            sequence.add(
                platform.create_RX_pulse(
                    qubit,
                    start=next_pulse_start,
                    relative_phase=virtual_z_phases[qubit] + phase,
                )
            )
        if isinstance(gate, (gates.RX, gates.RY)):
            phase = 0 if isinstance(gate, gates.RX) else np.pi / 2
            phase -= 0 if gate.parameters[0] > 0 else np.pi
            sequence.add(
                platform.create_RX90_pulse(
                    qubit,
                    start=next_pulse_start,
                    relative_phase=virtual_z_phases[qubit] + phase,
                )
            )

        next_pulse_start = (sequence.finish + 40) if index == 0 else sequence.finish

    # Add measurement pulse
    measurement_start = sequence.finish

    MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
    sequence.add(MZ_pulse)

    return sequence


def workshop(num=24, depth=10):
    return list(np.random.randint(0, num, size=depth))


def calculate_probabilities(result1):
    """Calculates two-qubit outcome probabilities from individual shots."""
    shots = np.stack([result1.shots]).T
    values, counts = np.unique(shots, axis=0, return_counts=True)
    nshots = np.sum(counts)
    return [cnt / nshots for cnt in counts]


nqubits = 5
qubit = 2
nshots = 10000
depths = [1, 2, 5, 10, 12, 14]
runs = 10
# Define platform and load specific runcard
runcard = "qibolab/src/qibolab/runcards/qw5q_gold_qblox.yml"
timestr = time.strftime("%Y%m%d-%H%M")
shutil.copy(runcard, f"{timestr}_runcard.yml")

platform = Platform("qblox", runcard)

platform.connect()
platform.setup()
platform.start()

start_time = time.time()
execution_number = 0
groundstate_probabilities = []
data = {
    "nqubits": 5,
    "qubits": [qubit],
    "nshots": nshots,
    "depths": depths,
    "runs": runs,
}
for depth in depths:
    data[depth] = []
    groundstate_probabilities.append(0)
    for run in range(runs):
        execution_number += 1
        if execution_number % 30 == 0:
            log.info(f"execution munber {execution_number}, circuit depth {depth}")
            time_elapsed = time.time() - start_time
            total_number_executions = len(depths) * runs
            remaining_time = (
                time_elapsed * total_number_executions / execution_number - time_elapsed
            )
            log.info(
                f"estimated time to completion {int(remaining_time)//60}m {int(remaining_time) % 60}s"
            )

        c = workshop(num=len(int_to_gate), depth=depth)
        sequence = circuit_to_sequence(platform, nqubits, qubit, c)
        results = platform.execute_pulse_sequence(sequence, nshots=nshots)
        probs = (
            np.random.uniform(0.8, 0.9) ** depth
        )  # calculate_probabilities(results[qubit])[0]

        circuit_qibo = Circuit(nqubits)
        circuit_qibo.add([int_to_gate[i](qubit) for i in c])
        circuit_qibo.add(gates.M(qubit))
        result = circuit_qibo(nshots=nshots)
        sim_probs = {k: float(v / nshots) for k, v in result.frequencies().items()}

        # print(c)
        # print(probs)
        # print(sim_probs)
        # print()
        data[depth].append(
            {
                "circuit": [int(x) for x in c],
                "hardware_probabilities": probs,
                "simulation_probabilities": sim_probs,
            }
        )

        groundstate_probabilities[-1] += probs
    groundstate_probabilities[-1] /= runs

data["groundstate probabilities"] = groundstate_probabilities
with open(f"{timestr}_strb.json", "w") as file:
    json.dump(data, file)

platform.stop()
platform.disconnect()

# Fitting
popt, perr = fit_exp1B_func(depths, groundstate_probabilities)
data["popt"] = {
    "A": popt[0],
    "p": popt[1],
    "B": popt[2],
}
data["Gate fidelity"] = "{:.4f}".format(gate_fidelity(data["popt"]["p"]))
data["Gate fidelity primitive"] = "{:.4f}".format(gate_fidelity(data["popt"]["p"]))
with open(f"{timestr}_strb.json", "w") as file:
    json.dump(data, file)

# Report
fig = scatter(data)
fig.show()
