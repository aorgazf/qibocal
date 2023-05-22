from collections import defaultdict

import numpy as np
from qibo import gates
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.transpilers.unitary_decompositions import u3_decomposition


def _compile_gate(gate, platform, sequence, virtual_z_phases, moment_start):
    """Compiler 'inspired' by Stavros' PR #367"""
    qubit = gate.target_qubits[0]
    # Virtual gates
    if isinstance(gate, gates.I):
        pass
    if isinstance(gate, gates.Z):
        virtual_z_phases[qubit] += np.pi
    if isinstance(gate, gates.RZ):
        virtual_z_phases[qubit] += gate.parameters[0]
    # X
    if isinstance(gate, (gates.X, gates.Y)):
        phase = 0 if isinstance(gate, gates.X) else -np.pi / 2
        sequence.add(
            platform.create_RX_pulse(
                qubit,
                start=moment_start,
                relative_phase=virtual_z_phases[qubit] + phase,
            )
        )
    # RX
    if isinstance(gate, (gates.RX, gates.RY)):
        phase = 0 if isinstance(gate, gates.RX) else -np.pi / 2
        phase += 0 if gate.parameters[0] > 0 else -np.pi
        sequence.add(
            platform.create_RX90_pulse(
                qubit,
                start=moment_start,
                relative_phase=virtual_z_phases[qubit] + phase,
            )
        )
    # U3 pulses
    if isinstance(gate, gates.U3):
        theta, phi, lam = gate.parameters
        virtual_z_phases[qubit] += lam
        sequence.add(
            platform.create_RX90_pulse(
                qubit,
                start=moment_start,
                relative_phase=virtual_z_phases[qubit],
            )
        )
        virtual_z_phases[qubit] += theta
        sequence.add(
            platform.create_RX90_pulse(
                qubit,
                start=sequence.finish,
                relative_phase=virtual_z_phases[qubit] - np.pi,
            )
        )
        virtual_z_phases[qubit] += phi

    return sequence, virtual_z_phases


SINGLE_QUBIT_CLIFFORDS = {
    # Virtual gates
    0: lambda q: gates.I(q),
    1: lambda q: gates.Z(q),
    2: lambda q: gates.RZ(q, np.pi / 2),
    3: lambda q: gates.RZ(q, -np.pi / 2),
    # pi rotations
    4: lambda q: gates.X(q),  # gates.U3(q, np.pi, 0, np.pi),
    5: lambda q: gates.Y(q),  # U3(q, np.pi, 0, 0),
    # pi/2 rotations
    6: lambda q: gates.RX(q, np.pi / 2),  # U3(q, np.pi / 2, -np.pi / 2, np.pi / 2),
    7: lambda q: gates.RX(q, -np.pi / 2),  # U3(q, -np.pi / 2, -np.pi / 2, np.pi / 2),
    8: lambda q: gates.RY(q, np.pi / 2),  # U3(q, np.pi / 2, 0, 0),
    9: lambda q: gates.RY(q, np.pi / 2),  # U3(q, -np.pi / 2, 0, 0),
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


def inverse_single_qubit_circuit(circuit_inds, q=0, gates_dict=None):
    """Returns inverse gate of type :class:`qibo.gates.U3` given a list of int indices."""
    # Get int to gate translation dictionary
    if gates_dict is None:
        gates_dict = SINGLE_QUBIT_CLIFFORDS

    unitary = np.linalg.multi_dot([gates_dict[i](q).matrix for i in circuit_inds[::-1]])
    inverse_unitary = np.transpose(np.conj(unitary))
    theta, phi, lam = u3_decomposition(inverse_unitary)
    return gates.U3(q, theta, phi, lam)


def circuit_to_sequence(platform, qubit, circuit_inds, invert=True, gates_dict=None):
    """Converts a list of int indices to a pulse sequence with inverse gate and measurement."""
    # Get int to gate translation dictionary
    gates_dict = SINGLE_QUBIT_CLIFFORDS if gates_dict is None else gates_dict

    # Define PulseSequence
    sequence = PulseSequence()
    virtual_z_phases = defaultdict(int)

    next_pulse_start = 0
    for index in circuit_inds:
        if index == 0:
            continue
        gate = gates_dict[index](qubit)
        sequence, virtual_z_phases = _compile_gate(
            gate, platform, sequence, virtual_z_phases, next_pulse_start
        )
        next_pulse_start = sequence.finish

    # Inverse the sequence
    if invert:
        invert_gate = inverse_single_qubit_circuit(circuit_inds, qubit, gates_dict)
        sequence, virtual_z_phases = _compile_gate(
            invert_gate, platform, sequence, virtual_z_phases, next_pulse_start
        )

    # Add measurement pulse
    measurement_start = sequence.finish

    MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
    sequence.add(MZ_pulse)

    return sequence
