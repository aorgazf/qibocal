from qibo import models, gates
from qibo.models import Circuit
from qibo.gates import UnitaryChannel
from qibo.noise import NoiseModel, PauliError
import numpy as np
import pdb

init_state = np.array([[1/4, 0], [0, 3/4]])

c2 = Circuit(2, density_matrix=True)
c2.add(gates.X(0))
c2.add(gates.X(1))
c = c2.light_cone(0)[0]

# import pdb
# pdb.set_trace()
# c.add(gates.X(0))
# c.add(gates.RZ(0, np.pi/2))
# c.add(c2.light_cone(0))

c.add(gates.M(0))
print(c.draw())
state_ad = init_state - np.eye(2) / 2
ex = c.execute(state_ad)
print(ex.state().round(3))

h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
print("\nHadamard:\n", (h @ state_ad @ h.T.conj()).round(3))

# x = np.array([[0, 1], [1, 0]])
# print("\nX:\n", (x @ state_ad @ x.T.conj()).round(3))

# y = np.array([[0, -1j], [1j, 0]])
# print("\nY:\n", (y @ state_ad @ y.T.conj()).round(3))

# z = np.array([[1, 0], [0, -1]])
# print("\nZ:\n", (z @ state_ad @ z.T.conj()).round(3))

rx2 = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
print("\nRX(π/2):\n", (rx2 @ state_ad @ rx2.T.conj()).round(3))



# class UnitaryError:
#     """Quantum error associated with the `qibo.gates.UnitaryChannel`.
#     Args:
#         options (tuple): see :class:`qibo.gates.UnitaryChannel`
#     """

#     def __init__(self, probabilities, unitary_matrix):
#         self.options = [probabilities, unitary_matrix]

#     def channel(self, qubits, probabilities, unitary_matrix):
#         return UnitaryChannel(probabilities, [([qubits], unitary_matrix)])

# class KrausError:
#     """Quantum error associated with the `qibo.gates.UnitaryChannel`.
#     Args:
#         options (tuple): see :class:`qibo.gates.UnitaryChannel`
#     """

#     def __init__(self, ops):
#         self.options = ops

#     def channel(self, qubits, ops):
#         return gates.KrausChannel(ops)

# class Error:
#     def __init__(self, options, channel):
#         self.options = options
#         self._channel = channel

#     def channel(self, qubits, ops):
#         return self._channel

# # def apply(noise, circuit):
# #     noisy_circuit = circuit.__class__(**circuit.init_kwargs)
# #     for gate in circuit.queue:
# #         noisy_circuit.add(gate)
# #         if gate.__class__ in noise.errors:
# #             error, qubits = noise.errors.get(gate.__class__)
# #             if qubits is None:
# #                 qubits = gate.qubits
# #             else:
# #                 qubits = tuple(set(gate.qubits) & set(qubits))
# #             noisy_circuit.add(error.channel(error.options[0], [(qubits, error.options[1])]))
# #     noisy_circuit.measurement_tuples = dict(circuit.measurement_tuples)
# #     noisy_circuit.measurement_gate = circuit.measurement_gate
# #     return noisy_circuit

# # Build specific noise model with Unitary error
# # unitary_noise = np.array([[ 0.97015162-0.21719531j, -0.00191304+0.1078349j], [0.05692658+0.09160453j, 0.94430668-0.31089045j]])
# # noise = NoiseModel()
# # noise.add(UnitaryError([1], unitary_noise), gates.X)

# # define a sqrt(0.4) * X gate
# a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
# # define a sqrt(0.6) * CNOT gate
# a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
#                               [0, 0, 0, 1], [0, 0, 1, 0]])
# # define the nosie rho -> 0.4 X{1} rho X{1} + 0.6 CNOT{0, 2} rho CNOT{0, 2}
# noise = NoiseModel()
# # noise.add(KrausError([[((0,), a1), ((0, 2), a2)]]), gates.X)

# channel = gates.KrausChannel([((1,), a1), ((0, 2), a2)])
# error = Error([1], channel)
# noise.add(error, gates.X)

# # Generate noiseless circuit
# c = models.Circuit(3, density_matrix=True)
# c.add([gates.X(0), gates.H(1), gates.CZ(0, 1), gates.X(1), gates.X(2)])
# c.add([gates.X(0)])

# # Apply noise to the circuit according to the noise model
# # noisy_c = apply(noise, c)
# noisy_c = noise.apply(c)
# print(noisy_c.draw())
# print(noisy_c.summary())

'''
from noise import NoiseModel, UnitaryError, CustomError, KrausError
import models
from models import Circuit
import numpy as np
import gates

# define a sqrt(0.4) * X gate
a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
# define a sqrt(0.6) * CNOT gate
a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 0, 1], [0, 0, 1, 0]])

# Build specific noise model with Kraus error rho -> 0.4 X{1} rho X{1} + 0.6 CNOT{0, 2} rho CNOT{0, 2}
# noise = NoiseModel()
# error = KrausError([((1,), a1), ((0, 2), a2)])
# noise.add(error, gates.X)

# channel = gates.KrausChannel([((1,), a1), ((0, 2), a2)])
# error = Error([gates.X, gates.H])
# noise.add(error, gates.Y)




# # define |0><0|
# a1 = np.array([[1, 0], [0, 0]])
# # define |0><1|
# a2 = np.array([[0, 1], [0, 0]])

# # Create an Error associated with Kraus Channel rho -> |0><0| rho |0><0| + |0><1| rho |0><1|
# error = Error(gates.KrausChannel([((0,), a1), ((0,), a2)]))

# Build specific noise model with Unitary error
# unitary_noise = np.array([[0.97-0.217j, -0.002+0.108j], [0.057+0.092j, 0.944-0.311j]])
# print(unitary_noise @ unitary_noise.T.conj())
# noise = NoiseModel()
# noise.add(UnitaryError([1], [unitary_noise]), gates.X)

# # Generate noiseless circuit
# c = models.Circuit(3, density_matrix=False)
# c.add([gates.X(0), gates.H(1), gates.CZ(0, 1), gates.X(1), gates.X(2)])



# Generate random unitary matrix
def random_unitary():
    herm_generator = np.random.normal(size=(2, 2)) + 1j * np.random.normal(size=(2,2))
    herm_generator = (herm_generator + herm_generator.T.conj()) / 2

    from scipy.linalg import expm
    lam = 0.1
    return expm(1j * lam * herm_generator)

# u1 = random_unitary()
# u2 = random_unitary()
# u1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# probabilities = [0.3, 0.7]
# unitary_error = UnitaryError(probabilities, [u1, u2])

# # define a sqrt(0.4) * Id gate
# a1 = np.sqrt(0.4) * np.array([[1, 0, 0, 0], 
#                               [0, 1, 0, 0],
#                               [0, 0, 1, 0], 
#                               [0, 0, 0, 1]])

# # # define a sqrt(0.6) * Id gate
# a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], 
#                               [0, 1, 0, 0],
#                               [0, 0, 1, 0], 
#                               [0, 0, 0, 1]])

# define a sqrt(0.4) * Id gate
# a1 = np.sqrt(0.4) * np.array([[1, 0], [0, 1]])
# # define a sqrt(0.6) * Id gate
# a2 = np.sqrt(0.6) * np.array([[1, 0], [0, 1]])

# # define a sqrt(0.6) * Id gate

# unitary_error = KrausError([a1, a2])

# noise = NoiseModel()
# # noise.add(unitary_error, gates.X, 1)
# noise.add(unitary_error, gates.CNOT)
# noise.add(unitary_error, gates.Z, (0, 1))

# circuit = Circuit(3, density_matrix=True)
# circuit.add(gates.CNOT(0, 1))
# circuit.add(gates.Z(1))
# circuit.add(gates.X(1))
# circuit.add(gates.X(2))
# circuit.add(gates.Z(2))
# circuit.add(gates.M(0, 1, 2))

# target_circuit = Circuit(3, density_matrix=True)
# target_circuit.add(gates.CNOT(0, 1))
# target_circuit.add(gates.UnitaryChannel(probabilities, [([0], u1), ([0], u2)]))
# target_circuit.add(gates.UnitaryChannel(probabilities, [([1], u1), ([1], u2)]))
# target_circuit.add(gates.Z(1))
# target_circuit.add(gates.UnitaryChannel(probabilities, [([1], u1), ([1], u2)]))
# target_circuit.add(gates.X(1))
# target_circuit.add(gates.UnitaryChannel(probabilities, [([1], u1), ([1], u2)]))
# target_circuit.add(gates.X(2))
# target_circuit.add(gates.Z(2))
# target_circuit.add(gates.M(0, 1, 2))

a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 0, 1], [0, 0, 1, 0]])
error_channel = gates.KrausChannel([([1], a1), ([0, 2], a2)])
custom_error = CustomError(error_channel)

noise = NoiseModel()
noise.add(custom_error, gates.X, 1)
noise.add(custom_error, gates.CNOT)
noise.add(custom_error, gates.Z, (0, 1))

circuit = Circuit(3, density_matrix=True)
circuit.add(gates.CNOT(0, 1))
circuit.add(gates.Z(1))
circuit.add(gates.X(1))
circuit.add(gates.X(2))
circuit.add(gates.Z(2))
circuit.add(gates.M(0, 1, 2))

target_circuit = Circuit(3, density_matrix=True)
target_circuit.add(gates.CNOT(0, 1))
target_circuit.add(error_channel)
target_circuit.add(gates.Z(1))
target_circuit.add(error_channel)
target_circuit.add(gates.X(1))
target_circuit.add(error_channel)
target_circuit.add(gates.X(2))
target_circuit.add(gates.Z(2))
target_circuit.add(gates.M(0, 1, 2))

# Apply noise to the circuit according to the noise model
noisy_c = noise.apply(circuit)
print(noisy_c.draw())
# print(noisy_c.summary())
executed = noisy_c(nshots=10)
print(executed.samples())
print(np.sum(executed.probabilities()))

# Apply noise to the circuit according to the noise model
print(target_circuit.draw())
# print(noisy_c.summary())
executed = target_circuit(nshots=10)
print(executed.samples())
print(np.sum(executed.probabilities()))
'''