from copy import deepcopy
import numpy as np
from scipy.linalg import expm
from sympy import *
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger
import pdb

from constants import *


def create_superoperator(gate_matr=np.eye(2), js=[0, 0, 0], gs=[0, 0]):
    """ Create superoperator L s.t. d/dt rho = L rho for a concrete GKSL equation"""
    # Constants
    jx = js[0]
    jy = js[1]
    jz = js[2]
    g0 = gs[0]
    g1 = gs[1]
    # g0 = 0 #1 / T2 # 1e9 / 8385 # T2 = 8385 nanoseconds
    # g1 = 0 #1 / (1000 * T2) # 1e9 / 8385 # T2 = 8385 nanoseconds

    # # Create hamiltonian H=o0/2*g0 + o1/2*z1 + Jx0x1
    hamiltonian = np.array([[jz, 0, 0, jx-jy],
                            [0, -jz, jx+jy, 0],
                            [0, jx+jy, -jz, 0],
                            [jx-jy, 0, 0, jz]], dtype=complex)

    # hamiltonian -= (FREQ0 / 2) * np.kron(s3, np.eye(2, dtype=complex))
    # hamiltonian -= (FREQ1 / 2) * np.kron(np.eye(2, dtype=complex), s3)
    # hamiltonian = np.array([[jz, 0, jx-1j*jy, 0],
    #                         [0, jz, 0, jx-1j*jy],
    #                         [jx+1j*jy, 0, -jz, 0],
    #                         [0, jx+1j*jy, 0, -jz]], dtype=complex)
    # hamiltonian += (o0 / 2) * np.kron(gate_matr, np.eye(2))    
    gate_to_hamiltonian = ham_from_unitary(gate_matr, 1) / GATE_TIME

    # gate_to_hamiltonian = np.kron(gate_to_hamiltonian, np.eye(2))
    # # Check hamiltonian
    exph = expm(-1j * gate_to_hamiltonian * GATE_TIME)
    if not np.allclose(exph, (gate_matr), atol=1e-7):
        print("WRONG HAMILTONIAN FOR ", gate_matr.round(3))

    hamiltonian += gate_to_hamiltonian

    # GKSL equation
    def get_rho_dot(rho):
        # Computer the commutator
        com = hamiltonian @ rho - rho @ hamiltonian
        
        # x0rx0
        # x0 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        # xrho0 = x0 @ rho @ x0.T
        

        # z0rz0
        z0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        zrho0 = z0 @ rho @ z0.T

        # z1rz1
        z1 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        zrho1 = z1 @ rho @ z1.T
        
        # Compute the result
        res = -1j * com
        res += g0 * (zrho0 - rho)
        res += g1 * (zrho1 - rho)
        return res

    # Calculate <k| L[ |i><j| ] |l>
    l_matr = np.zeros((16, 16), dtype=complex)
    for k in range(4):
        k_vec = (basis[:, k]).reshape(1, -1)
        for l in range(4):
            l_vec = (basis[:, l]).reshape(-1, 1)
            for i in range(4):
                i_vec = (basis[:, i]).reshape(-1, 1)
                for j in range(4):
                    j_vec = (basis[:, j]).reshape(1, -1)
                    ij_matr = i_vec @ j_vec
                    L_ij = get_rho_dot(ij_matr)
                    l_matr[k * 4 + l][i * 4 + j] = (k_vec @ L_ij @ l_vec)[0][0]

    # Calculate exp(dt * L)
    exp_l = expm(GATE_TIME * l_matr)

    # Transform to pauli
    paulis_nq = get_paulis(2)
    res = []
    for p in paulis_nq:
        p_row = p.reshape(1, -1).conj()
        res.append(p_row[0])
    u_p = np.array(res)

    expl_pauli = u_p @ exp_l @ u_p.T.conj()

    return expl_pauli


def get_nonmark_fourier(qubits_list=[0], irrep_labels=[0], js=[0, 0, 0], gs=[0, 0], ps=0):
    p0 = ps[0]
    p1 = ps[1]

    # Get the group of gates
    gates_list = deepcopy(cliffords_list_2q) # get_cliffords_list(qubits_list)
    gates_list_ideal = cliffords_dict_ideal[str(qubits_list)]
    # get_cliffords_list_ideal(qubits_list)
    noisy_gates = [] #noisy_gates_dict[gates_key]


    # Get omega(g) and tau(g)
    inds = get_irrep_inds(irrep_labels, qubits_list)
    # gate_2q = np.kron(gates_list[0], np.eye(2))
    pauli_repr = to_pauli_op(gates_list[0])
    ideal_pauli_repr = to_pauli_op(gates_list_ideal[0])
    irrep = ideal_pauli_repr[np.repeat(inds, len(inds)), np.tile(inds, len(inds))].reshape((len(inds), len(inds)))
    # Get phi(g) and tau(g)^dagger tensor phi(g)
    if len(noisy_gates) == 0 or arreqclose_in_list(gates_list[0], noisy_gates):
        pauli_repr = create_superoperator(gates_list[0], js, gs)
    if len(inds) == 1:
        fourier = np.conj(irrep[0,0]) * pauli_repr 
    else:
        fourier = np.kron(irrep.conj(), pauli_repr) 
    # Repeat for all the gates 
    for i in range(1, len(gates_list)):
        # gate_2q = np.kron(gates_list[i], np.eye(2))
        pauli_repr = to_pauli_op(gates_list[i])
        ideal_pauli_repr = to_pauli_op(gates_list_ideal[i])
        irrep = ideal_pauli_repr[np.repeat(inds, len(inds)), np.tile(inds, len(inds))].reshape((len(inds), len(inds)))
        if len(noisy_gates) == 0 or arreqclose_in_list(gates_list[i], noisy_gates):
            pauli_repr = create_superoperator(gates_list[i], js, gs)
        if len(inds) == 1:
            fourier += np.conj(irrep[0,0]) * pauli_repr 
        else:
            fourier += np.kron(irrep.conj(), pauli_repr)
        # print(irrep.round(3))
        # pdb.set_trace()

    # Uniform probability = 1/|G| 
    fourier /= len(gates_list)

    d_p0 = np.array([[1, 0, 0, 0], [0, 1-p0, 0, 0], [0, 0, 1-p0, 0], [0, 0, 0, 1-p0]], dtype=complex)
    d_p1 = np.array([[1, 0, 0, 0], [0, 1-p1, 0, 0], [0, 0, 1-p1, 0], [0, 0, 0, 1-p1]], dtype=complex)
    fourier = np.kron(np.eye(irrep.shape[0], dtype=complex), np.kron(d_p0, d_p1)) @ fourier
    return fourier


def get_nonmark_decays(qubits_list=[0], irrep_labels=[0], js=[0, 0, 0], gs=[0, 0], ps=[0, 0], round=7):
    fourier = get_nonmark_fourier(qubits_list, irrep_labels, js, gs, ps)
    eigvals = np.linalg.eigvals(fourier).round(round)
    # eigvals, eigvects = np.linalg.eig(fourier).round(7)
    eigvals = list(set(eigvals))
    eigvals[::-1].sort()
    
    return np.array(eigvals)


# def decays_and_coeffs_nonm(gates_key='xid', jx=0, jy=0, jz=0, g0=0, g1=0, round=7, povm_key='z', init_density=np.array([[1, 0], [0, 0]])):
#     fourier = get_nonmark_fourier(gates_key, jx,  jy, jz, g0, g1)
#     # A matrix for Z-basis measurement and initial state |0><0|
#     mql = get_mql_nonm(povm_key, gates_key, init_density) # np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
#     # Get the eigenvectors and eigenvalues of the Fourier matrix s.t. r @ np.diag(eigvals) @ l == fourier
#     eigvals, r = np.linalg.eig(fourier)
#     l = np.linalg.inv(r)
#     # Find the indices corresponing to the obtained eigenvalues in descending order
#     sorted_indices = np.abs(eigvals).argsort()[::-1]
    
#     # A,B
#     ab = l[sorted_indices, :] @ mql @ r[:, sorted_indices]
#     coeffs = np.array([ab[i][i] for i in range(len(ab))])
#     decays = eigvals[sorted_indices]

#     sorted_indices = np.abs(coeffs * (decays ** 2)).argsort()[::-1]
#     coeffs = coeffs[sorted_indices]
#     decays = decays[sorted_indices]
#     all_coeffs = []
#     all_decays = []
#     for i in range(len(coeffs)):
#         if np.abs(coeffs[i]) > 1e-5 and np.abs(decays[i]) > 1e-5:
#             d_ind = -1
#             for j in range(len(all_decays)):
#                 if np.abs(decays[i] - all_decays[j]) < 1e-5:
#                     d_ind = j
#                     break
            
#             if d_ind != -1:
#                 all_coeffs[d_ind] += coeffs[i]
#             else:
#                 all_coeffs.append(coeffs[i])
#                 all_decays.append(decays[i])

#     coeffs_real = []
#     coeffs_imag = []
#     decays_real = []
#     decays_imag = []
#     for i in range(len(all_coeffs)):
#         if np.abs(all_coeffs[i]) > 1e-5 or np.abs(all_decays[i]) > 1e-5:
#             coeffs_real.append(np.round(np.real(all_coeffs[i]), 3))
#             coeffs_imag.append(np.round(np.imag(all_coeffs[i]), 3))
#             decays_real.append(np.real(all_decays[i]))
#             decays_imag.append(np.imag(all_decays[i]))
#     # print(np.round(all_coeffs, 3))
#     # print(np.round(all_decays, 3))

#     coeff_abs_vals = np.array(coeffs_real) ** 2 + np.array(coeffs_imag) ** 2

#     return decays_real, decays_imag, coeff_abs_vals, coeffs_real, coeffs_imag


# print(get_nonmark_fourier('cliffords', 10, 10, 10).round(3))