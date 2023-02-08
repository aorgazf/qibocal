from copy import deepcopy
import numpy as np
from scipy.linalg import expm
from sympy import *
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger


from qibo import gates, matrices
from qibo.quantum_info.basis import vectorization, comp_basis_to_pauli
from qibocal.calibrations.protocols.utils import ONEQUBIT_CLIFFORD_PARAMS
from qibocal.calibrations.protocols.abstract import SingleCliffordsFactory

# Dictionaries
from fourier_info import gates_dict, noisy_gates_dict, irrep_dict, povm_dict
# Functions
from fourier_info import to_pauli_op, superop_to_pauli, density_to_pauli, get_povm_list, arreqclose_in_list, get_paulis, get_sl_pinv, get_m, get_xl

import pdb

gate_time = 40 * 1e-9


def get_m_nonm(inds=['z']):
    ''' Get matrix M = sum |E_i)(ee|(E_i|'''
    povm_list = get_povm_list(inds)

    # Environmant states as rows
    env_0 = np.array([[1, 0], [0, 0]])
    env_0 = vectorization(env_0).reshape(1, -1)
    env_1 = np.array([[0, 0], [0, 1]])
    env_1 = vectorization(env_1).reshape(1, -1)
    
    # env_2 = np.array([[0, 1], [0, 0]]) # ?
    # env_2 = vectorization(env_2).reshape(1, -1)
    # env_3 = np.array([[0, 0], [1, 0]])
    # env_3 = vectorization(env_3).reshape(1, -1) # ?
    

    # POVM elements as columns and rows
    ei_col = vectorization(povm_list[0]).reshape(-1, 1)
    ei_row = vectorization(povm_list[0]).reshape(1, -1).conj()
    res = ei_col @ np.kron(env_0, ei_row)
    res += ei_col @ np.kron(env_1, ei_row)
    # res += ei_col @ np.kron(env_2, ei_row)
    # res += ei_col @ np.kron(env_3, ei_row)
    for i in range(len(povm_list)):
        ei_col = vectorization(povm_list[i]).reshape(-1, 1)
        ei_row = vectorization(povm_list[i]).reshape(1, -1).conj()
        res += ei_col @ np.kron(env_0, ei_row)
        res += ei_col @ np.kron(env_1, ei_row)
        # res += ei_col @ np.kron(env_2, ei_row)
        # res += ei_col @ np.kron(env_3, ei_row)
    return res


def get_m_nonm_pauli(inds=['z']):
    ''' Get matrix M = sum |E_i)(ee|(E_i| in Pauli basis'''
    povm_list = get_povm_list(inds)

    # Environmant states as rows
    env_0 = density_to_pauli(np.array([[1, 0], [0, 0]]))
    env_0 = env_0.reshape(1, -1)
    env_1 = density_to_pauli(np.array([[0, 0], [0, 1]])) 
    env_1 = env_1.reshape(1, -1)

    # env_2 = density_to_pauli(np.array([[0, 1], [0, 0]]))
    # env_2 = env_2.reshape(1, -1)
    # env_3 = density_to_pauli(np.array([[0, 0], [1, 0]])) 
    # env_3 = env_3.reshape(1, -1)

    # POVM elements as columns and rows
    povm_pauli = (density_to_pauli(povm_list[0])) 
    ei_col = povm_pauli.reshape(-1, 1)
    ei_row = povm_pauli.reshape(1, -1).conj()

    res = ei_col @ np.kron(ei_row, env_0)
    res += ei_col @ np.kron(ei_row, env_1)
    # res += ei_col @ np.kron(env_2, ei_row)
    # res += ei_col @ np.kron(env_3, ei_row)
    for i in range(1, len(povm_list)):
        povm_pauli = (density_to_pauli(povm_list[i])) 
        ei_col = povm_pauli.reshape(-1, 1)
        ei_row = povm_pauli.reshape(1, -1).conj()
        res += ei_col @ np.kron(ei_row, env_0)
        res += ei_col @ np.kron(ei_row, env_1)
        # res += ei_col @ np.kron(env_2, ei_row)
        # res += ei_col @ np.kron(env_3, ei_row)
    return res


def partial_trace(a, n=2, d=4):
    dim = a.shape[0] // n
    res = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for k in range(n):
                res[i][j] += a[i+k*(d**2)][j+k*(d**2)]
    return res


def get_mql_nonm(povm_key='z', gates_key='xid', init_density=np.array([[1, 0], [0, 0]])):

    real_init_density = np.kron(init_density, np.array([[1, 0], [0, 0]])) # \rho_{SE}

    m_matr = get_m_nonm_pauli(povm_key) # (get_m_nonm(povm_key)) 
    x_l = get_xl(gates_key)
    m_l = x_l.T.conj() @ m_matr

    s_l = get_sl_pinv(povm_key, gates_key)
    state_pauli = density_to_pauli(init_density)
    real_state_pauli = density_to_pauli(real_init_density)
    # state_outer_prod = 
    q_l = (real_state_pauli.reshape(-1, 1) @ state_pauli.reshape(1, -1).conj() @ x_l @ s_l).T.conj()

    # mq_l = np.reshape(m_l, (-1, 1), order="F") @ np.reshape(q_l, (1, -1), order="F")
    mq_l = m_l.reshape(-1, 1) @ q_l.reshape(1, -1).conj()
    # pdb.set_trace()
    if irrep_dict[gates_key][-1] > 1:
        return partial_trace(mq_l, irrep_dict[gates_key][-1])
    else:
        return mq_l


basis = np.eye(4)

def ham_from_unitary(gate=np.eye(2)):
    # Create a 2-qubit matrix
    gate2q = np.kron(gate, np.eye(2)) 
    # Get eigendecomposition of the gate matrix
    w, v = np.linalg.eig(gate2q)
    # Calculate H s.t. G = exp(-iH)
    ham = np.zeros(gate2q.shape, dtype=complex)
    for i in range(len(w)):
        if np.abs(np.imag(w[i])) < 1e-7:
            val = 1j * np.log(np.abs(w[i]))
            if w[i] < 0:
                val += np.pi
        else:
            val = np.arcsin(-np.imag(w[i]))
        

        # val = 1j * np.log(np.abs(w[i])) + np.pi if w[i] < 0 else 1j * np.log(w[i])
        ham += (val) * ((v[:, i]).reshape(-1, 1) @ (np.linalg.inv(v)[i, :]).reshape(1, -1))

    # Check
    exph = expm(-1j * ham)
    if not np.allclose(gate2q, exph, atol=1e-7):
        print("WRONG FOR:\n", gate2q.round(3),'\n', exph.round(3))

    return ham

def create_superoperator(gate_matr=np.eye(2), J=0, g0=0, g1=0):
    """ Create superoperator L s.t. d/dt rho = L rho for a concrete GKSL equation"""
    # dt = 40 * 1e-9
    # Variables
    o1 = 0 #np.pi / gate_time # z1
    
    # Create hamiltonian H=o0/2*g0 + o1/2*z1 + Jx0x1
    hamiltonian = np.array([[o1 / 2, 0, 0, J],
                            [0, -o1 / 2, J, 0],
                            [0, J, o1 / 2, 0],
                            [J, 0, 0, -o1 / 2]], dtype=complex)
    # hamiltonian += (o0 / 2) * np.kron(gate_matr, np.eye(2)) 

    # gate_z2 = np.kron(np.eye(2), np.array([[1, 0], [0, -1]])) 
    # hamiltonian += ham_from_unitary(gate_z2) / dt

    gate_to_hamiltonian = ham_from_unitary(gate_matr)
    # gate_to_hamiltonian = np.kron(gate_to_hamiltonian, np.eye(2))
    hamiltonian += gate_to_hamiltonian / gate_time

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
    exp_l = expm(gate_time * l_matr)

    # Transform to pauli
    # paulis_nq = get_paulis(2)
    # res = []
    # for p in paulis_nq:
    #     p_row = p.reshape(1, -1).conj()
    #     res.append(p_row[0])
    # u_p = np.array(res)

    # Transform to pauli
    # paulis_nq = get_paulis(2)
    # res = []
    # for p in paulis_nq:
    #     p_row = p.reshape(1, -1).conj()
    #     res.append(p_row[0])
    # u_p = np.array(res)

    # expl_pauli = u_p @ exp_l @ u_p.T.conj()
    expl_pauli = superop_to_pauli(exp_l) # u_p @ exp_l @ u_p.T.conj()

    return expl_pauli 


def get_nonmark_fourier(gates_key='xid', J=0, g0=0, g1=0):
    # Get the group of gates
    gates_list = gates_dict[gates_key]
    noisy_gates = noisy_gates_dict[gates_key]

    # Get omega(g) and tau(g)
    irrep_inds = irrep_dict[gates_key]
    gate_2q = np.kron(gates_list[0], np.eye(2))
    pauli_repr = to_pauli_op(gate_2q)
    ideal_pauli_repr = to_pauli_op(gates_list[0])
    irrep = ideal_pauli_repr[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1]

    # Get phi(g) and tau(g)^dagger tensor phi(g)
    if len(noisy_gates) == 0 or arreqclose_in_list(gates_list[0], noisy_gates):
        pauli_repr = create_superoperator(gates_list[0], J, g0, g1)
    if irrep_inds[-1] > 1:
        fourier = np.conj(irrep[0,0]) * pauli_repr
    else:
        fourier = np.kron(irrep.conj(), pauli_repr)
    # pdb.set_trace()
    # Repeat for all the gates 
    for i in range(1, len(gates_list)):
        gate_2q = np.kron(gates_list[i], np.eye(2))
        pauli_repr = to_pauli_op(gate_2q)
        ideal_pauli_repr = to_pauli_op(gates_list[i])
        irrep = ideal_pauli_repr[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1]
        if len(noisy_gates) == 0 or arreqclose_in_list(gates_list[i], noisy_gates):
            pauli_repr = create_superoperator(gates_list[i], J, g0, g1)
        if irrep_inds[-1] > 1:
            fourier += np.conj(irrep[0,0]) * pauli_repr
        else:
            fourier += np.kron(irrep.conj(), pauli_repr)
        # pdb.set_trace()
    # Uniform probability = 1/|G|
    fourier /= len(gates_list)
    
    # Create a dictionary of eigenvalues
    # eigvals = np.linalg.eigvals(fourier).round(3)
    # eigvals[::-1].sort()
    # print(eigvals)
    # res_dict = {}
    # for e in eigvals:
    #     if e == 0:
    #         continue
    #     elif np.imag(e) == 0:
    #         key = str(np.real(e))
    #     else:
    #         key = str(np.real(e)) + "±" + str(np.abs(np.imag(e))) + "i"
    #     if key in res_dict.keys():
    #         res_dict[key] += 1
    #     elif e != 0:
    #         res_dict[key] = 1
    return fourier


def nonm_validation(gates_key='xid', J=0, g0=0, g1=0, povm_key='z', init_density=np.array([[1, 0], [0, 0]])):
    fourier = get_nonmark_fourier(gates_key, J, g0, g1)
    # A matrix for Z-basis measurement and initial state |0><0|
    mql = get_mql_nonm(povm_key, gates_key, init_density) # np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    # Get the eigenvectors and eigenvalues of the Fourier matrix s.t. r @ np.diag(eigvals) @ l == fourier
    eigvals, r = np.linalg.eig(fourier)
    l = np.linalg.inv(r)
    # Find the indices corresponing to the obtained eigenvalues in descending order
    sorted_indices = np.abs(eigvals).argsort()[::-1]
    # Get the number of dominant decays
    n_l = irrep_dict[gates_key][-1] * 4 # n_l * d_E^2
    
    # A,B
    ab = l[sorted_indices, :] @ mql @ r[:, sorted_indices]
    coeffs = np.array([ab[i][i] for i in range(len(ab))])
    # sorted_indices = np.abs(coeffs).argsort()[::-1]

    # # # A: Dominant part
    # r1 = r[:, sorted_indices[:n_l]]
    # l1 = l[sorted_indices[:n_l], :]
    # # pdb.set_trace()
    # a = l1 @ mql @ r1
    # # # B: Subdominant part
    # r2 = r[:, sorted_indices[n_l:]]                                                                                                                                                        
    # l2 = l[sorted_indices[n_l:], :]
    # b = l2 @ mql @ r2
    # # # d = l @ mql @ r
    # # # print([d[i][i].round(3) for i in range(len(d))] )
    # # # # What should be returned?
    # coeffs = np.array([a[i][i] for i in range(len(a))] + [b[i][i] for i in range(len(b))])
    decays = eigvals[sorted_indices]

    sorted_indices = np.abs(coeffs * (decays ** 2)).argsort()[::-1]
    coeffs = coeffs[sorted_indices]
    decays = decays[sorted_indices]
    # print(coeffs.round(3))
    # print(decays.round(3))
    dom_coeffs = []
    dom_decays = []
    for i in range(len(coeffs)):
        if np.abs(coeffs[i]) > 1e-5 and np.abs(decays[i]) > 1e-5:
            d_ind = -1
            for j in range(len(dom_decays)):
                if np.abs(decays[i] - dom_decays[j]) < 1e-5:
                    d_ind = j
                    break
            
            if d_ind != -1:
                dom_coeffs[d_ind] += coeffs[i]
            else:
                dom_coeffs.append(coeffs[i])
                dom_decays.append(decays[i])
    # pdb.set_trace()

    return dom_coeffs, dom_decays

# m = get_m_nonm()
# print(m.round(3).real)
# print(m.shape, '\n')
# m = get_m_nonm_pauli()
# print((m*np.sqrt(2)).round(3).real)
# print(m.shape)
# gates_key = 'xid'
# # # fourier = get_nonmark_fourier(gates_key, 10, 0, 0)
# j = 2 / gate_time
# c, d = nonm_validation(gates_key, j, 0, 0)
# cnt = 0
# for i in range(len(c)):
#     if np.abs(c[i]) > 1e-5 and np.abs(d[i]) > 1e-5:
#         cnt += 1
#         print(cnt, "\n", c[i].round(3), "*", d[i].round(3), "^m\n")

# j =0 #1 / (2 * gate_time)
# cs, ds = nonm_validation('cliffords', j, 0, 0)
# print(np.round(cs, 3))
# print(np.round(ds, 3))
