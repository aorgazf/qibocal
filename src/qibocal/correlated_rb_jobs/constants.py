
import numpy as np
from copy import deepcopy
from scipy.linalg import expm
import pdb

from qibo.quantum_info.basis import vectorization, comp_basis_to_pauli

from constants import * #cliffords_list, gates_dict, noisy_gates_dict, irrep_dict, xl_dict, povm_dict


### Time constants
GATE_TIME = 40 * 1e-9
T1 = 21785 * 1e-9
T2 = 8385 * 1e-9

FREQ0 = 3.92656e+9
FREQ1 = 7.453328e+9

### Gates
def rx(theta):
    ''' Get RX matrix'''
    return np.array([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]])

def ry(theta):
    ''' Get RY matrix'''
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])

# 1-qubit Paulis
s0 = np.eye(2, dtype=complex)
s1 = np.array([[0,1],[1,0]], dtype=complex)
s2 = np.array([[0,-1j],[1j,0]], dtype=complex)
s3 = np.array([[1,0],[0,-1]], dtype=complex)

# Single qubit Cliffords for 1 qubit
cliffords_list = [
    np.array([
    [(1+0j), 0j], 
    [0j, (1+0j)]
    ]),
    np.array([
    [0, -1j], 
    [-1j, 0]
    ]),
    np.array([
    [(6.123233995736766e-17+0j), (-1+0j)], 
    [(1+0j), (6.123233995736766e-17+0j)]
    ]),
    np.array([
    [(-1j), 0j], 
    [0j, (1j)]
    ]),
    np.array([
    [(0.7071067811865476+0j), -0.7071067811865475j], 
    [-0.7071067811865475j, (0.7071067811865476+0j)]
    ]),
    np.array([
    [(0.7071067811865476+0j), 0.7071067811865475j], 
    [0.7071067811865475j, (0.7071067811865476+0j)]
    ]),
    np.array([
    [(0.7071067811865476+0j), (-0.7071067811865475+0j)], 
    [(0.7071067811865475+0j), (0.7071067811865476+0j)]
    ]),
    np.array([
    [(0.7071067811865476+0j), (0.7071067811865475+0j)], 
    [(-0.7071067811865475+0j), (0.7071067811865476+0j)]
    ]),
    np.array([
    [(0.7071067811865476-0.7071067811865475j), 0j], 
    [0j, (0.7071067811865476+0.7071067811865475j)]
    ]),
    np.array([
    [(0.7071067811865476+0.7071067811865475j), 0j], 
    [0j, (0.7071067811865476-0.7071067811865475j)]
    ]),
    np.array([
    [0, (-0.7071067811865475-0.7071067811865475j)], 
    [(0.7071067811865475-0.7071067811865475j), 0]
    ]),
    np.array([
    [(6.123233995736766e-17-0.7071067811865475j), -0.7071067811865475j], 
    [-0.7071067811865475j, (0+0.7071067811865475j)]
    ]),
    np.array([
    [(6.123233995736766e-17-0.7071067811865475j), (-0.7071067811865475+0j)], 
    [(0.7071067811865475+0j), (6.123233995736766e-17+0.7071067811865475j)]
    ]),
    np.array([
    [(6.123233995736766e-17+0j), (-0.7071067811865475+0.7071067811865475j)], 
    [(0.7071067811865475+0.7071067811865475j), (6.123233995736766e-17+0j)]
    ]),
    np.array([
    [(6.123233995736766e-17+0.7071067811865475j), -0.7071067811865475j], 
    [-0.7071067811865475j, (6.123233995736766e-17-0.7071067811865475j)]
    ]),
    np.array([
    [(6.123233995736766e-17-0.7071067811865475j), (0.7071067811865475+0j)], 
    [(-0.7071067811865475+0j), (6.123233995736766e-17+0.7071067811865475j)]
    ]),
    np.array([
    [(0.5000000000000001-0.5j), (-0.5-0.5j)], 
    [(0.5-0.5j), (0.5000000000000001+0.5j)]
    ]),
    np.array([
    [(0.5000000000000001+0.5j), (0.5+0.5j)], 
    [(-0.5+0.5j), (0.5000000000000001-0.5j)]
    ]),
    np.array([
    [(0.5000000000000001-0.5j), (-0.5+0.5j)], 
    [(0.5+0.5j), (0.5000000000000001+0.5j)]
    ]),
    np.array([
    [(0.5000000000000001+0.5j), (0.5-0.5j)], 
    [(-0.5-0.5j), (0.5000000000000001-0.5j)]
    ]),
    np.array([
    [(0.5000000000000001-0.5j), (0.5-0.5j)], 
    [(-0.5-0.5j), (0.5000000000000001+0.5j)]
    ]),
    np.array([
    [(0.5+0.5j), (-0.5+0.5j)], 
    [(0.5+0.5j), (0.5-0.5j)]
    ]),
    np.array([
    [(0.5+0.5j), (-0.5-0.5j)], 
    [(0.5-0.5j), (0.5-0.5j)]
    ]),
    np.array([
    [(0.5-0.5j), (0.5+0.5j)], 
    [(-0.5+0.5j), (0.5+0.5j)]
    ])
]

# Single qubit Cliffords for 2 qubits
cliffords_list_2q = []
for c1 in cliffords_list:
    for c2 in cliffords_list:
        cliffords_list_2q.append(np.kron(c1, c2))

# def get_cliffords_list(qubits_list=[0, 1], nqubits=2):
#     res = deepcopy(cliffords_list) if 0 in qubits_list else [np.eye(2, dtype=complex) for _ in range(len(cliffords_list))]
    
#     for i in range(1, nqubits):
#         if i not in qubits_list:
#             for j in range(len(res)):
#                 res[j] = np.kron(res[j], np.eye(2, dtype=complex)) 
#         else:
#             new_res = []
#             for r in res:
#                 for c in cliffords_list:
#                     new_res.append(np.kron(r, c))

#             res = deepcopy(new_res)
#     return res

def get_cliffords_list_ideal(qubits_list=[0, 1], nqubits=2):
    if len(qubits_list) == 2:
        return deepcopy(cliffords_list_2q)

    res = []
    if 0 in qubits_list:
        for c1 in cliffords_list:
            for _ in range(len(cliffords_list)):
                res.append(c1)
    else:
        for _ in range(len(cliffords_list)):
            for c2 in cliffords_list:
                res.append(c2)
    # for _ in qubits_list:
    #     if len(res):
    #         new_res = []
    #         for r in res:
    #             for c in cliffords_list:
    #                 new_res.append(np.kron(r, c))
    #         res = deepcopy(new_res)
    #     else:
    #         res = deepcopy(cliffords_list)
    return res


# Dictionary of applied gates
cliffords_dict_ideal = {}
for qubits_list in [[0], [1], [0, 1]]:
    cliffords_dict_ideal[str(qubits_list)] = get_cliffords_list_ideal(qubits_list)
    
# Dictionary of gate sets
gates_dict = {
    '0': cliffords_list,
    '1': cliffords_list,
    '01': cliffords_list_2q
}

# Dictionary of noisy gates (I, Z are not noisy)
noisy_gates_dict = {
    '0': [],
    '1': [],
    '01': []
}

# Dictionary of indices corresponding to the irrep of each gate set and multiplicity of this irrep
def get_irrep_inds(irrep_labels=[0], qubits_list=[0]):
    # import pdb
    # pdb.set_trace()
    if len(irrep_labels) != len(qubits_list):
        print("lengths labels and qubit list are different")
    irreps_dict = {
        0: [0],
        1: [1, 2, 3]
    }
    res = irreps_dict[irrep_labels[0]]
    for i in range(1, len(irrep_labels)):
        tmp_inds = irreps_dict[irrep_labels[i]]
        new_res = []
        for r in res:
            for t in tmp_inds:
                new_res.append(4 * r + t)
        res = deepcopy(new_res)
    return np.array(res)


# Dictionary of x_lambda s.t. P_l = X_l*X_l^\dag
xl_dict = {
    'xid': np.array([[0, 0], [0, 0], [1, 0], [0, 1]]),
    'paulis': np.array([[0], [0], [0], [1]]),
    'cliffords': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    'x': np.array([[0, 0], [0, 0], [1, 0], [0, 1]]),
    'id': np.eye(4)
}

# Dictionary of all POVM elements E_i
povm_dict = {
    'z0': np.array([[1, 0], [0, 0]]),
    'z1': np.array([[0, 0], [0, 1]]),
    'x0': np.array([[0.5, 0.5], [0.5, 0.5]]),
    'x1': np.array([[0.5, -0.5], [-0.5, 0.5]]),
    'y0': np.array([[0.5, -1j/2], [1j/2, 0.5]]),
    'y1': np.array([[0.5, 1j/2], [-1j/2, 0.5]])
}


# Create a list of 1-qubit paulis
s0 = np.eye(2, dtype=complex)
s1 = np.array([[0,1],[1,0]], dtype=complex)
s2 = np.array([[0,-1j],[1j,0]], dtype=complex)
s3 = np.array([[1,0],[0,-1]], dtype=complex)
paulis = np.array([s0, s1, s2, s3]) / np.sqrt(2)

PAULI_DIM = 1
PAULIS_NQ = deepcopy(paulis)

basis = np.eye(4)


# Check if numpy array is in list of numpy arrays
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr)), False)


def base_convert(num, base, length=2):
    ''' Convert a number to some base '''
    if not(2 <= base <= 9):
        return

    new_num = ''
    while num > 0:
        new_num = str(num % base) + new_num
        num //= base

    while len(new_num) < length:
        new_num = '0' + new_num 
    return np.array(list(new_num), dtype=int)


def get_paulis(n):
    ''' Get array of all n-qubit Paulis '''
    global PAULI_DIM
    global PAULIS_NQ
    
    if n == PAULI_DIM:
        return PAULIS_NQ
     
    res = []
    for i in range(4 ** n):
        inds = base_convert(i, 4, length=n) 
        el = paulis[inds[0]]
        for j in range(1, n):
            el = np.kron(el, paulis[inds[j]])
        res.append(el)
    
    PAULI_DIM = n
    PAULIS_NQ = res
    return res


def inner_prod(a, b):
    # Calculate (a, b) = sum(a*ij bij)
    res = np.dot(a.conj().reshape(1, -1), b.reshape(-1, 1))
    return complex(res)


def partial_trace(a, n=2, d=4):
    dim = a.shape[0] // n
    res = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for k in range(n):
                res[i][j] += a[i+k*(d**2)][j+k*(d**2)]
    return res


def density_to_pauli(density):
    ''' Returns Pauli vector of a density'''
    qubits_num = int(np.log2(density.shape[0]))
    nq_paulis = get_paulis(qubits_num)
    
    bloch_vec = []
    for i in range(0, len(nq_paulis)):
        el = inner_prod(nq_paulis[i], density)
        bloch_vec.append(el)
    return np.array(bloch_vec)  # / (2 ** qubits_num)


def to_pauli_op(u=np.eye(2), dtype="complex"):
    ''' Transfrom a unitary operator to Pauli-Liouville superoperator'''
    dim = len(u)
    qubits_num = int(np.log2(dim))
    nq_paulis = get_paulis(qubits_num)
    # nq_paulis = deepcopy(paulis)
    
    pauli_dim = len(nq_paulis)
    res = np.zeros((pauli_dim, pauli_dim), dtype=dtype)
    for i in range(pauli_dim):
        Pi = nq_paulis[i]
        for j in range(pauli_dim):
            Pj = nq_paulis[j]
            res[i][j] = np.trace(Pi @ (u @ Pj @ u.T.conj())) # / dim
            
    return res


def superop_to_pauli(a):
    qubits_num = int(np.log2(np.sqrt(a.shape[0])))
    u_p = comp_basis_to_pauli(qubits_num, normalize=True)
    return u_p @ a @ u_p.T.conj()


def ham_from_unitary(gate=np.eye(2), nqubits=1):
    # Get eigendecomposition of the gate matrix
    w, v = np.linalg.eig(gate)
    # Calculate H s.t. G = exp(-iH)
    ham = np.zeros(gate.shape, dtype=complex)
    for i in range(len(w)):
        if np.real(w[i]) > 1:
            w[i] = 1
        elif np.real(w[i]) < -1:
            w[i] = -1
        val = np.arccos(np.real(w[i]))
        if np.imag(w[i]) > 0:
            val *= -1
        ham += (val) * ((v[:, i]).reshape(-1, 1) @ (np.linalg.inv(v)[i, :]).reshape(1, -1))

    # Check
    exph = expm(-1j * ham)
    if not np.allclose(gate, exph, atol=1e-7):
        print("WRONG FOR:\n", gate.round(3),'\n', exph.round(3))

        # import pdb
        # pdb.set_trace()

    return ham
