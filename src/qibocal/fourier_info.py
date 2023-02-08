from copy import deepcopy
import numpy as np
from scipy.linalg import expm
from sympy import *
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger


from qibo import gates, matrices
from qibo.quantum_info.basis import *
from qibo.quantum_info.superoperator_transformations import *
# from qibocal.calibrations.protocols.utils import ONEQUBIT_CLIFFORD_PARAMS
# from qibocal.calibrations.protocols.abstract import SingleCliffordsFactory

import pdb

# from custom_rb import create_superoperator

def rx(theta):
    ''' Get RX matrix'''
    return np.array([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]])

def ry(theta):
    ''' Get RY matrix'''
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])


def single_cliffords_list():
    '''Generate list of single qubit cliffords''' 
    res_list = [
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
    return res_list

paulis_list = [
    np.eye(2),
    np.array([[0,1],[1,0]]),
    np.array([[0,-1j],[1j,0]]),
    np.array([[1,0],[0,-1]])
]

noisy_paulis_list = [
    np.array([[0,1],[1,0]]),
    np.array([[0,-1j],[1j,0]])
]

noisy_pauli_group = [
    np.array([[0,1],[1,0]]),
    np.array([[0,-1j],[1j,0]])
]
pauli_group = deepcopy(paulis_list)
for s in paulis_list:
    pauli_group.append(-s)
    pauli_group.append(1j * s)
    pauli_group.append(-1j * s)
    noisy_pauli_group.append(-s)
    noisy_pauli_group.append(1j * s)
    noisy_pauli_group.append(-1j * s)


# for s in noisy_pauli_list:
#     noisy_pauli_group.append(-s)


# Dictionary of gate sets
gates_dict = {
    'xid': [matrices.I, matrices.X],
    'three': [matrices.I, rx(2 * np.pi/3), matrices.X, rx(4*np.pi/3)],
    'four': [matrices.I, rx(np.pi/2), matrices.X, rx(3*np.pi/2)],
    'paulis': pauli_group, #[matrices.I, matrices.X, matrices.Y, matrices.Z],
    'cliffords': single_cliffords_list(),
    'x': [matrices.X],
    'id': [matrices.I]
}

# Dictionary of noisy gates (I, Z are not noisy)
noisy_gates_dict = {
    'xid': [matrices.X],
    'three': [rx(2 * np.pi/3), matrices.X, rx(4*np.pi/3)],
    'four': [rx(np.pi/2), matrices.X, rx(3*np.pi/2)],
    'paulis': noisy_pauli_group, #[matrices.X, matrices.Y],
    'cliffords': [],
    'x': [matrices.X],
    'id': [matrices.I]
}

# Dictionary of indices corresponding to the irrep of each gate set and multiplicity of this irrep
irrep_dict = {
    'xid': (2, 3, 2), # YZ
    'three': (3, 3, 1),
    'four': (3, 3, 1), # Z because YZ part is reducible in complex space (even though irreducible for real)
    'paulis': (3, 3, 1), # X
    'cliffords': (1, 3, 1), # XYZ
    'x': (2, 3, 2),
    'id': (0, 3, 4)
}

# Dictionary of x_lambda s.t. P_l = X_l*X_l^\dag
# xl_dict = {
#     'xid': np.array([[0, 0], [0, 0], [1, 0], [0, 1]]),
#     'four': np.array([[0, 0], [0, 0], [1, 0], [0, 1]]), # np.array([[0], [0], [1], [0]]), # 
#     'paulis': np.array([[0], [0], [0], [1]]),
#     'cliffords': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
#     'x': np.array([[0, 0], [0, 0], [1, 0], [0, 1]]),
#     'id': np.eye(4)
# }

def get_xl(gates_key='xid'):
    irrep_inds = irrep_dict[gates_key]
    dim = irrep_inds[1] - irrep_inds[0] + 1
    xl = np.zeros((4, dim))
    cnt = 0
    for i in range(irrep_inds[0], irrep_inds[1]+1):
        xl[i][cnt] = 1
        cnt += 1
    
    return xl


def get_basis(gates_key='xid'):
    if gates_key == "four":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1j/np.sqrt(2), -1j/np.sqrt(2)], [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]]) 
    elif gates_key == 'three':
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/np.sqrt(2), -1j/np.sqrt(2)], [0, 0, 1j/np.sqrt(2), -1/np.sqrt(2)]])
    return np.eye(4)

# Dictionary of all POVM elements E_i
povm_dict = {
    'z0': np.array([[1, 0], [0, 0]]),
    'z1': np.array([[0, 0], [0, 1]]),
    'x0': np.array([[0.5, 0.5], [0.5, 0.5]]),
    'x1': np.array([[0.5, -0.5], [-0.5, 0.5]]),
    'y0': np.array([[0.5, -1j/2], [1j/2, 0.5]]),
    'y1': np.array([[0.5, 1j/2], [-1j/2, 0.5]])
}


# Check if numpy array is in list of numpy arrays
def arreqclose_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr)), False)


# Create a list of 1-qubit paulis
s0 = np.eye(2)
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.array([[1,0],[0,-1]])
paulis = np.array([s0, s1, s2, s3]) / np.sqrt(2)
pauli_notnorm = np.array([s0, s1, s2, s3]) # / np.sqrt(2)

PAULI_DIM = 1
PAULIS_NQ = deepcopy(paulis)

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


def get_paulis(n, norm=True):
    ''' Get array of all n-qubit Paulis '''
    global PAULI_DIM
    global PAULIS_NQ
    
    if n == PAULI_DIM and norm:
        return PAULIS_NQ
     
    res = []
    for i in range(4 ** n):
        inds = base_convert(i, 4, length=n) 
        el = paulis[inds[0]]if norm else pauli_notnorm[inds[0]]
        for j in range(1, n):
            el = np.kron(el, paulis[inds[j]]) if norm else np.kron(el, pauli_notnorm[inds[j]])
        res.append(el)
    
    PAULI_DIM = n
    PAULIS_NQ = res
    return res


def inner_prod(a, b):
    # Calculate (a, b) = sum(a*ij bij)
    res = np.dot(a.conj().reshape(1, -1), b.reshape(-1, 1))
    return complex(res)


def partial_trace(a, n=2):
    dim = a.shape[0] // n
    res = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for k in range(n):
                res[i][j] += a[n*i+k][n*j+k]
    return res


def density_to_pauli(density, norm=True):
    ''' Returns Pauli vector of a density'''
    qubits_num = int(np.log2(density.shape[0]))
    nq_paulis = get_paulis(qubits_num, norm=norm) # FIXME norm
    
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
            res[i][j] = np.trace(Pi @ (u @ Pj @ u.T.conj()) ) # / dim
    return res


def sympy_to_pauli_op(u=Identity(2)):
    ''' Transfrom a unitary operator to Pauli-Liouville superoperator'''
    dim = 2
    nq_paulis = deepcopy(paulis)
    
    pauli_dim = len(nq_paulis)
    res = zeros(pauli_dim, pauli_dim)
    for i in range(pauli_dim):
        Pi = Matrix(nq_paulis[i])
        for j in range(pauli_dim):
            Pj = Matrix(nq_paulis[j])
            m_el = Pi * (u * Pj * Dagger(u))
            trace_el = 0
            for row in range(m_el.shape[0]):
                trace_el = trace_el + m_el[row,row]
            res[i, j] = trace_el # / dim
    return res


def superop_to_pauli(a):
    qubits_num = int(np.log2(np.sqrt(a.shape[0])))
    u_p = comp_basis_to_pauli(qubits_num, normalize=True)
    return u_p @ a @ u_p.T.conj()


def get_povm_list(inds=['z']):
    ''' Get list of POVM elements depending on the bases choice. '''
    povm_list = []
    for i in inds:
        for j in ['0', '1']:
            povm_list.append(povm_dict[i+j] / len(inds))
    povm_list = np.array(povm_list)
    return povm_list


def get_m(inds=['z']):
    ''' Get matrix M = sum |E_i)(E_i|'''
    povm_list = get_povm_list(inds)

    res = vectorization(povm_list[0], order="system") .reshape(-1, 1) @ vectorization(povm_list[0], order="system") .reshape(1, -1).conj()
    for i in range(1, len(povm_list)):
        res += vectorization(povm_list[i], order="system") .reshape(-1, 1) @ vectorization(povm_list[i], order="system") .reshape(1, -1).conj()
    return res

# TODO remove cnt and print(S)
# cnt = 0
def get_s(povm_key='z', gates_key='xid', tol=7):
    m_matr = superop_to_pauli(get_m(povm_key))
    gates_list = gates_dict[gates_key]
    
    omega_g = to_pauli_op(gates_list[0]) # np.kron(gates_list[0], gates_list[0])
    res = (omega_g.T.conj() @ m_matr @ omega_g) 
    
    for i in range(1, len(gates_list)):
        omega_g = to_pauli_op(gates_list[i]) # np.kron(gates_list[i], gates_list[i])
        res += (omega_g.T.conj() @ m_matr @ omega_g) 
    res /= len(gates_list)
    
    res = np.linalg.pinv(res.round(tol))
    # global cnt
    # if cnt == 0:
    #     cnt += 1
    #     print("S =\n",res.round(3))
    return res


def get_sl_pinv(povm_key='z', gates_key='xid', tol=7):

    m_matr = superop_to_pauli(get_m(povm_key))
    gates_list = gates_dict[gates_key]
    
    omega_g = to_pauli_op(gates_list[0])
    res = (omega_g.T.conj() @ m_matr @ omega_g) 
    
    for i in range(1, len(gates_list)):
        omega_g = to_pauli_op(gates_list[i])
        res += (omega_g.T.conj() @ m_matr @ omega_g) 
    res /= len(gates_list)

    gates_basis = get_basis(gates_key)
    res = gates_basis.T.conj() @ np.linalg.pinv(res.round(tol)) @ gates_basis
    # res = get_s(povm_key, gates_key, tol)

    irrep_inds = irrep_dict[gates_key]
    res = res[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1]
    return res


def get_mql(povm_key='z', gates_key='xid', init_density=np.array([[1, 0], [0, 0]])):

    gates_basis = get_basis(gates_key)
    m_matr = gates_basis.T.conj() @ superop_to_pauli(get_m(povm_key)) @ gates_basis # np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=complex) # for 'z' measurement
    x_l = get_xl(gates_key)
    m_l = x_l.T.conj() @ m_matr

    s_l = get_sl_pinv(povm_key, gates_key)
    state_pauli = gates_basis.T.conj() @ density_to_pauli(init_density)
    state_outer_prod = state_pauli.reshape(-1, 1) @ state_pauli.reshape(1, -1).conj()
    q_l = (state_outer_prod @ x_l @ s_l).T.conj()

    mq_l = m_l.reshape(-1, 1) @ q_l.reshape(1, -1).conj()
    # pdb.set_trace()
    if irrep_dict[gates_key][-1] > 1:
        return partial_trace(mq_l, irrep_dict[gates_key][-1])
    else:
        return mq_l


def validation(fourier, povm_key='z', gates_key='xid', init_density=np.array([[1, 0], [0, 0]])):
    # A matrix for Z-basis measurement and initial state |0><0|
    mql = get_mql(povm_key, gates_key, init_density)# np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    # Get the eigenvectors and eigenvalues of the Fourier matrix s.t. r @ np.diag(eigvals) @ l == fourier
    eigvals, r = np.linalg.eig(fourier)
    l = np.linalg.inv(r)
    # Find the indices corresponing to the obtained eigenvalues in descending order
    sorted_indices = np.abs(eigvals).argsort()[::-1]
    # Get the number of dominant decays
    n_l = irrep_dict[gates_key][-1]
    # A: Dominant part
    r1 = r[:, sorted_indices[:n_l]]
    l1 = l[sorted_indices[:n_l], :]
    a = l1 @ mql @ r1
    # B: Subdominant part
    r2 = r[:, sorted_indices[n_l:]]                                                                                                                                                        
    l2 = l[sorted_indices[n_l:], :]
    b = l2 @ mql @ r2
    # What should be returned?
    coeffs = np.array([a[i][i] for i in range(len(a))] + [b[i][i] for i in range(len(b))])
    # coeffs = np.array([a[0][0], a[1][1], b[0][0], b[1][1]])
    decays = eigvals[sorted_indices]

    # import pdb
    # pdb.set_trace()

    # Sort by dominant components for m=10 (a*p^10)
    # sorted_indices = (coeffs * (decays ** 2)).argsort()[::-1]
    # coeffs = coeffs[sorted_indices]
    # decays = decays[sorted_indices]

    dom_coeffs = []
    dom_decays = []
    # pdb.set_trace()
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
                # print(len(dom_coeffs), "\n", coeffs[i].round(3), "*", decays[i].round(3), "^m\n")
    return dom_coeffs, dom_decays


def get_fourier(gates_key='xid'):
    gates_list = gates_dict[gates_key]
    gates_basis = get_basis(gates_key)

    # Get omega(g) and tau(g)
    pauli_repr = gates_basis.T.conj() @ to_pauli_op(gates_list[0]) @ gates_basis
    irrep_inds = irrep_dict[gates_key]
    irrep = pauli_repr[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1]
    if irrep_inds[-1] > 1:
        res = np.conj(irrep[0][0]) * pauli_repr / len(gates_list)
    else:
        res = np.kron(irrep.conj(), pauli_repr) / len(gates_list)

    for i in range(1, len(gates_list)):
        pauli_repr = gates_basis.T.conj() @ to_pauli_op(gates_list[i]) @ gates_basis
        irrep = pauli_repr[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1]
        if irrep_inds[-1] > 1:
            res += np.conj(irrep[0][0]) * pauli_repr / len(gates_list)
        else:
            res += np.kron(irrep.conj(), pauli_repr) / len(gates_list)

    return res


def round_sympy(matr, n=3):
    matr_round = matr
    for i in range(matr.shape[0]):
        for j in range(matr.shape[1]):
            ex2 = matr[i, j]
            for _ in range(3):
                ex1 = ex2.evalf(n=n)
                ex2 = ex1
                for a in preorder_traversal(ex1):
                    if isinstance(a, Float):
                        ex2 = ex2.subs(a, round(a, n))
            matr_round[i, j] = ex2.evalf(n=n)
    return matr_round


def get_noisy_fourier(noise=np.eye(4), gates_key='xid', noisy_gates=[]):
    # Get the group of gates
    gates_list = gates_dict[gates_key]

    # Transform from Pauli-Liouville to irrep's basis
    gates_basis = get_basis(gates_key)
    noise = gates_basis.T.conj() @ noise @ gates_basis

    # Get omega(g) and tau(g)
    irrep_inds = irrep_dict[gates_key]
    pauli_repr = gates_basis.T.conj() @ to_pauli_op(gates_list[0]) @ gates_basis
    irrep = pauli_repr[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1]

    # Get phi(g) and tau(g)^dagger tensor phi(g)
    if len(noisy_gates) == 0 or arreqclose_in_list(gates_list[0], noisy_gates):
        # print(0)
        pauli_repr = noise @ pauli_repr
    if irrep_inds[-1] > 1:
        res = np.conj(irrep[0,0]) * pauli_repr
    else:
        res = np.kron(irrep.conj(), pauli_repr)

    # Repeat for all the gates 
    for i in range(1, len(gates_list)):
        pauli_repr = gates_basis.T.conj() @ to_pauli_op(gates_list[i]) @ gates_basis
        irrep = pauli_repr[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1]
        if len(noisy_gates) == 0 or arreqclose_in_list(gates_list[i], noisy_gates):
            # print(i)
            pauli_repr = noise @ pauli_repr
        if (irrep_inds[1]-irrep_inds[0]+1) == irrep_inds[2]:
            res += np.conj(irrep[0,0]) * pauli_repr
        else:
            try:
                res += np.kron(irrep.conj(), pauli_repr)
            except:
                print("ERROR in line 394")
                print(res.shape, res.dtype)
                print(irrep.shape, irrep.dtype)
                print(pauli_repr.shape, pauli_repr.dtype)
    import pdb
    pdb.set_trace()
    return res / len(gates_list)


def get_noisy_fourier_sym(noise, gates_key='xid', noisy_gates=[]):
    # Get the group of gates
    gates_list = gates_dict[gates_key]
    
    # Pauli Liouville to irrep's basis 
    gates_basis = get_basis(gates_key)
    noise = Matrix(gates_basis.T.conj()) * noise * Matrix(gates_basis)

    # Get omega(g) and tau(g)
    irrep_inds = irrep_dict[gates_key]
    pauli_repr = Matrix(gates_basis.T.conj() @ to_pauli_op(gates_list[0]) @ gates_basis)
    irrep = Matrix(pauli_repr[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1])

    # Get phi(g) and tau(g)^dagger tensor phi(g)
    if len(noisy_gates) == 0 or arreqclose_in_list(gates_list[0], noisy_gates):
        # print(0)
        pauli_repr = noise * Matrix(pauli_repr)
    if irrep_inds[-1] > 1:
        res = (irrep[0,0]) * pauli_repr
    else:
        res = TensorProduct((irrep), pauli_repr)
    # Repeat for all the gates 
    for i in range(1, len(gates_list)):
        pauli_repr = Matrix(gates_basis.T.conj() @ to_pauli_op(gates_list[i]) @ gates_basis)
        irrep = Matrix(pauli_repr[irrep_inds[0]:irrep_inds[1]+1, irrep_inds[0]:irrep_inds[1]+1])
        if len(noisy_gates) == 0 or arreqclose_in_list(gates_list[i], noisy_gates):
            # print(i)
            pauli_repr = noise * Matrix(pauli_repr)
        if irrep_inds[-1] > 1:
            res += (irrep[0,0]) * pauli_repr
        else:
            res += TensorProduct((irrep), pauli_repr) 
    res = round_sympy(res)
    return res / len(gates_list)


def get_pauli_noise(px=None, py=None, pz=None):
    if px is None or py is None or pz is None:
        px_value = Symbol('px') if px is None else px
        py_value = Symbol('py') if py is None else py
        pz_value = Symbol('pz') if pz is None else pz
        ax = 1 - 2 * py_value - 2 * pz_value
        ay = 1 - 2 * px_value - 2 * pz_value
        az = 1 - 2 * px_value - 2 * py_value
        noise = Matrix([[1, 0, 0, 0], [0, ax, 0, 0], [0, 0, ay, 0], [0, 0, 0, az]])
    else:
        ax = 1 - 2 * py - 2 * pz
        ay = 1 - 2 * px - 2 * pz
        az = 1 - 2 * px - 2 * py
        noise = np.array([[1, 0, 0, 0], [0, ax, 0, 0], [0, 0, ay, 0], [0, 0, 0, az]], dtype=complex)
    # print(noise)
    return noise


def get_unitary_noise():
    # if len(unitary) == 0:
    #     U = MatrixSymbol('U', 2, 2)
    #     unitary_matr = Matrix(U)
    # else:
    #     unitary_matr = deepcopy(unitary)
    
    # u_pauli = sympy_to_pauli_op(unitary_matr) # too much

    a = Symbol('a')
    b = Symbol('b')
    phi = Symbol('varphi', real=True)
    u_matr = Matrix([[a, b], [-exp(I*phi)*conjugate(b), exp(I*phi)*conjugate(a)]])
    u_pauli = sympy_to_pauli_op(u_matr) 
    return u_pauli


def get_rand_unitary_noise(n=2):
    def get_generator(n, is_complex=True):
        if n % 2:
            return np.eye(2) / np.sqrt(2)
        # Get dimension
        # Generate a random Gaussian Hermitian matrix
        ob = np.random.normal(0, 1, size=(n, n))
        # Add complex values
        if is_complex:
            b = np.random.normal(0, 1, size=(n, n))
            ob = ob + b*1j
            ob = (ob + (ob.conj().T)) / 2
        return ob

    def random_unitary(n, lam = 0.1, is_complex=True):
        g = get_generator(n)
        return expm(1j * g * lam)

    u = random_unitary(n)
    return to_pauli_op(u)


def get_thermal_relaxation_noise(t1=None, t2=None, t=None, a0=None):
    if t1 is None or t2 is None or t is None:
        t1_value = Symbol('T1') if t1 is None else t1
        t2_value = Symbol('T2') if t2 is None else t2
        t_value = Symbol('t') if t is None else t
        a0_value = Symbol('a0') if a0 is None else a0

        noise = Matrix([[1, 0, 0, 0],
                        [0, exp(-1*t_value/t2_value), 0, 0],
                        [0, 0, exp(-1*t_value/t2_value), 0],
                        [(1-2*a0_value)*(exp(-1*t_value/t1_value)-1), 0, 0, exp(-1*t_value/t1_value)]])
    else:
        a0 = 0 if a0 is None else a0
        noise = np.array([[1, 0, 0, 0],
                        [0, np.exp(-t/t2), 0, 0],
                        [0, 0, np.exp(-t/t2), 0],
                        [(1-2*a0)*(np.exp(-t/t1)-1), 0, 0, np.exp(-t/t1)]], dtype=complex)
    
    return noise


def get_noise(noise_label='pauli', noise_params=[]):
    if noise_label == 'unitary':
        if len(noise_params) > 0:
            return to_pauli_op(noise_params)
        else:
            return get_rand_unitary_noise()

    if noise_label == 'pauli':
        if len(noise_params) > 0:
            return get_pauli_noise(*noise_params)
        else:
            return get_pauli_noise()
    
    elif noise_label == 'tr':
        if len(noise_params) > 0:
            return get_thermal_relaxation_noise(*noise_params)
        else:
            return get_thermal_relaxation_noise()
    
    if isinstance(noise_params, np.ndarray):
        return to_pauli_op(noise_params)
    else:
        return np.eye(4)


def round_sympy_dict(dict={}):
    round_decays = {}
    for decay in dict.items():
        r_decay = decay[0]
        for a in preorder_traversal(decay[0]):
            if isinstance(a, Float):
                r_decay = r_decay.subs(a, round(a, 3))
        if r_decay == 0:
            continue
        elif r_decay in (round_decays.keys()):
            round_decays[r_decay] = decay[1] # FIXME if += then KeyError
        elif conjugate(r_decay) in list(round_decays.keys()):
            round_decays[conjugate(r_decay)] = decay[1]
        else:
            round_decays[r_decay] = decay[1]
    return round_decays


def get_decays(gates_key='xid', noise='pauli', noise_params=[]):
    noise = get_noise(noise, noise_params)
    # print(noise_params)
    noisy_gates = noisy_gates_dict[gates_key]
    # print(noisy_gates)
    if isinstance(noise, np.ndarray):
        fourier = get_noisy_fourier(noise, gates_key, noisy_gates)  
        eigvals = np.linalg.eigvals(fourier)
        eigvals[::-1].sort()
        # print(eigvals)
        res = {}
        for e in eigvals:
            if e == 0:
                continue
            elif np.imag(e) == 0:
                key = str(np.real(e))
            else:
                key = str(np.real(e)) + "±" + str(np.abs(np.imag(e))) + "i"
            if key in res.keys():
                res[key] += 1
            elif e != 0:
                res[key] = 1
    else:
        fourier = get_noisy_fourier_sym(noise, gates_key, noisy_gates)
        # print(fourier)
        eigs = fourier.eigenvals()
        res = round_sympy_dict(eigs)
        res = round_sympy_dict(res)
    return res


# gates_key = 'paulis'
# noisy_gates = noisy_gates_dict[gates_key] 
# = [
#     s1 # np.array([[0, -1], [1, 0]])
# ]

# Fourier for symbolic Pauli noise
# noise = get_pauli_noise()

# Thermal Relaxation noise for X gate
# noise = get_thermal_relaxation_noise()

# Random unitary nosie
# noise = get_unitary_noise()

# fourier_matr = get_noisy_fourier(noise, gates_key, noisy_gates)
# print(fourier_matr, '\n')
# P, D = fourier_matr.diagonalize()
# print(D)

# print(get_decays('four', 'tr', [0.1, 0.05, 0.01, 0]))

# print(get_decays('paulis'))


def check_validation(gates_key='xid', noise='pauli', noise_params=[0.01, 0.02, 0.03]):
    noise = get_noise(noise, noise_params)
    noisy_gates = noisy_gates_dict[gates_key]
    # print(noisy_gates)
    fourier = get_noisy_fourier(noise, gates_key, noisy_gates)
    # print(fourier)
    return validation(fourier, 'z', gates_key)


# print(get_decays('id', 'tr'))
# print(get_nonmark_decays('xid', 12, 0, 0))
# (check_validation('cliffords'))
# print(to_pauli_op(np.eye(2)))
# noise = 
# print(f.shape)
# print(f.round(3))
# print(validation(f, 'z', 'cliffords'))
# print(get_noisy_fourier(get_noise('pauli'), 'cliffords'))