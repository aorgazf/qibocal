import numpy as np
from scipy.linalg import expm
import random

from qibo import models
from qibo import gates
from qibo.noise import NoiseModel, PauliError, ThermalRelaxationError
from qibo import quantum_info

from qibo import hamiltonians
from qibo.symbols import X, Y, Z

from qibocal.calibrations.protocols.xfilterrb import perform, perform_nonm, ham_to_unitary_matr, perform_kraus
from qibo.quantum_info.basis import vectorization, comp_basis_to_pauli

from myfilterrb import get_ab_f

import qibo
qibo.set_backend("numpy") # for KrausChannel to work (not invertible)

def unvectorization(state):
    dim = int(np.sqrt(state.shape[0]))
    matr = np.zeros((dim, dim), dtype=state.dtype)
    block_cnt = 0
    for row in range(dim):
        for col in range(dim):
            ind = (row // 2) * 4 + (row % 2) + (col // 2) * 8 + (col % 2) * 2
            matr[row][col] = state[ind]

    return matr

# Define CustomError that stores a Channel for the noise model
class CustomError:
    def __init__(self, options, channel):
        self.options = options
        self._channel = channel
    
    def channel(self, qubits, options):
        return self._channel


# Generate random unitary matrix close to Id. U=exp(i*l*G)
def rand_unitary(lam=0.1):
        herm_generator = np.random.normal(size=(2, 2)) + 1j * np.random.normal(size=(2,2))
        herm_generator = (herm_generator + herm_generator.T.conj()) / 2

        from scipy.linalg import expm
        return expm(1j * lam * herm_generator)


# From computational to Pauli-Liouville basis
# Create a list of 1-qubit paulis
s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
paulis = [s0, s1, s2, s3]
def to_pauli_op(u=np.eye(2)):
    res = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            res[i][j] = np.trace(paulis[i] @ (u @ paulis[j] @ u.T.conj())) / 2
    return res


# Calculate fourier transform of the noisy implementation
def fourier_sign(noise=np.eye(4)):
    # Reference representation of Id in Pauli basis
    id_ideal = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    
    # Reference representation of X in Pauli basis
    x_ideal = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, -1]])
    
    # Noisy representation of X in Pauli basis
    x_noisy = x_ideal @ noise
    # res = (id_ideal - x_noisy) / 2
    res = - x_noisy # for {X}
    return res


# Calculate expected decays for unitary noise
def unitary_decays(u):
    pauli_unitary = to_pauli_op(u)
    # print("\nPauli un=\n\n", pauli_unitary.round(3))
    fourier_matr = fourier_sign(pauli_unitary.round(3))
    # print("\nFourier=\n\n", fourier_matr) 
    eigvals, v = np.linalg.eig(fourier_matr.round(3))
    # print("\nEigs=\n\n", eigvals.round(3)) 
    # print("\nV=\n\n", v.round(3))
    # print("\nV-1=\n\n", np.linalg.inv(v).round(3))
    res = []
    for e in eigvals:
        if e > 1e-5:
            res.append(e)
    return np.array(res)


def pauli_noise_model(px=0, py=0, pz=0, init_state=None):

    # Define the noise model
    paulinoise = PauliError(px, py, pz)
    noise = NoiseModel()
    noise.add(paulinoise, gates.X)

    # Calculate the expexted decay 1-px-pz
    # exp_decay = 1 - px - py
    exp_decay = 1 - 2*px - 2*py # For {X}

    # Run the protocol
    perform(nqubits, depths, runs, nshots, noise=noise, 
    title=f"Pauli Noise Error: {px}, {py}, {pz}. Expected decay: {exp_decay}", ndecays=1, init_state=init_state) 


def tr_noise_model(t1, t2, t, a0=0, init_state=None):
    # Define the noise model
    thermal_relaxation = ThermalRelaxationError(t1, t2, t, a0)
    noise = NoiseModel()
    noise.add(thermal_relaxation, gates.X)

    # Print the expexted decay [1+exp(-t/t1)]/2
    # exp_decay = (1 + np.exp(-t/t1)) / 2
    exp_decay = np.array([np.exp(-t/t1), np.exp(-t/(t2))]) # For {X}

    # Run the protocol
    perform(nqubits, depths, runs, nshots, noise=noise, 
    title=f"Thermal Relaxation Error: {t1}, {t2}, {t}, {a0}. Expected decay: {exp_decay.round(3)}", ndecays=2, init_state=init_state) 


def unitary_noise_model(matr=None):
    # Generate a random unitary close to Id, U=exp(-i*H*lam)
    unitary_matr = matr if matr is not None else rand_unitary(lam=0.3)
    # Create a unitary channel
    unitary_channel = gates.UnitaryChannel([1], [((0,), unitary_matr)])
    # Define the noise model
    unitary_error = CustomError([0], unitary_channel) # temporary solution
    noise = NoiseModel()
    noise.add(unitary_error, gates.X)

    print("Unitary noise:\n", unitary_matr)
    exp_decays = unitary_decays(unitary_matr)
    print(exp_decays)

    # Run the protocol
    perform(nqubits, depths, runs, nshots, noise=noise, 
    title=f"Unitary noise: {unitary_matr.round(2)}.<br>Expected decays: {exp_decays.round(3)}", ndecays=2) 


def kraus_noise_model(time_step=1):
    zeros = np.array([[1, 0], [0, 0]])
    ones = np.array([[0, 0], [0, 1]])
    sigma_minus = np.array([[0, 1], [0, 0]])

    def jc_kraus_channel(time=1, omega=np.pi):
        # Define Kraus operators
        K0 = zeros + np.cos(omega * time / 2) * ones
        K1 = -1j * np.sin(omega * time / 2) * sigma_minus

    # Define a Kraus channel
    kraus = gates.KrausChannel([((0,), K0), ((0,), K1)])

# Perform parameters
nqubits = 1
depths = range(1, 30, 1)
runs = 1000
nshots = 2048

# Pauli noise: px, py, pz
# pauli_noise_model(0.1, 0.2, 0.3)
# 0.03, 0.1, init_state=np.array([[3/4, 3/8-np.sqrt(3)/8*1j], [3/8+np.sqrt(3)/8*1j, 1/4]]))

# Thermal relaxation error: t1, t2, t, a0=0
# tr_noise_model(0.1, 0.05, 0.01, 0, init_state=np.array([[1, 0], [0, 0]]))
# ([[3/4, 3/8-np.sqrt(3)/8*1j], [3/8+np.sqrt(3)/8*1j, 1/4]]))

# Random unitary noise
# unitary_noise_model()

# Non-Markovian dynamics

def get_eigs_u(dt=0.1, ok=1, jk=1, g1=0, g2=0):
    ui = ham_to_unitary_matr(0, dt, ok, jk)
    ui = gates.UnitaryChannel([1], [((0,1), ui)])
    pauli_i = ui.to_pauli_liouville(normalize=True)

    ux = ham_to_unitary_matr(1, dt, ok, jk, g1, g2)
    ux = gates.UnitaryChannel([1], [((0,1), ux)])
    pauli_x = ux.to_pauli_liouville(normalize=True)
    # fourier_phi = -pauli_x # for {X}
    fourier_phi = 0.5 * (pauli_i - pauli_x)
    w, v = np.linalg.eig(fourier_phi)
    return w

dt = 0.1
ok = 1 # omega1 = omega0 * ok
jk = 1 # J = 1 / (dt * jk)
g1 = 0
g2 = 0

# Define the noise model
t1, t2, t, a0 = (0.1, 0.05, 0.01, 0)
thermal_relaxation = ThermalRelaxationError(t1, t2, t, a0)
# noise = NoiseModel()
# noise.add(thermal_relaxation, gates.Unitary)

# Print the expexted decay [1+exp(-t/t1)]/2
# exp_decay = (1 + np.exp(-t/t1)) / 2
# exp_decay = np.array([np.exp(-t/t1), np.exp(-t/(t2))]) # For {X}

# perform_nonm(nqubits, depths, runs, nshots, ndecays=4, dt=dt, ok=ok, jk=jk, g1=g1, g2=g2) # , noise=noise)
# print(get_eigs_u(dt, ok, jk, g1, g2).round(3))


def nnnonmark_gate(xi_inds=[0], dt=0.1, ok=1, jk=2):
        omega0 = np.pi / dt
        omega1 = omega0 * ok
        J = 1 / (dt * jk)

        gates_list = []

        for k in xi_inds:
            print(k)

            unitary_matrix = ham_to_unitary_matr(k, dt, ok, jk)
            print(unitary_matrix)
            gates_list.append(gates.Unitary(unitary_matrix, 0, 1))
        
        c = models.Circuit(2, density_matrix=True)
        c.add(gates_list)
        c.add(gates.M(0))
        # import pdb
        # pdb.set_trace() 
        print(c.draw())
        ex = c()
        print(ex.probabilities())
        return (gates_list)


def get_eigs_x(ham, dt=0.1):
    eigs_ham = np.linalg.eigvals(ham)
    res_eigs = []
    for i in range(len(eigs_ham)):
        for j in range(len(eigs_ham)):
            if i != j:
                lij = eigs_ham[i] - eigs_ham[j]
                res_eigs.append(np.exp(-1j * lij * dt))
    res_eigs = np.array(res_eigs)
    print(res_eigs.round(3))

# for j in range(1, 9):
#     jk = j / 4
#     print(f"\nJ=1/(dt*{jk})\n")
#     eigvals = get_eigs_u(dt, ok, jk)
#     print(set(eigvals.round(3)))

### New non-Markovian model

basis = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

def create_superoperator(is_x=False, J=10, g0=0, g1=0, is_z=True):
    dt = 0.1
    # Variables
    o0 = np.pi / dt if is_x else 0
    o1 = np.pi / dt

    # Hamiltonian
    hamiltonian = np.array([[o1 / 2, 0, o0 / 2, J],
                        [0, -o1 / 2, J, o0 / 2],
                        [o0 / 2, J, o1 / 2, 0],
                        [J, o0 / 2, 0, -o1 / 2]])

    # GKSL equation
    def get_rho_dot(rho):
        # Computer the commutator
        com = hamiltonian @ rho - rho @ hamiltonian
        
        # z0rz0
        z0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
        zrho0 = z0 @ rho @ z0.T
        # x0rx0
        x0 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        xrho0 = x0 @ rho @ x0.T
        # z1rz1
        z1 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        zrho1 = z1 @ rho @ z1.T
        
        # Compute the result
        res = -1j * com
        res += g0 * (zrho0 - rho) if is_z else g0 * (xrho0 - rho)
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
    exp_l = expm(dt * l_matr)

    return exp_l


def superoperator_to_choi(sup):
    choi = np.zeros(sup.shape, dtype=complex)
    for i in range(4):
        i_vec = basis[i].reshape(-1, 1) 
        for j in range(4):
            j_vec = basis[j].reshape(1, -1) 
            ij_matr = i_vec @ j_vec
            sup_state = (sup @ ij_matr.reshape(-1, 1)).reshape(4, 4)
            choi += np.kron(sup_state, ij_matr)
    return choi


def choi_to_kraus(choi):
    # Get Kraus operators M_i = sqrt(w_i) * unvec(v_i) 
    kraus_list = []
    eigvals, eigvecs = np.linalg.eig(choi)
    for i in range(len(eigvals)):
        w = eigvals[i]
        v = (eigvecs[:, i]).reshape(4, 4)
        if abs(w) > 1e-9:
            kraus_list.append(((0, 1), np.sqrt(w) * v))
            # print(i, kraus_list[-1][1].round(3))
    return kraus_list


is_z = True
dt = 0.1
j = 1 / (dt)
g0 = 1 / (dt * 4)
g1 = 1 / (dt * 8)

sup_i = create_superoperator(J=j, g0=g0, g1=g1, is_z=is_z)
# print("\nSUP_I =\n", sup_i.round(3))

choi_i = superoperator_to_choi(sup_i)
kraus_i_list = choi_to_kraus(choi_i)
kraus_i = gates.KrausChannel(choi_to_kraus(choi_i))

sup_x = create_superoperator(is_x=True, J=j, g0=g0, g1=g1, is_z=is_z)
# print("\nSUP_X =\n", sup_x.round(3))

choi_x = superoperator_to_choi(sup_x)
kraus_x = gates.KrausChannel(choi_to_kraus(choi_x))

# state = np.array([[3/4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1/4]])
# # state_vec = vectorization(state)
# state_vec = state.reshape(-1, 1)
# state_vec = sup_i @ state_vec
# # print(unvectorization(state_vec).round(3))
# print(state_vec.reshape(4, 4).round(3))

# for i in range(10):
#     is_x = random.choice([True, False])
#     state_vec = sup_x @ state_vec if is_x else sup_i @ state_vec 
#     print(state_vec.reshape(4, 4).round(3))
#     # print("x" if is_x else "i", "\n", unvectorization(state_vec).round(3))

# Fourier eigenvalues
u = comp_basis_to_pauli(2, normalize=True)
# pauli_i = np.matmul(u, np.matmul(sup_i, np.transpose(np.conj(u)))) 
pauli_i = kraus_i.to_pauli_liouville(normalize=True)
# pauli_x = np.matmul(u, np.matmul(sup_x, np.transpose(np.conj(u)))) 
pauli_x = kraus_x.to_pauli_liouville(normalize=True)
fourier = 1 / 2 * (pauli_i - pauli_x)
w, v = np.linalg.eig(fourier)
print("Eigenvalues of Fourier =\n", w.round(3), "\n")

get_ab_f(fourier)

decays = []
for l in w.round(3):
    if abs(l) > 1e-5 and l not in decays and np.conj(l) not in decays:
        decays.append(l)

title = f"$dt={dt}, J={j}, \gamma_0={g0}, \gamma_1={g1}, \sigma^{'{'}(0){'}'}_{'z' if is_z else 'x'};  z_i = {decays}$"
# perform_kraus(nqubits, depths, runs, nshots, ndecays=4, kraus_i=kraus_i, kraus_x=kraus_x, title=title)



def check_sups(state, is_x):
    sup = sup_x if is_x else sup_i
    # print(sup)
    pauli_sup = kraus_x.to_superop() if is_x else kraus_i.to_superop() 
    # print(pauli_sup)
    pauli_sup_state = (pauli_sup @ (state).flatten(order='F')).reshape(4, 4).T
    sup_state = (sup @ (state).reshape(-1, 1)).reshape(4, 4)
    is_eq = np.allclose(sup_state, pauli_sup_state, atol=1e-5) and np.allclose(sup, pauli_sup, atol=1e-5)
    if not is_eq:
        print("ISEQ = ", is_eq, '\n')
        print("SUP\n")
        print(sup_state.round(3))
        print("KRAUS\n")
        print(pauli_sup_state.round(3))

    return sup_state


# Check numerically if Kraus operators are correct 
def check(state, is_x=False):
    sup = create_superoperator(is_x)
    ch = superoperator_to_choi(sup)
    
    # state = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    sup_state = unvectorization(sup @ vectorization(state))

    kraus_list = choi_to_kraus(ch)
    from copy import deepcopy
    kraus_state = np.zeros(state.shape, dtype=complex)
    for kraus in kraus_list:
        k = kraus[-1]
        kraus_state += k @ state @ k.T.conj()

    is_eq = np.allclose(sup_state, kraus_state, atol=1e-5)
    if not is_eq:
        print("ISEQ = ", is_eq, '\n')
        print("SUP\n")
        print(sup_state.round(3))
        print("KRAUS\n")
        print(kraus_state.round(3))

    return kraus_state

# st2 = check_sups(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), False)
# for i in range(5):
#     is_x = random.choice([True, False])
#     print("IS_X=", is_x)
#     check_sups(st2, is_x)

# from myfilterrb import get_ab_f
# get_ab_f(fourier)


# print("\nP1=\n", pauli_supi.round(3))
# print("\nP2=\n", pauli_i.round(3))
# print(np.allclose(pauli_i, pauli_supi))
# for i in range(16):
#     for j in range(16):
#         if abs(pauli_supi[i][j] - pauli_i[i][j]) > 1e-7:
#             print(i, j, end=', ')
#             print(np.round(pauli_supi[i][j], 3), np.round(pauli_i[i][j], 3))

# state = np.array([[str(i) + str(j) for j in range(4)] for i in range(4)])
# print(state)
# v_state = vectorization(state)
# print("\nVSTATE =\n", v_state)
# v_state = unvectorization(v_state)
# print(v_state)