import numpy as np
from scipy.linalg import expm

from qibo import models
from qibo import gates
from qibo.noise import NoiseModel, PauliError, ThermalRelaxationError, UnitaryError

from qibocal.calibrations.protocols.xfilterrb import perform, perform_nonm, ham_to_unitary_matr, perform_kraus
from qibocal.calibrations.protocols.filterrb import perform, gates_keys, gates_dict, noisy_gates_dict

from fourier_info import get_decays, check_validation
from nonmark_validation import nonm_validation


# TMP: Store experimental decay parameters for different j, g0, g1
from qibocal.fitting.rb_methods import fit_expn_func
from qibocal.calibrations.protocols.filterrb import single_qubit_filter, NonMarkovianFactory, FilterRBExperiment, FilterRBResult, irrep_dict

import qibo
qibo.set_backend("numpy") # for KrausChannel to work (not invertible)

# New stuff
from qibo.quantum_info.superoperator_transformations import *

from qibo.backends import GlobalBackend
backend = GlobalBackend()

GATE_TIME = 40 * 1e-9

# Generate random unitary matrix close to Id. U=exp(i*l*G)
def rand_unitary(lam=0.1):
    herm_generator = np.random.normal(size=(2, 2)) + 1j * np.random.normal(size=(2,2))
    herm_generator = (herm_generator + herm_generator.T.conj()) / 2

    from scipy.linalg import expm
    return expm(1j * lam * herm_generator)


def pauli_noise_model(px=0, py=0, pz=0, factory_id='xid', init_state=None):

    # Define the noise model
    paulinoise = PauliError(px, py, pz)
    noise = NoiseModel()
    if factory_id in noisy_gates_dict.keys():
        for g in noisy_gates_dict[factory_id]:
            # print(g)
            noise.add(paulinoise, g)
        if len(noisy_gates_dict[factory_id]) == 0:
            noise.add(paulinoise, gates.Unitary) 
    else:
        noise.add(paulinoise, gates.Unitary) 
    return noise


def tr_noise_model(t1, t2, t, a0=0, factory_id='xid', init_state=None):
    # Define the noise model
    thermal_relaxation = ThermalRelaxationError(t1, t2, t, a0)
    noise = NoiseModel()
    if factory_id in noisy_gates_dict.keys():
        for g in noisy_gates_dict[factory_id]:
            # print(g)
            noise.add(thermal_relaxation, g)
        if len(noisy_gates_dict[factory_id]) == 0:
            noise.add(thermal_relaxation, gates.Unitary) 
    else:
        noise.add(thermal_relaxation, gates.Unitary)

    return noise 


def unitary_noise_model(matr=None, factory_id='xid'):
    # Generate a random unitary close to Id, U=exp(-i*H*lam)
    unitary_matr = matr if matr is not None else rand_unitary(lam=0.3)
    print("Unitary noise:\n", unitary_matr)
    # Define the noise model
    unitary_error = UnitaryError([1], [unitary_matr])
    noise = NoiseModel()
    if factory_id in noisy_gates_dict.keys():
        for g in noisy_gates_dict[factory_id]:
            # print(g)
            noise.add(unitary_error, g)
        if len(noisy_gates_dict[factory_id]) == 0:
            noise.add(unitary_error, gates.Unitary) 
    else:
        noise.add(unitary_error, gates.Unitary)
    return noise, unitary_matr


# Non-Markovian

basis = np.eye(4)

def ham_from_unitary(gate=np.eye(2), nqubits=2):
    # Create a 2-qubit matrix
    gate2q = np.kron(gate, np.eye(2)) if nqubits == 2 else gate
    # Get eigendecomposition of the gate matrix
    w, v = np.linalg.eig(gate2q)
    # Calculate H s.t. G = exp(-iH)
    ham = np.zeros(gate2q.shape, dtype=complex)
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
    if not np.allclose(gate2q, exph, atol=1e-7):
        print("WRONG FOR G=\n", gate2q.round(3),'\nexp(-iH)=\n', exph.round(3))

    return ham


def create_superoperator(gate_matr=np.eye(2), J=0, g0=0, g1=0):
    """ Create superoperator L s.t. d/dt rho = L rho for a concrete GKSL equation"""
    # dt = 40 * 1e-9
    # Variables
    o1 = 0 #np.pi / gate_time # z0 + z1
    
    # Create hamiltonian H=o0/2*g0 + o1/2*z1 + Jx0x1
    hamiltonian = np.array([[0, 0, 0, J],
                            [0, 0, J, 0],
                            [0, J, 0, 0],
                            [J, 0, 0, 0]], dtype=complex)
    # hamiltonian += (o0 / 2) * np.kron(gate_matr, np.eye(2)) 
    hamiltonian += (1 / 10 * GATE_TIME) * np.array([[0, 0, 1, 0],
                                                    [0, 0, 0, 1],
                                                    [1, 0, 0, 0],
                                                    [0, 1, 0, 0]], dtype=complex)

    # gate_z2 = np.kron(np.eye(2), np.array([[1, 0], [0, -1]])) 
    # hamiltonian += ham_from_unitary(gate_z2) / dt

    gate_to_hamiltonian = ham_from_unitary(gate_matr)
    # gate_to_hamiltonian = np.kron(gate_to_hamiltonian, np.eye(2))
    hamiltonian += gate_to_hamiltonian / GATE_TIME

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
                    l_matr[i * 4 + j][k * 4 + l] = (k_vec @ L_ij @ l_vec)[0][0]

    # print("\nLMATR =\n", l_matr.round(3))
    # Calculate exp(dt * L)
    exp_l = expm(GATE_TIME * l_matr)
    # print("\nEXPL =\n", exp_l.round(3))
    return exp_l


def superoperator_to_choi(sup):
    # choi = np.zeros(sup.shape, dtype=complex)
    # for i in range(4):
    #     i_vec = basis[i].reshape(-1, 1) 
    #     for j in range(4):
    #         j_vec = basis[j].reshape(1, -1) 
    #         ij_matr = i_vec @ j_vec
    #         sup_state = (sup @ ij_matr.reshape(-1, 1)).reshape(4, 4)
    #         choi += np.kron(sup_state, ij_matr)
    # print("\nCHOI =\n", choi.round(3))

    choi = liouville_to_choi(sup)
    return choi


def my_choi_to_kraus(choi):
    # Get Kraus operators M_i = sqrt(w_i) * unvec(v_i) 
    # kraus_list = []
    # eigvals, eigvecs = np.linalg.eigh(choi)
    # for i in range(len(eigvals)):
    #     w = eigvals[i]
    #     v = (eigvecs[:, i]).reshape(4, 4)
    #     if abs(w) > 1e-9:
    #         kraus_list.append(((0, 1), np.sqrt(w) * v))
            # print(f"\nKRAUS{len(kraus_list)} =\n", (np.sqrt(w) * v).round(3))
    kraus_ops, coefficients = choi_to_kraus(choi)
    kraus_list = []
    for i in range(len(kraus_ops)):
        kraus_list.append(((0, 1), (coefficients[i]) * kraus_ops[i]))
    return kraus_list # gates.KrausChannel(kraus_list)


def kraus_noise_model(factory_id='xid', J=0, g0=0, g1=0):
    gate_list = gates_dict[factory_id]
    implementation_list = []

    for g in gate_list:
        if len(noisy_gates_dict[factory_id]) != 0 and type(g) not in noisy_gates_dict[factory_id]:
            # g_matr = g.asmatrix(backend)
            # gate12_matr = np.kron(g_matr, np.array([[1, 0], [0, -1]]))
            # gate12 = gates.Unitary(gate12_matr, 0, 1)
            implementation_list.append(g)
        else:
            g_matr = g.asmatrix(backend)
            superop = create_superoperator(g_matr, J, g0, g1)
            choi = superoperator_to_choi(superop)
            kraus_list = my_choi_to_kraus(choi)
            kraus_channel = gates.KrausChannel(kraus_list)
            implementation_list.append(kraus_channel)
    return implementation_list


def check_kraus_gates(factory_id='xid'):
    state1 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    state2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    state30 = np.array([[3/4, 3/8-np.sqrt(3)/8*1j], [3/8+np.sqrt(3)/8*1j, 1/4]])
    state31 = np.array([[1/4, 0], [0, 3/4]])
    state3 = np.kron(state30, state31)
    states = [state1, state2, state3]

    gate_list = gates_dict[factory_id]

    for g in gate_list:
        g_matr = g.asmatrix(backend)
        superop = create_superoperator(g_matr, 0, 0, 0)
        choi = superoperator_to_choi(superop)
        kraus_list = my_choi_to_kraus(choi)

        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        gate_2q = np.kron(g_matr, np.eye(2))

        for state in states:
            gate_state = gate_2q @ state @ gate_2q.T.conj()

            kraus_state = np.zeros(state.shape, dtype=complex)
            for kraus in kraus_list:
                k = kraus[-1]
                kraus_state += k @ state @ k.T.conj()

            is_eq = np.allclose(gate_state, kraus_state, atol=1e-5)
            if not is_eq:
                print("Gate:\n", np.kron(g_matr, np.eye(2)).round(3))
                for kraus in kraus_list:
                    print("Kraus:\n", np.round(kraus[-1], 3))
                # print("State:\n", state.round(3))
                # print("G r G:\n", gate_state.round(3))
                # print("K r K:\n", kraus_state.round(3))
                return False
    print("WORKS")
    return True


def test_filterrb(factory_id='xid', noise_label='pauli', noise_params=[], ndecays=1, init_state=np.array([[0, 0], [0, 1]])):
    # gates_keys = ["cliffords", "xid", "four", "paulis", "x", "id"]
    # noise_labels = ['pauli', 'tr', 'unitary']
    nqubits = 1
    max_depth = 30
    depths = range(1, max_depth + 1)
    runs = 300
    nshots = 1024
    noise=None

    if noise_label == 'unitary':
        # Random unitary noise
        noise, noise_params = unitary_noise_model(factory_id=factory_id)
    elif len(noise_params) == 0:
        noise_label = 'No'
        noise = None
    elif noise_label == 'pauli':
        # Pauli noise: px, py, pz
        # noise_params = [0.05, 0.02, 0.03]
        noise = pauli_noise_model(*noise_params, factory_id=factory_id) 
    elif noise_label == 'tr':
        # Thermal relaxation error: t1, t2, t, a0=0
        noise = tr_noise_model(*noise_params, factory_id=factory_id)
    else:
        noise_label = 'No'
        noise = None

    # expected_decays = get_decays(factory_id, noise_label, noise_params)
    # print("Expected")
    # print(expected_decays)

    cs, ds = check_validation(factory_id, noise_label, noise_params)
    th_result = "F_{\\lambda}(m)\\approx "
    cnt = 0
    for i in range(len(cs)):
        if np.abs(cs[i]) > 1e-5 and np.abs(ds[i]) > 1e-5:
            cnt += 1
            c = str(np.round((np.real(cs[i])), 3)) if np.abs(np.imag(cs[i])) < 1e-3 else "(" + str(np.round((np.real(cs[i])), 3)) + ("+" if np.imag(cs[i]) > 0 else "-") + str(np.round(np.abs(np.imag(cs[i])), 3)) + "i)"
            d = str(np.round((np.real(ds[i])), 3)) if np.abs(np.imag(ds[i])) < 1e-3 else "(" + str(np.round((np.real(ds[i])), 3)) + ("+" if np.imag(ds[i]) > 0 else "-") + str(np.round(np.abs(np.imag(ds[i])), 3)) + "i)"
            
            if cnt > 1:
                th_result += " + "
            th_result += c + "\\cdot" + d + "^m"
    if cnt == 0:
            th_result += "0"
    th_result += "$"

    if noise_label == 'unitary':
        noise_params = noise_params.round(3)

    # init_state_pi2 = np.array([[0, 0], [0, 1]], dtype=complex) # np.array([[0.5, 0+0.5j], [0-0.5j, 0.5]])
    title = "$\\text{" + factory_id + " RB. " + noise_label + " noise: " + str(noise_params) + ". Expected: }\\\\" + th_result
    perform(nqubits, depths, runs, nshots, noise=noise, title=title, factory_id=factory_id, coeffs=cs, decays=ds)# , init_state=init_state_pi2)


def test_nonmark_filterrb(factory_id="", j_val=0, g0_val=0, g1_val=0, init_state=np.array([[1, 0], [0, 0]])):
    nqubits = 2
    max_depth = 30
    depths = range(1, max_depth + 1)
    runs = 1500
    nshots = 2048

    while factory_id not in gates_keys:
        factory_id = input("factory_id: ")
    if j_val == 0 and g0_val == 0 and g1_val == 0:
        j_val = float(input("Type j:"))
        g0_val = float(input("Type g0:"))
        g1_val = float(input("Type g1:"))

    # dt = 40 * 1e-9
    if (check_kraus_gates(factory_id)):
        noise_params = [j_val,  g0_val, g1_val]
        nonmark_gates = kraus_noise_model(factory_id, *noise_params)

        cs, ds = nonm_validation(factory_id, *noise_params) # check_validation(factory_id, noise_label, noise_params)
        th_result = "F_{\\lambda}(m)\\approx "
        cnt = 0
        for i in range(len(cs)):
            if np.abs(cs[i]) > 1e-5 and np.abs(ds[i]) > 1e-5:
                cnt += 1
                c = str(np.round((np.real(cs[i])), 3)) if np.abs(np.imag(cs[i])) < 1e-3 else "(" + str(np.round((np.real(cs[i])), 3)) + ("+" if np.imag(cs[i]) > 0 else "-") + str(np.round(np.abs(np.imag(cs[i])), 3)) + "i)"
                d = str(np.round((np.real(ds[i])), 3)) if np.abs(np.imag(ds[i])) < 1e-3 else "(" + str(np.round((np.real(ds[i])), 3)) + ("+" if np.imag(ds[i]) > 0 else "-") + str(np.round(np.abs(np.imag(ds[i])), 3)) + "i)"
                
                if cnt > 1:
                    th_result += " + "
                th_result += c + "\\cdot" + d + "^m"
        if cnt == 0:
            th_result += "0"
        th_result += "$"
        title = "$\\text{" + factory_id + " RB. Non-Markovian noise: " + str(noise_params) + ". Expected: }\\\\" + th_result
        
        perform(nqubits, depths, runs, nshots, noise=None, title=title, factory_id=factory_id, gates_list=nonmark_gates, coeffs=cs, decays=ds)
    else:
        print(f"Non Markovian model for gate set '{factory_id}' is not implemented yet.")


def store_data(factory_id = 'xid', runs=700, filename=""):
    nqubits = 2
    depths = range(1, 30)
    nshots = 1024
    init_state = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    import csv
    filename = "/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/" + filename + "-" + factory_id + "-" + str(runs) + ".csv"
    f1 = open(filename, 'w')
    writer = csv.writer(f1)
    header = ["J", "g0", "g1", "Re(p)", "Im(p)", "Re(a)", "Im(a)", "Count"]
    writer.writerow(header)

    cnt = 0
    ks = np.array([100, 10, 7, 5, 2, 1, 1/2, 1/3, 1/5, 1/7, 1/10]) #  
    for k_g0 in ks:
        g0 = 1/(k_g0 * GATE_TIME) if k_g0 != 100 else 0
        for k_g1 in ks: # [0, 0.2, 0.5, 0.7, 0.9, 1, 2.5, 5, 10]:# [0, 0.7, 0.9, 1, 1.25, 2, 2.5, 4, 5, 10]:
            g1 = 1/(k_g1 * GATE_TIME) if k_g1 != 100 else 0
            for k_j in ks: # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30]:
                j = 1/(k_j * GATE_TIME) if k_j != 100 else 0
                cnt += 1
                # print(cnt)
                nonmark_gates = kraus_noise_model(factory_id, j, g0, g1)
                fitting_func = fit_expn_func
                ndecays = 4
                if len(init_state) != 2 ** nqubits:
                    while len(init_state) < 2 ** nqubits:
                        init_state = np.kron(init_state, init_state)
                try:
                    # Initiate the circuit factory and the faulty Experiment object.
                    factory = NonMarkovianFactory(nqubits, depths, runs, qubits=None, factory_id=factory_id, gate_set=nonmark_gates)
                    # SingleCliffordsFilterFactory(nqubits, depths, runs, qubits=qubits)
                    irrep_basis = irrep_dict[factory_id]
                    experiment = FilterRBExperiment(factory, nshots, noisemodel=None, init_state=init_state, irrep_basis=irrep_basis)
                    # Execute the experiment.
                    experiment.execute()
                    # Get experimental decays
                    experiment.apply_task(single_qubit_filter)
                    result = FilterRBResult(experiment.dataframe, fitting_func, "", ndecays)
                    coefs, params = result.return_decay_parameters() 
                    print(j, g0, g1, len(params))
                    for i in range(len(params)):
                        p = params[i]
                        a = coefs[i]
                        if np.abs(p) > 1e-5 and np.abs(a) > 1e-5:
                            data = [k_j, k_g0, k_g1, np.real(p), np.imag(p), np.real(a), np.imag(a), len(params)]
                            writer.writerow(data)
                except ValueError:
                    print("\nValueError for ", j, g0, g1)
                    continue

    f1.close()


def tr_distance(m=1, J=0, g0=0, g1=0):
    from fourier_info import single_cliffords_list
    gate_set = single_cliffords_list()
    gate_inds = np.random.randint(0, len(gate_set), size=m)
    c = models.Circuit(2, density_matrix=True)
    for i in gate_inds:
        g_matr = gate_set[i]
        superop = create_superoperator(g_matr, J, g0, g1)
        choi = superoperator_to_choi(superop)
        kraus_list = my_choi_to_kraus(choi)
        kraus_channel = gates.KrausChannel(kraus_list)
        c.add(kraus_channel)
    c.add(gates.M(0))
    # print(c.draw())

    dists = []
    
    state0 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    state1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=complex)
    for init_state in [state0, state1]:
        ideal_exec = c.execute(state0)
        ideal_state0 = ideal_exec.state()

        ideal_exec = c.execute(state1)
        ideal_state1 = ideal_exec.state()

        tr_dist = np.trace(np.abs(ideal_state0-ideal_state1)) / 2
        dists.append(tr_dist)
        # print(np.round(tr_dist, 3))
    return dists[0], dists[1]


if __name__ == '__main__':
    # for runs in [1, 20, 100]:
    #     print(runs)
    #     avg_dists = []
    #     for m in range(11):
    #         avg_dist = 0
    #         for _ in range(runs):
    #             avg_dist += tr_distance(m, 10, 0, 0)[0]
    #         avg_dists.append(avg_dist / runs)
    #     print("[", end="")
    #     for i in range(len(avg_dists)):
    #         print(avg_dists[i], end=", " if i < len(avg_dists)-1 else "")
    #     print("]")
    #     print(np.array(avg_dists).round(3), '\n')

    # # Store experimental non-Mark decay params for factory_id, runs
    # factory_id = 'xid'
    # runs = 750
    # store_data(factory_id, runs, filename='new-data')
    # store_data('x', runs, filename='nm')
    # store_data('paulis', runs, filename='nm')

    # Test Markovian filter RB
    # factory_id = 'xid' # xid, four, paulis, cliffords
    # noise_label = 'unitary' # No, pauli, tr, unitary
    # noise_params = [] # [0.1, 0.05, 0.01]
    # test_filterrb(factory_id, noise_label, noise_params) #, init_state=np.array([[3/4, 3/8-np.sqrt(3)/8*1j], [3/8+np.sqrt(3)/8*1j, 1/4]]))

    # Test non-Markovian filter RB
    factory_id = "cliffords"
    j = 1 / (1 * GATE_TIME)
    g0 = 0 #1e9 / (6933)
    g1 = 0 #1e9 / (8479) # 1 / (2 * gate_time)
    test_nonmark_filterrb(factory_id, j, g0, g1)

    # from fourier_info import single_cliffords_list, ONEQUBIT_CLIFFORD_PARAMS
    # cliffs = single_cliffords_list() 
    # # print(len(cliffs))

    # for cl in cliffs:
    #     print("np.array([")
    #     for row in range(cl.shape[0]):
    #         print("[", end="")
    #         for col in range(cl.shape[1]):
    #             print(cl[row][col], end=", " if col < cl.shape[1]-1 else "")
    #         print("]", end=", \n" if row < cl.shape[0]-1 else "\n")
    #     print("]),")
    
    # s3 = np.array([[1, 0], [0, -1]])
    # s2q = np.kron(s3, np.eye(2))
    # cliffs = [s3, np.eye(2)]

    # for i in range(len(cliffs)):
    #     print(i)
    #     gate = cliffs[i]
    #     print(ONEQUBIT_CLIFFORD_PARAMS[i])
    #     print(gate.round(3))

    #     w, v = np.linalg.eig(gate)
    #     res = np.zeros(gate.shape, dtype=complex)
    #     ham_w = []
    #     ham = np.zeros(gate.shape, dtype=complex)
    #     for i in range(len(w)):
    #         val = 1j * np.log(np.abs(w[i])) + np.pi if w[i] < 0 else 1j * np.log(w[i])
    #         ham += (val) * (v[:, i].reshape(-1, 1) @ v[:, i].reshape(1, -1).conj())

    #     ham2q = np.kron(ham, np.eye(2))
    #     exph = expm(-1j * ham2q)

    #     gate2q = np.kron(gate, np.eye(2))

    #     if np.allclose(exph, gate2q):
    #         print(f"\nWorks\n")
    #     else:
    #         print("\nDoes not work:\n", exph)

    # check_kraus_gates('cliffords')