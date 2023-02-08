import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# POVM
M_IC = np.array([[3 / 2, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 1/2, 0],
             [0, 0, 0, 1]]) / 8


M_Z = np.array([[1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1]])

# S^+_\lambda 
S_IC = np.array([[16, 0], [0, 8]])
S_Z = np.array([[0, 0], [0, 1]])

X_sign = np.array([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 1]])

# Gate Set
gate_i = np.eye(2)
gate_x = np.array([[0, 1], [1, 0]])
gate_set = [gate_i, gate_x]

# POVM
z0 = np.array([[1, 0], [0, 0]])
z1 = np.array([[0, 0], [0, 1]])
x0 = np.array([[0.5, 0.5], [0.5, 0.5]])
x1 = np.array([[0.5, -0.5], [-0.5, 0.5]])
y0 = np.array([[0.5, -0.5*1j], [0.5*1j, 0.5]])
y1 = np.array([[0.5, 0.5*1j], [-0.5*1j, 0.5]])

povm = [z0, z1] 
# print(povm)

# S+
s = np.zeros((4, 4), dtype=complex)
for e in povm:
    s += e.reshape(-1, 1) @ e.reshape(1, -1).conj() 
s /= (len(povm) // 2) ** 2
s_sign = s[2:, 2:]

s_plus = np.linalg.pinv(s)
s_plus_sign = np.linalg.pinv(s_sign)
# Initial state
initial_state = np.array([[1, 0], [0, 0]])
# np.array([[3/4, 3/8-np.sqrt(3)/8*1j], [3/8+np.sqrt(3)/8*1j, 1/4]])
# np.array([[1, 0], [0, 0]])


def irrep_sign(g):
    if np.array_equal(g, gate_x):
        return -1
    else:
        return 1

def pr_sign(state):
    res = 0
    for g in gate_set:
        res += irrep_sign(g) * (g @ state @ g.T.conj())
    return res.reshape(-1, 1)

def filter_sign(i, gate_list=[]):
    countx = 0
    for gate in gate_list:
        countx += int(np.array_equal(gate, gate_x))
    state_sign = s_plus @ pr_sign(initial_state)

    if countx % 2:
        state_sign = np.flip(state_sign, 0)

    return (povm[i].reshape(1, -1).conj() @ state_sign)[0][0]

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
    

def fourier_sign(noise=np.eye(4), noisy_gate=gate_x):
    res = np.zeros(noise.shape, dtype=complex)
    for gate in gate_set:
        g = to_pauli_op(gate)
        if np.array_equal(gate, noisy_gate):
            g = g @ noise
        res += irrep_sign(gate) * g 
    return res / len(gate_set)

# Check filter function
# for m in range(3):
#     print(m)
#     for i in range(len(povm)):
#         print(i, filter_sign(i, [gate_x] * m))

# Example: Pauli noise
# px, py, pz = (0.1, 0.03, 0.1)
# p_noise = np.array([[1, 0, 0, 0],
#                     [0, 1 - 2 * py - 2 * pz, 0, 0],
#                     [0, 0, 1 - 2 * px - 2 * pz, 0],
#                     [0, 0, 0, 1 - 2 * px - 2 * py]])

# Example: Thermal Relaxation Error
t1, t2, t, a0 = (0.05, 0.03, 0.01, 0)
p_noise = np.array([[1, 0, 0, 0], 
                    [0, np.exp(-t/t2), 0, 0],
                    [0, 0, np.exp(-t/t2), 0],
                    [(1-2*a0)*(np.exp(-t/t1)-1), 0, 0, np.exp(-t/t1)]])

def to_pauli_state(density=np.array([[1, 0], [0, 0]])):
    if density.shape == (2,2):
        a = density[0][0]
        b = density[0][1]
        return np.sqrt(2) * np.array([1 / 2, np.real(b), -np.imag(b), a - 1/2])
    else:
        return np.array([[1/2, 0], [0, 1/2]])

def partial_trace(a):
    res_shape = (a.shape[0] // 2, a.shape[1] // 2)
    res = np.zeros(res_shape, dtype=complex)
    for i in range(len(res)):
        for j in range(len(res[i])):
            res[i][j] = a[2*i+1][2*j+1] + a[2*i][2*j]
    return res

def get_abw():
    fourier_matr = fourier_sign(p_noise)
    # Get L, R
    w, v = np.linalg.eig(fourier_matr)
    print(v.round(3).real)
    dom_inds = []
    for j in range(len(v[0])):
        if v[2][j] == 1 or v[3][j] == 1:
            dom_inds.append(j)
    nondom_inds = [i for i in range(len(v[0])) if i not in dom_inds]
    r2 = v[:, nondom_inds]
    r1 = v[:, dom_inds]
    v_inv = np.linalg.inv(v)
    l2 = v_inv[nondom_inds, :]
    l1 = v_inv[dom_inds, :]
    
    # Get A, B
    M_sign = X_sign.T @ s 
    state_pauli = to_pauli_state(initial_state)
    
    Q_sign = (np.outer(state_pauli, state_pauli.conj()) @ X_sign @ s_plus_sign).T
    for_trace = np.outer(M_sign, Q_sign.conj())
    mq_matr = partial_trace(for_trace)
    
    return l1 @ mq_matr @ r1, l2 @ mq_matr @ r2, w[dom_inds], w[nondom_inds]

# a, b, wd, wn = get_abw()
# print("A =\n", a.real, "\n")
# print("B =\n", b.real, "\n")
# print("I =\n", np.diag(wd).round(3).real, "\n")
# print("O =\n", np.diag(wn).round(3).real, "\n")

def get_ab_f(fourier_matr):
    print(fourier_matr.round(3))
    # Get L, R
    w, v = np.linalg.eig(fourier_matr)
    print(v.round(3))
    dom_inds = []
    for j in range(len(v[0])):
        if v[2][j] > 0 or v[3][j] > 0:
            dom_inds.append(j)
    print(dom_inds)
    nondom_inds = [i for i in range(len(v[0])) if i not in dom_inds]
    r2 = v[:, nondom_inds]
    r1 = v[:, dom_inds]
    v_inv = np.linalg.inv(v)
    l2 = v_inv[nondom_inds, :]
    l1 = v_inv[dom_inds, :]
    
    # Get A, B
    M_sign = X_sign.T @ s 
    state_pauli = to_pauli_state(initial_state)
    
    Q_sign = (np.outer(state_pauli, state_pauli.conj()) @ X_sign @ s_plus_sign).T
    for_trace = np.outer(M_sign, Q_sign.conj())
    mq_matr = partial_trace(for_trace)
    print("L1 shape =", l1.shape)
    print("R2 shape =", r2.shape)
    print("MG shape =", mq_matr.shape)
    a = l1 @ mq_matr @ r1
    b = l2 @ mq_matr @ r2
    wd = w[dom_inds]
    wn = w[nondom_inds]
    print("A =\n", a.real, "\n")
    print("B =\n", b.real, "\n")
    print("I =\n", np.diag(wd).round(3).real, "\n")
    print("O =\n", np.diag(wn).round(3).real, "\n")



def omega(t):
    omega0 = np.pi / 30
    return 0 if t > np.pi/omega0 else omega0 

def solve_diffs(omega0=omega, t_start=0, t_end=60, dt=0.01, cl=[1], gl=[1], ol=[1], plot=False):
    n_steps = int(round((t_end-t_start)/dt))    # number of timesteps
    x_start = 1
    phi_start = 1

    X_arr = np.zeros(n_steps + 1, dtype=complex)  
    X_arr[0] = x_start    
    t_arr = np.zeros(n_steps + 1) 
    t_arr[0] = t_start           
    Phi_arr = np.zeros((2, n_steps + 1), dtype=complex)
    for phi in Phi_arr:
        phi[0] = phi_start

    # Euler's method
    for i in range (1, n_steps + 1):  
        X = X_arr[i-1]
        t = t_arr[i-1]
        
        sum_phi = 0
        for j in range(Phi_arr.shape[0]):
            Phi = Phi_arr[j][i-1]
            sum_phi += cl[j] * Phi
            dPhidt = -1j * cl[j] * X - (gl[j] / 2 + 1j * ol[j]) * Phi
            Phi_arr[j][i] = Phi + dt * dPhidt
            
        dXdt = -1j * omega0(t) * X - 1j * sum_phi
        X_arr[i] = X + dt*dXdt
        t_arr[i] = t + dt       

    if plot:
        # plotting the result
        fig = plt.figure()                                  
        plt.plot(t_arr, X_arr, label = 'x')   
        for i in range(len(Phi_arr)):
            plt.plot(t_arr, Phi_arr[i], label = f'$\phi_{i}$')
        plt.title('Title', fontsize = 12)    # add some title to your plot
        plt.xlabel('t', fontsize = 12)
        plt.ylabel('x(t), $\phi_l$(t)', fontsize = 12)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.grid(True)                        # show grid
        # plt.xaxis([t_start, t_end])     # show axes measures
        plt.legend()
        plt.savefig("/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/calibrations/protocols/plots" + datetime.now().strftime("%H:%M:%S"))


# cl = [0.2, 0.5]
# gl = [0.2, 0.5]
# ol = [0.2, 0.5]

# solve_diffs(cl=cl, gl=gl, ol=ol, plot=True)