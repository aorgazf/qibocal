from copy import deepcopy
import numpy as np
from sympy import *
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger


# # Create a list of 1-qubit paulis
# s0 = np.matrix([[1,0],[0,1]])
# s1 = np.matrix([[0,1],[1,0]])
# s2 = np.matrix([[0,-1j],[1j,0]])
# s3 = np.matrix([[1,0],[0,-1]])
# paulis = np.array([s0, s1, s2, s3]) 

# def base_convert(num, base, length=2):
#     ''' Convert a number to some base '''
#     if not(2 <= base <= 9):
#         return

#     new_num = ''
#     while num > 0:
#         new_num = str(num % base) + new_num
#         num //= base

#     while len(new_num) < length:
#         new_num = '0' + new_num 
#     return np.array(list(new_num), dtype=int)


# def get_paulis(n, norm=True):
#     ''' Get array of all n-qubit Paulis '''
        
#     res = []
#     for i in range(4 ** n):
#         inds = base_convert(i, 4, length=n) 
#         el = paulis[inds[0]]
#         for j in range(1, n):
#             el = np.kron(el, paulis[inds[j]])
#         res.append(el)
        
#     return res


# paulis_2q = get_paulis(2)


# def to_pauli_op(u=Matrix(np.eye(2)), dtype="complex"):
#     ''' Transfrom a unitray operator to Pauli-Liouville superoperator'''
#     dim = u.shape[0]
#     qubits_num = int(np.log2(dim))
#     if qubits_num == 1:
#         nq_paulis = deppcopy(paulis)
#     elif qubits_num == 2:
#         nq_paulis = deepcopy(paulis_2q)
#     else:
#         nq_paulis = get_paulis(qubits_num)
    
#     pauli_dim = len(nq_paulis)
#     res = np.zeros((pauli_dim, pauli_dim), dtype=dtype)
#     for i in range(pauli_dim):
#         Pi = Matrix(paulis_2q[i])
#         for j in range(pauli_dim):
#             Pj = Matrix(paulis_2q[j])
#             m_el = Pi * (u * Pj * Dagger(u)) / dim
#             trace_el = 0
#             for row in range(m_el.shape[0]):
#                 trace_el = trace_el + m_el[row,row]
#             res[i][j] = trace_el
            
#     return res

# to_pauli_op(np.eye(4)).shape

# # Symbols
# o0 = Symbol('Omega0', positive=True)
# o1 = Symbol('Omega1', positive=True)
# J = Symbol('J', positive=True)
# g0 = Symbol('gamma0', positive=True)
# g1 = Symbol('gamma1', positive=True)

# # Hamiltonian 
# hamiltonian = np.array([[o1 / 2, 0, o0 / 2, J],
#                        [0, -o1 / 2, J, o0 / 2],
#                        [o0 / 2, J, o1 / 2, 0],
#                        [J, o0 / 2, 0, -o1 / 2]])

# # Set constant values
# dt_value = 0.1
# o1_value = pi / dt_value

# # Create H for Id
# o0_value = 0
# h_id = Matrix(hamiltonian).subs([(o0, o0_value), (o1, o1_value)])

# # Compute U_I = exp(-itH)                     
# exp_id = (-I*0.1*h_id).exp()

# # Get superoperator representation of U_I
# sup_id = TensorProduct(exp_id, exp_id)

# # Get Pauli-Liouville representation of U_I
# pauli_id = to_pauli_op(exp_id, dtype="object")

# # Create H for X
# o0_value = pi / dt_value
# h_x = Matrix(hamiltonian).subs([(o0, o0_value), (o1, o1_value)])

# # Compute U_X = exp(-itH) = sum[exp(-itl)*P_l]                                                        
# ith_x = -I * dt_value * h_x
# P, D = ith_x.diagonalize()
# exp_x = P * D.exp() * P.inv()


# # Get superoperator representation of U_X
# sup_x = TensorProduct(exp_x, exp_x)

# # Get Pauli-Liouville representation of U_X
# pauli_x = to_pauli_op(exp_x, dtype="object")

# # Compute resulting Fourier in computational basis
# fourier_comp = 1 / 2  * (sup_id - sup_x)

# # Compute resulting Fourier in Pauli basis
# fourier_pauli = 1 / 2 * (pauli_id - pauli_x)

# # Check for J=10 and J=5
# # eigvals = fourier_comp.eigenvals()

# # eig_list = np.array((list(eigvals.keys())), dtype=complex)
# # print(eig_list)
# # print(fourier_comp.subs([(J, 5)]).eigenvals())

# # Displaying the array
# # print('Eig list:\n\n', eigvals)

# # Create a text file
# file = open("/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/sym_fc.txt", "w+")
 
# # Saving the array in a text file
# content = str(fourier_comp)
# file.write(content)
# file.close()

# # Create a text file
# file = open("/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/sym_fp.txt", "w+")
 
# # Saving the array in a text file
# content = str(fourier_pauli)
# file.write(content)
# file.close()
 
# Displaying the contents of the text file
# file = open("file1.txt", "r")
# content = file.read()
 
# print("\nContent in file1.txt:\n", content)
# file.close()

from copy import deepcopy
import numpy as np
from sympy import *
from sympy.physics.quantum.dagger import Dagger
from sympy.solvers import solve

o0 = Symbol('Omega0', positive=True)
o1 = Symbol('Omega1', positive=True)
J = Symbol('J', positive=True)
g0 = Symbol('gamma0', positive=True)
g1 = Symbol('gamma1', positive=True)
dt = 0.1 # Symbol('t', positive=True)

basis = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

def create_lindblad(is_x=False, is_z=True):
    # Variables
    o0 = pi / dt if is_x else 0
    o1 = pi / dt

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
    l_matr = np.zeros((16, 16), dtype=object)
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
    # exp_l = expm(dt * l_matr)

    return Matrix(l_matr)

l_id = create_lindblad()
l_x = create_lindblad(is_x=True)

exp_id = (dt*l_id).exp()

# Create a text file
file = open("/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/lindblad_exp_id.txt", "w+")
 
# Saving the array in a text file
content = str(exp_id)
file.write(content)
file.close()

try:
    exp_x = (dt*l_x).exp()
except error:
    P, D = (dt*l_x).diagonalize()
    exp_x = P * D.exp() * P.inv()

# Create a text file
file = open("/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/lindblad_exp_x.txt", "w+")

# Saving the array in a text file
content = str(exp_x)
file.write(content)
file.close()
 