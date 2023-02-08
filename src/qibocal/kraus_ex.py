from copy import deepcopy
import numpy as np
from sympy import *
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger

# Create a list of 1-qubit paulis
s0 = np.matrix([[1,0],[0,1]])
s1 = np.matrix([[0,1],[1,0]])
s2 = np.matrix([[0,-1j],[1j,0]])
s3 = np.matrix([[1,0],[0,-1]])
paulis = np.array([s0, s1, s2, s3]) 

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
        
    res = []
    for i in range(4 ** n):
        inds = base_convert(i, 4, length=n) 
        el = paulis[inds[0]]
        for j in range(1, n):
            el = np.kron(el, paulis[inds[j]])
        res.append(el)
        
    return res


paulis_2q = get_paulis(2)


def to_pauli_op(u=Matrix(np.eye(2)), dtype="complex"):
    ''' Transfrom a unitray operator to Pauli-Liouville superoperator'''
    dim = u.shape[0]
    qubits_num = int(np.log2(dim))
    if qubits_num == 1:
        nq_paulis = deppcopy(paulis)
    elif qubits_num == 2:
        nq_paulis = deepcopy(paulis_2q)
    else:
        nq_paulis = get_paulis(qubits_num)
    
    pauli_dim = len(nq_paulis)
    res = np.zeros((pauli_dim, pauli_dim), dtype=dtype)
    for i in range(pauli_dim):
        Pi = Matrix(paulis_2q[i])
        for j in range(pauli_dim):
            Pj = Matrix(paulis_2q[j])
            m_el = Pi * (u * Pj * Dagger(u)) / dim
            trace_el = 0
            for row in range(m_el.shape[0]):
                trace_el = trace_el + m_el[row,row]
            res[i][j] = trace_el
            
    return res

to_pauli_op(np.eye(4)).shape

# Symbols
o0 = Symbol('Omega0', positive=True)
o1 = Symbol('Omega1', positive=True)
J = Symbol('J', positive=True)
g0 = Symbol('gamma0', positive=True)
g1 = Symbol('gamma1', positive=True)

# Hamiltonian 
hamiltonian = np.array([[o1 / 2, 0, o0 / 2, J],
                       [0, -o1 / 2, J, o0 / 2],
                       [o0 / 2, J, o1 / 2, 0],
                       [J, o0 / 2, 0, -o1 / 2]])

# Set constant values
dt_value = 0.1
o1_value = np.pi / dt_value

# Create H for Id
o0_value = 0
h_id = Matrix(hamiltonian).subs([(o0, o0_value), (o1, o1_value)])

# Compute U_I = exp(-itH)                     
exp_id = (-I*0.1*h_id).exp()

# Get superoperator representation of U_I
sup_id = TensorProduct(exp_id, exp_id)

# Get Pauli-Liouville representation of U_I
pauli_id = to_pauli_op(exp_id, dtype="object")

# Create H for X
o0_value = np.pi / dt_value
h_x = Matrix(hamiltonian).subs([(o0, o0_value), (o1, o1_value)])

# Compute U_X = exp(-itH) = sum[exp(-itl)*P_l]                                                        
ith_x = -I * dt_value * h_x
P, D = ith_x.diagonalize()
exp_x = P * D.exp() * P.inv()

# Get superoperator representation of U_X
sup_x = TensorProduct(exp_x, exp_x)

# Get Pauli-Liouville representation of U_X
pauli_x = to_pauli_op(exp_x, dtype="object")

# Compute resulting Fourier in computational basis
# fourier_comp = 1 / 2  * (sup_id - sup_x)
# print("Fourier comp done")
# Simplify
# fourier_comp_s = simplify(fourier_comp)
# # Round
# fourier_comp_sr = deepcopy(fourier_comp_s)
# for a in preorder_traversal(fourier_comp_s):
#     if isinstance(a, Float):
#         fourier_comp_sr = fourier_comp_sr.subs(a, round(a, 1))

# Compute resulting Fourier in Pauli basis
fourier_pauli = 1 / 2 * (pauli_id - pauli_x)
print("Fourier pauli done")
# Simplify
# fourier_pauli_s = simplify(fourier_pauli)
# # Round
# fourier_pauli_sr = deepcopy(fourier_pauli_s)
# for a in preorder_traversal(fourier_pauli_s):
#     if isinstance(a, Float):
#         fourier_pauli_sr = fourier_pauli_sr.subs(a, round(a, 1))

# Check for J=10 and J=5
# eigvals = fourier_comp.eigenvals()

# eig_list = np.array((list(eigvals.keys())), dtype=complex)
# print(eig_list)
# print(fourier_comp.subs([(J, 5)]).eigenvals())

# Displaying the array
# print('Eig list:\n\n', fourier_comp)
# file = open("/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/simplified_fourier.txt", "w+")
 
# # Saving the array in a text file
# content = "Fourier in comp basis:\n\n" + str(fourier_comp_sr) + "\n\nFourier in Pauli-Liouville basis:\n\n" + str(fourier_pauli_sr) + "\n"
# file.write(content)
# file.close()

content = "Simplified and rounded to 3 digits Fourier in Pauli-Liouville basis \n\n"
for i in range(16):
    for j in range(16):
        print(i*16+j)
        el = simplify(fourier_pauli[i, j])
        elr = el
        for a in preorder_traversal(el):
            if isinstance(a, Float):
                elr = elr.subs(a, round(a, 1))
        content += str(i) + str(j) + "\n" + str(elr) + "\n\n"

file = open("/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/sr_pauli.txt", "w+")
file.write(content)
file.close()
 
# Displaying the contents of the text file
# file = open("file1.txt", "r")
# content = file.read()
 
# print("\nContent in file1.txt:\n", content)
# file.close()



"""import numpy as np
from qibo import models, hamiltonians
# create critical (h=1.0) TFIM Hamiltonian for three qubits
hamiltonian = hamiltonians.TFIM(3, h=1.0)
# initialize evolution model with step dt=1e-2
evolve = models.StateEvolution(hamiltonian, dt=1e-2)
# initialize state to |+++>
initial_state = np.ones(8) / np.sqrt(8)
# execute evolution for total time T=2
final_state2 = evolve(final_time=2, initial_state=initial_state)
print(final_state2)"""