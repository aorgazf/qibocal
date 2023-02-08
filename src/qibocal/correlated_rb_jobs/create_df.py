import csv
import numpy as np
from constants import GATE_TIME
from time import time

from correlated_poles import get_nonmark_decays

header = ["kx", "ky", "kz", "g0", "g1", "p0", "p1", "Real", "Imaginary"]

gate_keys = ["0-1", "1-1", "01-11", "01-01", "01-10"]
# files = []
# files_dict = {}
# for key in gate_keys:
#     file = open('/Users/liza/Documents/Plots/InteractivePlot/correlated_sliders/correlated_poles' + key + '.csv', 'w')
#     files.append(file)
#     files_dict[key] = csv.writer(file)
#     files_dict[key].writerow(header)

def get_qubits_irreps(key="0-0"):
    qubit_inds = key.split("-")[0]
    q_list = []
    for c in qubit_inds:
        q_list.append(int(c))
    irrep_inds = key.split("-")[1]
    i_list = []
    for c in irrep_inds:
        i_list.append(int(c))
    return q_list, i_list



ks = [0, 20, 10, 5, 2]

for key in gate_keys:
    file = open('/home/users/yelyzaveta.vodovozova/qibocal/src/qibocal/correlated_rb_jobs/poles' + key + '.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(header)

    q, i = get_qubits_irreps(key)

    for kx in range(len(ks)):
        jx = 1 / (kx * GATE_TIME) if kx != 0 else 0
        for ky in range(len(ks)):
            jy = 1 / (ky * GATE_TIME) if ky != 0 else 0
            for kz in range(len(ks)):
                jz = 1 / (kz * GATE_TIME) if kz != 0 else 0
                for g0 in [0, 1 / (10 * GATE_TIME)]:
                    for g1 in [0, 1 / (10 * GATE_TIME)]:
                        for p0 in [0, 0.1]:
                            for p1 in [0, 0.1]:
                                for key in gate_keys:
                                    q, i = get_qubits_irreps(key)
                                    decay = np.max(get_nonmark_decays(q, i, [jx, jy, jz], [g0, g1], [p0, p1]))
                                    data = [kx, ky, kz, g0, g1, p0, p1, np.real(decay), np.imag(decay)]
                                    writer.writerow(data)
    file.close()