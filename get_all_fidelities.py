import os
import pickle

import qibo

from qibocal.protocols.characterization.RB import (
    interleaved,
    simulfiltered,
    standard_rb,
    xid_rb,
)

# Set backend and platform.
runcard = "/home/users/jadwiga.wilkens/qibolab/src/qibolab/runcards/qw5q_gold_qblox.yml"
backend_name = "qibolab"
platform_name = "qw5q_gold_qblox"
qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
backend = qibo.backends.GlobalBackend()
platform = backend.platform
platform.connect()
platform.setup()
platform.start()
# Set directory where to store results.
directory = "rb_results10"
if not os.path.isdir(directory):
    os.mkdir(directory)


# nqubits = 5
# niter = 20
# all_qubits = [[k] for k in range(4)]
# standardrb_results = []

qubits_list = [[2], [3]]
nqubits = 5
qubits = [2, 3]
niter = 2
depths = [2, 5, 10]  # ,15,20,25,30]
nshots = 1024


# params = simulfiltered.RBParameters(nqubits, qubits, depths, niter, nshots)
# scan = simulfiltered.Scan(params.nqubits, params.depths * params.niter, params.qubits)

# data_list = []
# # Iterate through the scan and execute each circuit.
# for c in scan:
#     # The inverse and measurement gate don't count for the depth.
#     depth = c.depth - 1
#     platform.stop()
#     platform.disconnect()

#     qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
#     backend = qibo.backends.GlobalBackend()
#     platform = backend.platform
#     platform.connect()
#     platform.setup()
#     platform.start()

#     samples = c.execute(nshots=params.nshots).samples()
#     # Every executed circuit gets a row where the data is stored.
#     data_list.append({"depth": depth, "samples": samples, "circuit": c})
# data = simulfiltered.RBData(data_list)
# data.attrs = params.__dict__
# with open(f"{directory}/simulf_data_qubit{qubits[0]}.pkl", "wb") as f:
#     pickle.dump(data, f)
# result = simulfiltered.extract(data)
# result.fit()
# result.calculate_fidelities()
# with open(f"{directory}/simulf_result_qubit{qubits[0]}.pkl", "wb") as f:
#     pickle.dump(result, f)
# print(result.fidelity_dict)

for qubits in qubits_list:
    params = standard_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
    scan = standard_rb.StandardRBScan(
        params.nqubits, params.depths * params.niter, params.qubits
    )
    data_list = []
    # Iterate through the scan and execute each circuit.
    for c in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (c.depth - 2) if c.depth > 1 else 0
        # platform.stop()
        # platform.disconnect()

        # qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
        # backend = qibo.backends.GlobalBackend()
        # platform = backend.platform
        # platform.connect()
        # platform.setup()
        # platform.start()

        samples = c.execute(nshots=params.nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append({"depth": depth, "samples": samples})
    data = standard_rb.RBData(data_list)
    data.attrs = params.__dict__
    with open(f"{directory}/standard_data_qubit{qubits[0]}.pkl", "wb") as f:
        pickle.dump(data, f)
    # data = standard_rb.acquire(params)
    result = standard_rb.extract(data)
    result.fit()
    result.calculate_fidelities()
    with open(f"{directory}/standard_result_qubit{qubits[0]}.pkl", "wb") as f:
        pickle.dump(result, f)
    # print(result.fidelity_dict)

# depths = list(range(1, 15, 2))
# for qubits in qubits_list:
#     params = xid_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
#     scan = xid_rb.XIdScan(params.nqubits, params.depths * params.niter, params.qubits)
#     data_list = []
#     # Iterate through the scan and execute each circuit.
#     for c in scan:
#         # The inverse and measurement gate don't count for the depth.
#         depth = c.depth - 1
#         platform.stop()
#         platform.disconnect()

#         qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
#         backend = qibo.backends.GlobalBackend()
#         platform = backend.platform
#         platform.connect()
#         platform.setup()
#         platform.start()
#         nx = len(c.gates_of_type("x"))
#         samples = c.execute(nshots=params.nshots).samples()
#         # Every executed circuit gets a row where the data is stored.
#         data_list.append({"depth": depth, "samples": samples, "nx": nx})
#     data = xid_rb.RBData(data_list)
#     data.attrs = params.__dict__
#     with open(f"{directory}/xid_data_qubit{qubits[0]}.pkl", "wb") as f:
#         pickle.dump(data, f)
#     # data = xid_rb.acquire(params)
#     result = xid_rb.extract(data)
#     result.fit()
#     with open(f"{directory}/xid_result_qubit{qubits[0]}.pkl", "wb") as f:
#         pickle.dump(result, f)


# params = interleaved.RBParameters(nqubits, qubits, depths, niter, nshots)
# scan = interleaved.Scan(params.nqubits, params.depths * params.niter, params.qubits)
# data_list = []
# # Iterate through the scan and execute each circuit.
# for c in scan:
#     # The inverse and measurement gate don't count for the depth.
#     depth = (c.depth + 1) // 3 if c.depth > 0 else 0
#     platform.stop()
#     platform.disconnect()

#     qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
#     backend = qibo.backends.GlobalBackend()
#     platform = backend.platform
#     platform.connect()
#     platform.setup()
#     platform.start()

#     samples = c.execute(nshots=params.nshots).samples()
#     # Every executed circuit gets a row where the data is stored.
#     data_list.append({"depth": depth, "samples": samples})
# data = interleaved.RBData(data_list)
# data.attrs = params.__dict__
# with open(f"{directory}/interleaved_data_qubit{qubits[0]}.pkl", "wb") as f:
#     pickle.dump(data, f)
# # data = interleaved.acquire(params)
# result = interleaved.extract(data)
# result.fit()
# # result.calculate_fidelities()
# # standardrb_results.append(result)
# with open(f"{directory}/interleaved_result_qubit{qubits[0]}.pkl", "wb") as f:
#     pickle.dump(result, f)


# for qubits in all_qubits:
#     params = standard_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
#     scan = standard_rb.StandardRBScan(params.nqubits, params.depths * params.niter, params.qubits)
#     data_list = []
#     # Iterate through the scan and execute each circuit.
#     for c in scan:
#         # The inverse and measurement gate don't count for the depth.
#         depth = (c.depth - 2) if c.depth > 1 else 0
#         platform.stop()
#         platform.disconnect()

#         qibo.set_backend(backend=backend_name, platform=platform_name, runcard=runcard)
#         backend = qibo.backends.GlobalBackend()
#         platform = backend.platform
#         platform.connect()
#         platform.setup()
#         platform.start()

#         samples = c.execute(nshots=params.nshots).samples()
#         # Every executed circuit gets a row where the data is stored.
#         data_list.append({"depth": depth, "samples": samples})
#     data = standard_rb.RBData(data_list)
#     data.attrs = params.__dict__
#     # data = standard_rb.acquire(params)
#     result = standard_rb.extract(data)
#     result.fit()
#     result.calculate_fidelities()
#     standardrb_results.append(result)
#     with open(f"{directory}/standard_qubit{qubits[0]}.pkl", "wb") as f:
#         pickle.dump(result, f)
#     print(result.fidelity_dict)


# print(result.fidelity_dict)
# depths = [1, 3, 5, 7, 9]
# for qubits in all_qubits:
#     params = xid_rb.RBParameters(nqubits, qubits, depths, niter, nshots)
#     data = xid_rb.acquire(params)
#     result = xid_rb.extract(data)
#     result.fit()
#     standardrb_results.append(result)
#     with open(f"{directory}/xid_qubit{qubits[0]}.pkl", "wb") as f:
#         pickle.dump(result, f)

platform.stop()
platform.disconnect()
