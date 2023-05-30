from dataclasses import dataclass
from typing import Iterable, Union

from qibo.gates import X
from qibo.noise import NoiseModel

from qibocal.auto.operation import Routine
from qibocal.calibrations.niGSC.simulfilteredrb import ModuleFactory as Scan
from qibocal.calibrations.niGSC.simulfilteredrb import filter_function
from qibocal.protocols.characterization.RB.result import (
    DecayResult,
    ResultContainer,
    get_hists_data,
)
from qibocal.protocols.characterization.RB.utils import extract_from_data

from .data import RBData
from .params import RBParameters

NoneType = type(None)


@dataclass
class SimulResult(DecayResult):
    pass


def setup_scan(params: RBParameters) -> Iterable:
    return Scan(params.nqubits, params.depths * params.niter, params.qubits)


def execute(
    scan: Iterable,
    nshots: Union[int, NoneType] = None,
    noise_model: Union[NoiseModel, NoneType] = None,
):
    # Execute
    data_list = []
    for c in scan:
        depth = c.depth
        # nx = c.gate_types[X]
        if noise_model is not None:
            c = noise_model.apply(c)
        samples = c.execute(nshots=nshots).samples()
        data_list.append({"depth": depth, "samples": samples, "circuit": c})
    return data_list


from itertools import product

import numpy as np


def aggregate(data: RBData):
    result_list = []
    for kk in range(2 ** len(data.attrs["qubits"])):
        # df = data.get(["depth", f"irrep{kk}"])
        extract_from_data
        # Histogram
        hists = get_hists_data(data, f"irrep{kk}")
        # Build the result object
        result_list.append(
            SimulResult(
                *extract_from_data(data, f"irrep{kk}", "depth", "mean"),
                hists=hists,
                meta_data=data.attrs,
            )
        )
    return ResultContainer(result_list)


def acquire(params: RBParameters, *args) -> RBData:
    scan = setup_scan(params)
    if params.noise_model:
        from qibocal.calibrations.niGSC.basics import noisemodels

        noise_model = getattr(noisemodels, params.noise_model)(*params.noise_params)
    else:
        noise_model = None
    # Here the platform can be connected
    data = execute(scan, params.nshots, noise_model)
    for datarow in data:
        datarow = filter_function(datarow["circuit"], datarow)

    simul_data = RBData(data)
    simul_data.attrs = params.__dict__

    return simul_data


def extract(data: RBData):
    result = aggregate(data)
    result.fit()
    return result


def plot(data: RBData, result: SimulResult, *args):
    table_str = "".join(
        [f" | {key}: {value}<br>" for key, value in {**result.meta_data}.items()]
    )
    fig = result.plot()
    return [fig], table_str


simulfiltered_rb = Routine(acquire, extract, plot)
