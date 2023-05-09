from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame
from qibo.noise import NoiseModel
from qibolab.platforms.abstract import AbstractPlatform
from scipy.optimize import OptimizeResult, minimize

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.calibrations.niGSC.basics.plot import plot_qq
from qibocal.calibrations.niGSC.standardrb import (
    ModuleExperiment,
    ModuleFactory,
    build_report,
    get_aggregational_data,
    post_processing_sequential,
)


@dataclass
class NelderMeadRBParameters(Parameters):
    """NelderMead Randomized Benchmarking runcard inputs."""

    nqubits: int
    qubits: list
    depths: list
    runs: int
    nshots: int
    frequency_step: int = 100_000_000


@dataclass
class NelderMeadRBResults(Results):
    """NelderMead RB outputs."""

    df: DataFrame

    def save(self, path):
        self.df.to_pickle(path)


class NelderMeadRBData:
    """NelderMead RB data acquisition."""

    def __init__(self, minimize_result: OptimizeResult, experiment: ModuleExperiment):
        self.loss = minimize_result.fun
        self.optimal_frequency = minimize_result.x
        self.extra = minimize_result
        self.experiment = experiment

    def save(self, path):
        self.experiment.save(path)

    def load(self, path):
        self.experiment.load(path)


def rb_loss(params, qubit, experiment, platform):
    print(params, type(params))
    platform.single_qubit_natives[qubit]["RX"]["frequency"] = params[0]
    print(platform.single_qubit_natives[qubit]["RX"]["frequency"])
    experiment.perform(experiment.execute)
    post_processing_sequential(experiment)
    df = get_aggregational_data(experiment)
    rb_decay = df["popt"][0]["p"]
    print(np.abs(rb_decay - 1) * 10)
    return np.abs(rb_decay - 1) * 10


def _acquisition(
    params: NelderMeadRBParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> NelderMeadRBData:
    factory = ModuleFactory(
        params.nqubits, params.depths * params.runs, qubits=params.qubits
    )
    experiment = ModuleExperiment(factory, nshots=params.nshots)
    init_freq = platform.single_qubit_natives[params.qubits[0]]["RX"]["frequency"]
    print(init_freq, type(init_freq))
    minimize_result = minimize(
        rb_loss,
        init_freq,
        args=(params.qubits[0], experiment, platform),
        options={"maxiter": 15},
    )

    return NelderMeadRBData(minimize_result, experiment)


def _fit(data: NelderMeadRBData) -> NelderMeadRBResults:
    df = get_aggregational_data(data.experiment)
    return NelderMeadRBResults(df)


def _plot(data: NelderMeadRBData, fit: NelderMeadRBResults, qubit):
    """Plotting function for NelderMeadRB."""
    return [build_report(data.experiment, fit.df)], " a | b | c "


neldermeadrb = Routine(_acquisition, _fit, _plot)
