from dataclasses import dataclass, field

from pandas import DataFrame
from qibo.noise import NoiseModel
from qibolab.platforms.abstract import AbstractPlatform

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.calibrations.niGSC.basics.plot import plot_qq
from qibocal.calibrations.niGSC.XIdrb import (
    ModuleExperiment,
    ModuleFactory,
    build_report,
    get_aggregational_data,
    post_processing_sequential,
)


@dataclass
class XIdRBParameters(Parameters):
    """XId Randomized Benchmarking runcard inputs."""

    nqubits: int
    qubits: list
    depths: list
    runs: int
    nshots: int
    noise_model: NoiseModel = field(default_factory=NoiseModel)
    noise_params: list = field(default_factory=list)


@dataclass
class XIdRBResults(Results):
    """XId RB outputs."""

    df: DataFrame

    def save(self, path):
        self.df.to_pickle(path)


class XIdRBData:
    """XId RB data acquisition."""

    def __init__(self, experiment: ModuleExperiment):
        self.experiment = experiment

    def save(self, path):
        self.experiment.save(path)

    def to_csv(self, path):
        self.save(path)

    def load(self, path):
        self.experiment.load(path)


def _acquisition(
    params: XIdRBParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> XIdRBData:
    factory = ModuleFactory(
        params.nqubits, params.depths * params.runs, qubits=params.qubits
    )
    experiment = ModuleExperiment(
        factory, nshots=params.nshots, noise_model=params.noise_model
    )
    experiment.perform(experiment.execute)
    post_processing_sequential(experiment)
    return XIdRBData(experiment)


def _fit(data: XIdRBData) -> XIdRBResults:
    df = get_aggregational_data(data.experiment)
    return XIdRBResults(df)


def _plot(data: XIdRBData, fit: XIdRBResults, qubit):
    """Plotting function for XIdRB."""
    return build_report(data.experiment, fit.df)


xid_rb = Routine(_acquisition, _fit, _plot)
