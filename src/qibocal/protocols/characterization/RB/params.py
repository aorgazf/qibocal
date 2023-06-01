from dataclasses import dataclass, field
from typing import Union

from qibocal.auto.operation import Parameters


@dataclass
class RBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    nqubits: int
    """The amount of qubits on the chip """
    qubits: list
    """A list of indices which qubit(s) should be benchmarked """
    depths: Union[list, dict]
    """A list of depths/sequence lengths. If a dictionary is given the list will be build."""
    niter: int
    """Sets how many iterations over the same depth value."""
    nshots: int
    """For each sequence how many shots for statistics should be performed."""
    n_bootstrap: int = 1
    """Number of bootstrap iterations for the fit uncertainties. Defaults to 1."""
    noise_model: str = ""
    """For simulation purposes, string has to match what is in qibocal. ... basics.noisemodels"""
    noise_params: list = field(default_factory=list)
    """With this the noise model will be initialized, if not given random values will be used."""

    def __post_init__(self):
        if isinstance(self.depths, dict):
            self.depths = list(
                range(self.depths["start"], self.depths["stop"], self.depths["step"])
            )
