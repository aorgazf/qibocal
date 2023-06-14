from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qibo
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.bootstrap import bootstrap, data_errors
from qibocal.config import log, raise_error
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels

from .circuit_tools import (
    add_inverse_layer,
    add_measurement_layer,
    embed_circuit,
    layer_circuit,
)
from .fitting import exp1B_func, fit_exp1B_func
from .utils import extract_from_data, number_to_str, random_clifford

NPULSES_PER_CLIFFORD = 1.875


class DepthsDict(TypedDict):
    """Dictionary used to build a list of depths as ``range(start, stop, step)``."""

    start: int
    stop: int
    step: int


@dataclass
class StandardRBParameters(Parameters):
    """Standard Randomized Benchmarking runcard inputs."""

    depths: Union[list, DepthsDict]
    """A list of depths/sequence lengths. If a dictionary is given the list will be build."""
    niter: int
    """Sets how many iterations over the same depth value."""
    nshots: int
    """For each sequence how many shots for statistics should be performed."""
    uncertainties: Union[str, float] = 95
    """Method of computing the error bars and uncertainties of the data. If ``None``, does not
    compute the errors. If ``"std"``, computes the standard deviation. If ``float`` or ``int``
    between 0 and 100, computes the corresponding confidence interval. Defaults to ``95``."""
    n_bootstrap: int = 100
    """Number of bootstrap iterations for the fit uncertainties and error bars.
    If ``0``, gets the fit uncertainties from the fitting function and the error bars
    from the distribution of the measurements. Defaults to ``100``."""
    seed: Optional[int] = None
    """A fixed seed to initialize ``np.random.Generator``. If ``None``, uses a random seed.
    Defaults is ``None``."""
    noise_model: str = ""
    """For simulation purposes, string has to match what is in
    :mod:`qibocal.protocols.characterization.randomized_benchmarking.noisemodels`"""
    noise_params: Optional[list] = field(default_factory=list)
    """With this the noise model will be initialized, if not given random values will be used."""

    def __post_init__(self):
        if isinstance(self.depths, dict):
            self.depths = list(
                range(self.depths["start"], self.depths["stop"], self.depths["step"])
            )


class RBData(pd.DataFrame):
    """A pandas DataFrame child. The output of the acquisition function."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def to_csv(self, path):
        """Overwrite because qibocal action builder calls this function with a directory."""
        super().to_json(f"{path}/{self.__class__.__name__}.json", default_handler=str)


@dataclass
class StandardRBResult(Results):
    """Standard RB outputs."""

    fidelity: float
    """The overall fidelity of this qubit."""
    pulse_fidelity: float
    """The pulse fidelity of the gates acting on this qubit."""
    fitting_parameters: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    """Raw fitting parameters."""
    error_y: Optional[Union[float, List[float], np.ndarray]] = None
    """Error bars for y."""


def samples_to_p0(samples_list):
    """Computes the probabilitiy of 0 from the list of samples.

    Args:
        samples_list (list or np.ndarray): 3d array with rows corresponding to circuits
            containing ``nshots`` number of lists with ``nqubits`` amount of ``0`` and ``1``.
            e.g. ``samples_list`` for 1 circuit, 3 shots and 2 qubits looks like
            ``[[[0, 0], [0, 1], [1, 0]]]`` and ``p0=1/3``.

    Returns:
        list: list of probabilities corresponding to each row.
    """

    ground = np.array([0] * len(samples_list[0][0]))
    p0_list = []
    for samples in samples_list:
        p0_list.append(np.sum(np.product(samples == ground, axis=1)) / len(samples))
    return p0_list


def resample_p0(data, sample_size=100, homogeneous: bool = True):
    """Preforms parametric resampling of shots with binomial distribution
        and returns a list of "corrected" probabilites.

    Args:
        data (list or np.ndarray): list of probabilities for the binomial distribution.
        nshots (int): sample size for one probability distribution.

    Returns:
        list: resampled probabilities.
    """
    if homogeneous:
        return np.apply_along_axis(
            lambda p: samples_to_p0(
                np.random.binomial(n=1, p=1 - p, size=(1, sample_size, len(p))).T
            ),
            0,
            data,
        )
    resampled_data = []
    for row in data:
        resampled_data.append([])
        for p in row:
            samples_corrected = np.random.binomial(
                n=1, p=1 - p, size=(1, sample_size, *p.shape)
            ).T
            resampled_data[-1].append(samples_to_p0(samples_corrected))
    return resampled_data


def setup_scan(
    params: StandardRBParameters, qubits: Union[Qubits, List[QubitId]], nqubits: int
) -> Iterable:
    """Returns an iterator of single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.
        qubits (Dict[int, Union[str, int]] or List[Union[str, int]]):
            List of qubits the circuit is executed on.
        nqubits (int, optional): Number of qubits of the resulting circuits.
            If ``None``, sets ``len(qubits)``. Defaults to ``None``.

    Returns:
        Iterable: The iterator of circuits.
    """

    qubit_ids = list(qubits) if isinstance(qubits, dict) else qubits

    def make_circuit(depth):
        """Returns a random Clifford circuit with inverse of ``depth``."""

        # This function is needed so that the inside of the layer_circuit function layer_gen()
        # can be called for each layer of the circuit, and it returns a random layer of
        # Clifford gates. Could also be a generator, it just has to be callable.
        def layer_gen():
            """Returns a circuit with a random single-qubit clifford unitary."""
            return random_clifford(len(qubit_ids), params.seed)

        circuit = layer_circuit(layer_gen, depth)
        add_inverse_layer(circuit)
        add_measurement_layer(circuit)
        return embed_circuit(circuit, nqubits, qubit_ids)

    return map(make_circuit, params.depths * params.niter)


def _acquisition(
    params: StandardRBParameters,
    platform: Platform,
    qubits: Union[Qubits, List[QubitId]],
) -> RBData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params (StandardRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits (Dict[int, Union[str, int]] or List[Union[str, int]]):
            List of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    # For simulations, a noise model can be added.
    noise_model = None
    if params.noise_model:
        # FIXME implement this check outside acquisition
        if platform and platform.name != "dummy":
            raise_error(
                NotImplementedError,
                f"Backend qibolab ({platform}) does not perform noise models simulation.",
            )
        elif platform:
            log.warning(
                (
                    "Backend qibolab (%s) does not perform noise models simulation. "
                    "Setting backend to ``NumpyBackend`` instead."
                ),
                platform.name,
            )
            qibo.set_backend("numpy")
            platform = None

        noise_model = getattr(noisemodels, params.noise_model)(params.noise_params)
        params.noise_params = noise_model.params

    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    nqubits = platform.nqubits if platform else max(qubits) + 1
    scan = setup_scan(params, qubits, nqubits)

    # 2. Execute the scan.
    data_list = []
    # Iterate through the scan and execute each circuit.
    for circuit in scan:
        # The inverse and measurement gate don't count for the depth.
        depth = (circuit.depth - 2) if circuit.depth > 1 else 0
        if noise_model is not None:
            circuit = noise_model.apply(circuit)
        samples = circuit.execute(nshots=params.nshots).samples()
        # Every executed circuit gets a row where the data is stored.
        data_list.append({"depth": depth, "samples": samples})
    # Build the data object which will be returned and later saved.
    data = pd.DataFrame(data_list)

    # The signal here is the survival probability.
    standardrb_data = RBData(
        data.assign(signal=lambda x: samples_to_p0(x.samples.to_list()))
    )
    # Store the parameters to display them later.
    standardrb_data.attrs = params.__dict__
    return standardrb_data


def _fit(data: RBData) -> StandardRBResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """
    # Extract depths and probabilities
    x, y_scatter = extract_from_data(data, "signal", "depth", list)
    homogeneous = all(len(y_scatter[0]) == len(row) for row in y_scatter)

    # Extract fitting and bootstrap parameters if given
    uncertainties = data.attrs.get("uncertainties", None)
    n_bootstrap = data.attrs.get("n_bootstrap", 0)

    y_estimates, popt_estimates = y_scatter, []
    if n_bootstrap:
        # Non-parametric bootstrap resampling
        bootstrap_y = bootstrap(
            y_scatter,
            n_bootstrap,
            homogeneous=homogeneous,
            seed=data.attrs.get("seed", None),
        )

        # Parametric bootstrap resampling of of "corrected" probabilites from binomial distribution
        bootstrap_y = resample_p0(
            bootstrap_y, data.attrs.get("nshots", 1), homogeneous=homogeneous
        )

        # Compute y and popt estimates for each bootstrap iteration
        y_estimates = (
            np.mean(bootstrap_y, axis=1)
            if homogeneous
            else [np.mean(y_iter, axis=0) for y_iter in bootstrap_y]
        )
        popt_estimates = np.apply_along_axis(
            lambda y_iter: fit_exp1B_func(x, y_iter)[0],
            axis=0,
            arr=np.array(y_estimates),
        )

    # Fit the initial data and compute error bars
    y = [np.mean(y_row) for y_row in y_scatter]
    error_y = data_errors(y_estimates, uncertainties, symmetric=False, data_median=y)
    sigma = (
        np.max(error_y, axis=0)
        if error_y is not None and len(error_y.shape) == 2
        else error_y
    )
    popt, perr = fit_exp1B_func(x, y, sigma=sigma)

    # Compute fitting errors
    if len(popt_estimates):
        perr = data_errors(popt_estimates, uncertainties, data_median=popt)
        perr = perr.T if perr is not None else (0,) * len(popt)

    # Compute the fidelities
    infidelity = (1 - popt[1]) / 2
    fidelity = 1 - infidelity
    pulse_fidelity = 1 - infidelity / NPULSES_PER_CLIFFORD
    return StandardRBResult(fidelity, pulse_fidelity, (popt, perr), error_y)


def _plot(data: RBData, result: StandardRBResult, qubit) -> Tuple[List[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        result (StandardRBResult): Is called for the plot.
        qubit (_type_): Not used yet.

    Returns:
        Tuple[List[go.Figure], str]:
    """

    x, y_scatter = extract_from_data(data, "signal", "depth", list)
    y = [np.mean(y_row) for y_row in y_scatter]
    popt, perr = result.fitting_parameters
    label = "Fit: y=Ap^x<br>A: {}<br>p: {}<br>B: {}".format(
        number_to_str(popt[0], perr[0]),
        number_to_str(popt[1], perr[1]),
        number_to_str(popt[2], perr[2]),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.depth.tolist(),
            y=data.signal.tolist(),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="itertarions",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    # If result.error_y is given, create a dictionary for the error bars
    error_y_dict = None
    if result.error_y is not None:
        # Constant error bars
        if isinstance(result.error_y, Iterable) is False:
            error_y_dict = {"type": "constant", "value": result.error_y}
        # Symmetric error bars
        elif isinstance(result.error_y[0], Iterable) is False:
            error_y_dict = {"type": "data", "array": result.error_y}
        # Asymmetric error bars
        else:
            error_y_dict = {
                "type": "data",
                "symmetric": False,
                "array": result.error_y[1],
                "arrayminus": result.error_y[0],
            }
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                error_y=error_y_dict,
                line={"color": "#aa6464"},
                mode="markers",
                name="error bars",
            )
        )
    x_fit = np.linspace(min(x), max(x), len(x) * 20)
    y_fit = exp1B_func(x_fit, *popt)
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name=label,
            line=go.scatter.Line(dash="dot", color="#00cc96"),
        )
    )

    meta_data = deepcopy(data.attrs)
    meta_data.pop("depths")
    if not meta_data["noise_model"]:
        meta_data.pop("noise_model")
        meta_data.pop("noise_params")

    table_str = "".join(
        [
            f" | {key}: {value}<br>"
            for key, value in {
                **meta_data,
                "fidelity": number_to_str(result.fidelity, perr[1] / 2),
                "pulse_fidelity": number_to_str(
                    result.pulse_fidelity, perr[1] / (2 * NPULSES_PER_CLIFFORD)
                ),
            }.items()
        ]
    )
    return [fig], table_str


# Build the routine object which is used by qq-auto.
standard_rb = Routine(_acquisition, _fit, _plot)
