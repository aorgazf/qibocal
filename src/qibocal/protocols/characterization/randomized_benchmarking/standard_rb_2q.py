from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, TypedDict, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibo.backends import GlobalBackend
from qibolab.platform import Platform
from qibolab.qubits import QubitPairId

from qibocal.auto.operation import Results, Routine
from qibocal.auto.transpile import (
    dummy_transpiler,
    execute_transpiled_circuit,
    execute_transpiled_circuits,
)
from qibocal.config import raise_error
from qibocal.protocols.characterization.randomized_benchmarking import noisemodels
from qibocal.protocols.characterization.randomized_benchmarking.standard_rb import (
    RBData,
    RBType,
    StandardRBParameters,
    StandardRBResult,
)

from ..utils import table_dict, table_html
from .circuit_tools import add_inverse_2q_layer, add_measurement_layer, layer_2q_circuit
from .fitting import exp1B_func, fit_exp1B_func
from .utils import data_uncertainties, number_to_str, random_2q_clifford

NPULSES_PER_CLIFFORD = 3.5  # CHANGE


class Depthsdict(TypedDict):
    """dictionary used to build a list of depths as ``range(start, stop, step)``."""

    start: int
    stop: int
    step: int


@dataclass
class RB2QData(RBData):
    """The output of the acquisition function."""

    data: dict[QubitPairId, npt.NDArray[RBType]] = field(default_factory=dict)
    """Raw data acquired."""
    circuits: dict[QubitPairId, list[list[int]]] = field(default_factory=dict)
    """Clifford gate indexes executed."""

    def extract_probabilities(self, qubits):
        """Extract the probabilities given (`qubit`, `qubit`)"""
        probs = []
        for depth in self.depths:
            data_list = np.array(self.data[qubits[0], qubits[1], depth].tolist())
            data_list = data_list.reshape((-1, self.nshots))
            probs.append(np.count_nonzero(1 - data_list, axis=1) / data_list.shape[1])
        return probs


@dataclass
class StandardRB2QResult(Results):
    """Standard RB outputs."""

    fidelity: dict[QubitPairId, float]
    """The overall fidelity of this qubit."""
    pulse_fidelity: dict[QubitPairId, float]
    """The pulse fidelity of the gates acting on this qubit."""
    fit_parameters: dict[QubitPairId, tuple[float, float, float]]
    """Raw fitting parameters."""
    fit_uncertainties: dict[QubitPairId, tuple[float, float, float]]
    """Fitting parameters uncertainties."""
    error_bars: dict[QubitPairId, Optional[Union[float, list[float]]]] = None
    """Error bars for y."""

    # FIXME: fix this after https://github.com/qiboteam/qibocal/pull/597
    def __contains__(self, qubits: QubitPairId):
        return True


class RB2Q_Generator:
    """
    This class generates random two qubit cliffords for randomized benchmarking.
    """

    def __init__(self, seed):
        self.seed = seed
        self.local_state = (
            np.random.default_rng(seed)
            if seed is None or isinstance(seed, int)
            else seed
        )

    def random_index(self, gate_dict):
        """
        Generates a random index within the range of the given file len.

        Parameters:
        - file (Dict): Dict of gates.

        Returns:
        - int: Random index.
        """
        return self.local_state.integers(0, len(gate_dict.keys()), 1)

    def layer_gen(self):
        """
        Returns:
        - Gate: Random single-qubit clifford .
        """
        return random_2q_clifford(self.random_index)


def random_circuits(
    depth: int,
    targets: list[QubitPairId],
    niter,
    rb_gen,
    noise_model=None,
) -> Iterable:
    """Returns single-qubit random self-inverting Clifford circuits.

    Args:
        params (StandardRBParameters): Parameters of the RB protocol.
        targets (list[QubitId]):
            list of qubits the circuit is executed on.
        nqubits (int, optional): Number of qubits of the resulting circuits.
            If ``None``, sets ``len(qubits)``. Defaults to ``None``.

    Returns:
        Iterable: The iterator of circuits.
    """

    circuits = []
    indexes = defaultdict(list)
    for _ in range(niter):
        for target in targets:
            circuit, random_index = layer_2q_circuit(rb_gen, depth, target)
            add_inverse_2q_layer(circuit)
            add_measurement_layer(circuit)
            if noise_model is not None:
                circuit = noise_model.apply(circuit)
            circuits.append(circuit)
            indexes[target].append(random_index)

    return circuits, indexes


def _acquisition(
    params: StandardRBParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> RB2QData:
    """The data acquisition stage of Standard Randomized Benchmarking.

    1. Set up the scan
    2. Execute the scan
    3. Post process the data and initialize a standard rb data object with it.

    Args:
        params (StandardRBParameters): All parameters in one object.
        platform (Platform): Platform the experiment is executed on.
        qubits (dict[int, Union[str, int]] or list[Union[str, int]]): list of qubits the experiment is executed on.

    Returns:
        RBData: The depths, samples and ground state probability of each experiment in the scan.
    """

    backend = GlobalBackend()
    # For simulations, a noise model can be added.
    noise_model = None
    if params.noise_model is not None:
        if backend.name == "qibolab":
            raise_error(
                ValueError,
                "Backend qibolab (%s) does not perform noise models simulation. ",
            )

        noise_model = getattr(noisemodels, params.noise_model)(params.noise_params)
        params.noise_params = noise_model.params.tolist()
    # 1. Set up the scan (here an iterator of circuits of random clifford gates with an inverse).
    nqubits = len(targets)
    data = RB2QData(
        depths=params.depths,
        uncertainties=params.uncertainties,
        seed=params.seed,
        nshots=params.nshots,
        niter=params.niter,
    )

    circuits = []
    indexes = {}
    samples = []
    qubits_ids = targets
    rb_gen = RB2Q_Generator(params.seed)
    for depth in params.depths:
        # TODO: This does not generate multi qubit circuits
        circuits_depth, random_indexes = random_circuits(
            depth, qubits_ids, params.niter, rb_gen, noise_model
        )
        circuits.extend(circuits_depth)
        for qubit in random_indexes.keys():
            indexes[(qubit[0], qubit[1], depth)] = random_indexes[qubit]

    backend = GlobalBackend()
    transpiler = dummy_transpiler(backend)
    qubit_maps = [list(i) for i in targets] * (len(params.depths) * params.niter)

    # Execute the circuits
    if params.unrolling:
        _, executed_circuits = execute_transpiled_circuits(
            circuits,
            qubit_maps=qubit_maps,
            backend=backend,
            nshots=params.nshots,
            transpiler=transpiler,
        )
    else:
        executed_circuits = [
            execute_transpiled_circuit(
                circuit,
                qubit_map=qubit_map,
                backend=backend,
                nshots=params.nshots,
                transpiler=transpiler,
            )[1]
            for circuit, qubit_map in zip(circuits, qubit_maps)
        ]

    for circ in executed_circuits:
        samples.extend(circ.samples())
    samples = np.reshape(samples, (-1, nqubits, params.nshots))

    for i, depth in enumerate(params.depths):
        index = (i * params.niter, (i + 1) * params.niter)
        for nqubit, qubit_id in enumerate(targets):

            data.register_qubit(
                RBType,
                (qubit_id[0], qubit_id[1], depth),
                dict(
                    samples=samples[index[0] : index[1]][:, nqubit],
                ),
            )
    data.circuits = indexes

    return data


def _fit(data: RB2QData) -> StandardRB2QResult:
    """Takes a data frame, extracts the depths and the signal and fits it with an
    exponential function y = Ap^x+B.

    Args:
        data (RBData): Data from the data acquisition stage.

    Returns:
        StandardRBResult: Aggregated and processed data.
    """
    qubits = data.pairs
    fidelity, pulse_fidelity = {}, {}
    popts, perrs = {}, {}
    error_barss = {}
    for qubit in qubits:
        # Extract depths and probabilities
        x = data.depths
        probs = data.extract_probabilities(qubit)
        samples_mean = np.mean(probs, axis=1)
        # TODO: Should we use the median or the mean?
        median = np.median(probs, axis=1)

        error_bars = data_uncertainties(
            probs,
            method=data.uncertainties,
            data_median=median,
        )

        sigma = (
            np.max(error_bars, axis=0) if data.uncertainties is not None else error_bars
        )

        popt, perr = fit_exp1B_func(x, samples_mean, sigma=sigma, bounds=[0, 1])
        # Compute the fidelities
        infidelity = (1 - popt[1]) / 2
        fidelity[qubit] = 1 - infidelity
        pulse_fidelity[qubit] = 1 - infidelity / NPULSES_PER_CLIFFORD

        # conversion from np.array to list/tuple
        error_bars = error_bars.tolist()
        error_barss[qubit] = error_bars
        perrs[qubit] = perr
        popts[qubit] = popt

    return StandardRBResult(fidelity, pulse_fidelity, popts, perrs, error_barss)


def _plot(
    data: RB2QData, fit: StandardRB2QResult, target: QubitPairId
) -> tuple[list[go.Figure], str]:
    """Builds the table for the qq pipe, calls the plot function of the result object
    and returns the figure es list.

    Args:
        data (RBData): Data object used for the table.
        fit (StandardRBResult): Is called for the plot.
        target (_type_): Not used yet.

    Returns:
        tuple[list[go.Figure], str]:
    """

    qubits = target
    fig = go.Figure()
    fitting_report = ""
    x = data.depths
    raw_data = data.extract_probabilities(qubits)
    y = np.mean(raw_data, axis=1)
    raw_depths = [[depth] * data.niter for depth in data.depths]

    fig.add_trace(
        go.Scatter(
            x=np.hstack(raw_depths),
            y=np.hstack(raw_data),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="iterations",
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
    # Create a dictionary for the error bars
    error_y_dict = None
    if fit is not None:
        popt, perr = fit.fit_parameters[qubits], fit.fit_uncertainties[qubits]
        label = "Fit: y=Ap^x<br>A: {}<br>p: {}<br>B: {}".format(
            number_to_str(popt[0], perr[0]),
            number_to_str(popt[1], perr[1]),
            number_to_str(popt[2], perr[2]),
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
        if fit.error_bars is not None:
            error_bars = fit.error_bars[qubits]
            # Constant error bars
            if isinstance(error_bars, Iterable) is False:
                error_y_dict = {"type": "constant", "value": error_bars}
            # Symmetric error bars
            elif isinstance(error_bars[0], Iterable) is False:
                error_y_dict = {"type": "data", "array": error_bars}
            # Asymmetric error bars
            else:
                error_y_dict = {
                    "type": "data",
                    "symmetric": False,
                    "array": error_bars[1],
                    "arrayminus": error_bars[0],
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
    if fit is not None:
        fitting_report = table_html(
            table_dict(
                str(qubits),
                ["niter", "nshots", "uncertainties", "fidelity", "pulse_fidelity"],
                [
                    data.niter,
                    data.nshots,
                    data.uncertainties,
                    number_to_str(
                        fit.fidelity[qubits],
                        np.array(fit.fit_uncertainties[qubits][1]) / 2,
                    ),
                    number_to_str(
                        fit.pulse_fidelity[qubits],
                        np.array(fit.fit_uncertainties[qubits][1])
                        / (2 * NPULSES_PER_CLIFFORD),
                    ),
                ],
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Circuit depth",
        yaxis_title="Survival Probability",
    )

    return [fig], fitting_report


standard_rb_2q = Routine(_acquisition, _fit, _plot)
