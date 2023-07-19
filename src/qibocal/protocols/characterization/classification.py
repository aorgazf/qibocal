from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from sklearn.metrics import roc_auc_score, roc_curve

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.fitting.classifier import run
from qibocal.protocols.characterization.utils import get_color_state0, get_color_state1

MESH_SIZE = 50
MARGIN = 0
COLUMNWIDTH = 600
ROC_LENGHT = 800
ROC_WIDTH = 800
LEGEND_FONT_SIZE = 20
TITLE_SIZE = 25
SPACING = 0.1


@dataclass
class SingleShotClassificationParameters(Parameters):
    """SingleShotClassification runcard inputs."""

    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    classifiers_list: Optional[list[str]] = field(default_factory=lambda: ["qubit_fit"])
    """List of models to classify the qubit states"""
    savedir: Optional[str] = "classification_results"
    """Dumping folder of the classification results"""


ClassificationType = np.dtype([("i", np.float64), ("q", np.float64)])
"""Custom dtype for rabi amplitude."""


@dataclass
class SingleShotClassificationData(Data):
    nshots: int
    """Number of shots."""
    classifiers_list: Optional[str]
    """List of models to classify the qubit states"""
    hpars: dict[QubitId, dict] = field(metadata=dict(update="classifiers_hpars"))
    """Models' hyperparameters"""
    savedir: Optional[str] = "classification_results"
    """Dumping folder of the classification results"""
    data: dict[tuple[QubitId, int], npt.NDArray[ClassificationType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, qubit, state, i, q):
        """Store output for single qubit."""
        ar = np.empty(i.shape, dtype=ClassificationType)
        ar["i"] = i
        ar["q"] = q
        self.data[qubit, state] = np.rec.array(ar)


@dataclass
class SingleShotClassificationResults(Results):
    """SingleShotClassification outputs."""

    benchmark_table: dict
    y_tests: dict
    x_tests: dict
    models: dict
    names: list
    hpars: dict[QubitId, dict] = field(metadata=dict(update="classifiers_hpars"))

    threshold: dict[QubitId, float] = field(metadata=dict(update="threshold"))
    """Threshold for classification."""
    rotation_angle: dict[QubitId, float] = field(metadata=dict(update="iq_angle"))
    """Threshold for classification."""
    mean_gnd_states: dict[QubitId, list[float]] = field(
        metadata=dict(update="mean_gnd_states")
    )
    mean_exc_states: dict[QubitId, list[float]] = field(
        metadata=dict(update="mean_exc_states")
    )
    fidelity: dict[QubitId, float]
    assignment_fidelity: dict[QubitId, float]

    def save(self, path):
        pass


def _acquisition(
    params: SingleShotClassificationParameters,
    platform: Platform,
    qubits: Qubits,
) -> SingleShotClassificationData:
    """
    Method which implements the state's calibration of a chosen qubit. Two analogous tests are performed
    for calibrate the ground state and the excited state of the oscillator.
    The subscripts `exc` and `gnd` will represent the excited state |1> and the ground state |0>.

    Args:
        platform (:class:`qibolab.platforms.abstract.Platform`): custom abstract platform on which we perform the calibration.
        qubits (dict): dict of target Qubit objects to perform the action
        nshots (int): number of times the pulse sequence will be repeated.

    Returns:
        A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **iteration[dimensionless]**: Execution number
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

    """

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    hpars = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].finish
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])
        hpars[qubit] = qubits[qubit].classifiers_hpars
    # create a DataUnits object to store the results
    data = SingleShotClassificationData(
        params.nshots, params.classifiers_list, hpars, params.savedir
    )

    # execute the first pulse sequence
    state0_results = platform.execute_pulse_sequence(
        state0_sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        ),
    )

    # retrieve and store the results for every qubit
    for qubit in qubits:
        result = state0_results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit=qubit, state=0, i=result.voltage_i, q=result.voltage_q
        )

    # execute the second pulse sequence
    state1_results = platform.execute_pulse_sequence(
        state1_sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        ),
    )
    # retrieve and store the results for every qubit
    for qubit in qubits:
        result = state1_results[ro_pulses[qubit].serial]
        data.register_qubit(
            qubit=qubit, state=1, i=result.voltage_i, q=result.voltage_q
        )

    return data


def _fit(data: SingleShotClassificationData) -> SingleShotClassificationResults:
    qubits = data.qubits
    benchmark_tables = {}
    models_dict = {}
    y_tests = {}
    x_tests = {}
    hpars = {}
    threshold = {}
    rotation_angle = {}
    mean_gnd_states = {}
    mean_exc_states = {}
    fidelity = {}
    assignment_fidelity = {}
    for qubit in qubits:
        benchmark_table, y_test, x_test, models, names, hpars_list = run.train_qubit(
            data, qubit
        )
        benchmark_tables[qubit] = benchmark_table
        models_dict[qubit] = models
        y_tests[qubit] = y_test
        x_tests[qubit] = x_test
        hpars[qubit] = {}
        for i, model_name in enumerate(names):
            hpars[qubit][model_name] = hpars_list[i]
            if model_name == "qubit_fit":
                threshold[qubit] = models[i].threshold
                rotation_angle[qubit] = models[i].angle
                mean_gnd_states[qubit] = models[i].iq_mean0
                mean_exc_states[qubit] = models[i].iq_mean1
                fidelity[qubit] = models[i].fidelity
                assignment_fidelity[qubit] = models[i].assignment_fidelity

    return SingleShotClassificationResults(
        benchmark_tables,
        y_tests,
        x_tests,
        models_dict,
        names,
        hpars,
        threshold,
        rotation_angle,
        mean_gnd_states,
        mean_exc_states,
        fidelity,
        assignment_fidelity,
    )


def _plot(
    data: SingleShotClassificationData, fit: SingleShotClassificationResults, qubit
):
    figures = []
    fitting_report = ""

    models_name = fit.names
    state0_data = data.data[qubit, 0]
    state1_data = data.data[qubit, 1]

    max_x = (
        max(
            0,
            state0_data["i"].max(),
            state1_data["i"].max(),
        )
        + MARGIN
    )
    max_y = (
        max(
            0,
            state0_data["q"].max(),
            state1_data["q"].max(),
        )
        + MARGIN
    )
    min_x = (
        min(
            0,
            state0_data["i"].min(),
            state1_data["i"].min(),
        )
        - MARGIN
    )
    min_y = (
        min(
            0,
            state0_data["q"].min(),
            state1_data["q"].min(),
        )
        - MARGIN
    )
    i_values, q_values = np.meshgrid(
        np.linspace(min_x, max_x, num=MESH_SIZE),
        np.linspace(min_y, max_y, num=MESH_SIZE),
    )
    grid = np.vstack([i_values.ravel(), q_values.ravel()]).T

    accuracy = []
    training_time = []
    testing_time = []

    fig = make_subplots(
        rows=1,
        cols=len(models_name),
        horizontal_spacing=SPACING * 3 / len(models_name),
        vertical_spacing=SPACING,
        subplot_titles=(models_name),
        column_width=[COLUMNWIDTH] * len(models_name),
    )
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    fig_benchmarks = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=SPACING,
        vertical_spacing=SPACING,
        subplot_titles=("accuracy", "training time (s)", "testing time (s)"),
        # pylint: disable=E1101
        # column_width = [COLUMNWIDTH]*3
    )

    y_test = fit.y_tests[qubit]
    x_test = fit.x_tests[qubit]
    for i, model in enumerate(models_name):
        try:
            y_pred = fit.models[qubit][i].predict_proba(x_test)[:, 1]
        except AttributeError:
            y_pred = fit.models[qubit][i].predict(x_test)
        predictions = np.round(
            np.reshape(fit.models[qubit][i].predict(grid), q_values.shape)
        ).astype(np.int64)
        # Evaluate the ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)

        name = f"{model} (AUC={auc_score:.2f})"
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                name=name,
                mode="lines",
                marker=dict(size=3, color=get_color_state0(i)),
            )
        )

        max_x = max(grid[:, 0])
        max_y = max(grid[:, 1])
        min_x = min(grid[:, 0])
        min_y = min(grid[:, 1])

        fig_benchmarks.add_trace(
            go.Scatter(
                x=[model],
                y=[fit.benchmark_table[qubit]["accuracy"][i]],
                mode="markers",
                showlegend=False,
                # opacity=0.7,
                marker=dict(size=10, color=get_color_state1(i)),
            ),
            row=1,
            col=1,
        )

        fig_benchmarks.add_trace(
            go.Scatter(
                x=[model],
                y=[fit.benchmark_table[qubit]["training_time"][i]],
                mode="markers",
                showlegend=False,
                # opacity=0.7,
                marker=dict(size=10, color=get_color_state1(i)),
            ),
            row=1,
            col=2,
        )

        fig_benchmarks.add_trace(
            go.Scatter(
                x=[model],
                y=[fit.benchmark_table[qubit]["testing_time"][i]],
                mode="markers",
                showlegend=False,
                # opacity=0.7,
                marker=dict(size=10, color=get_color_state1(i)),
            ),
            row=1,
            col=3,
        )

        fig.add_trace(
            go.Scatter(
                x=state0_data["i"],
                y=state0_data["q"],
                name=f"q{qubit}/{model}: state 0",
                legendgroup=f"q{qubit}/{model}: state 0",
                mode="markers",
                showlegend=True,
                opacity=0.7,
                marker=dict(size=3, color=get_color_state0(i)),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=state1_data["i"],
                y=state1_data["q"],
                name=f"q{qubit}/{model}: state 1",
                legendgroup=f"q{qubit}/{model}: state 1",
                mode="markers",
                showlegend=True,
                opacity=0.7,
                marker=dict(size=3, color=get_color_state1(i)),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Contour(
                x=grid[:, 0],
                y=grid[:, 1],
                z=predictions.flatten(),
                showscale=False,
                # colorscale=["green", "red"],
                opacity=0.4,
                name="Score",
                hoverinfo="skip",
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=[np.average(state0_data["i"])],
                y=[np.average(state0_data["q"])],
                name=f"q{qubit}/{model}: state 0",
                legendgroup=f"q{qubit}/{model}: state 0",
                showlegend=False,
                mode="markers",
                marker=dict(size=10, color=get_color_state0(i)),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=[np.average(state1_data["i"])],
                y=[np.average(state1_data["q"])],
                name=f"q{qubit}/{model}: state 1",
                legendgroup=f"q{qubit}/{model}: state 1",
                showlegend=False,
                mode="markers",
                marker=dict(size=10, color=get_color_state1(i)),
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(
            title_text=f"i (V)",
            range=[min_x, max_x],
            row=1,
            col=i + 1,
            autorange=False,
            rangeslider=dict(visible=False),
        )
        fig.update_yaxes(
            title_text="q (V)",
            range=[min_y, max_y],
            scaleanchor="x",
            scaleratio=1,
            row=1,
            col=i + 1,
        )

        title_text = ""

        if models_name[i] == "qubit_fit":
            title_text += f"q{qubit}/{model} | average state 0: {fit.models[qubit][i].iq_mean0}<br>"
            title_text += f"q{qubit}/{model} | average state 1: {fit.models[qubit][i].iq_mean1}<br>"
            title_text += f"q{qubit}/{model} | rotation angle: {fit.models[qubit][i].angle:.3f}<br>"
            title_text += f"q{qubit}/{model} | threshold: {fit.models[qubit][i].threshold:.6f}<br>"
            title_text += (
                f"q{qubit}/{model} | fidelity: {fit.models[qubit][i].fidelity:.3f}<br>"
            )
            title_text += f"q{qubit}/{model} | assignment fidelity: {fit.models[qubit][i].assignment_fidelity:.3f}<br>"

        fitting_report += title_text

        fig.update_layout(
            # showlegend=False,
            uirevision="0",  # ``uirevision`` allows zooming while live plotting
            autosize=False,
            height=COLUMNWIDTH,
            width=COLUMNWIDTH * len(models_name),
            title=dict(text="Results", font=dict(size=TITLE_SIZE)),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                xanchor="left",
                y=-0.3,
                x=0,
                itemsizing="constant",
                font=dict(size=LEGEND_FONT_SIZE),
            ),
        )
        fig_benchmarks.update_yaxes(type="log", row=1, col=2)
        fig_benchmarks.update_yaxes(type="log", row=1, col=3)
        fig_benchmarks.update_layout(
            autosize=False,
            height=COLUMNWIDTH,
            width=COLUMNWIDTH * 3,
            title=dict(text="Benchmarks", font=dict(size=TITLE_SIZE)),
        )
        fig_roc.update_layout(
            width=ROC_WIDTH,
            height=ROC_LENGHT,
            title=dict(text="ROC curves", font=dict(size=TITLE_SIZE)),
            legend=dict(font=dict(size=LEGEND_FONT_SIZE)),
        )

    figures.append(fig_roc)
    figures.append(fig)
    figures.append(fig_benchmarks)
    return figures, fitting_report


single_shot_classification = Routine(_acquisition, _fit, _plot)
