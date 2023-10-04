from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Qubits, Routine
from qibocal.fitting.classifier import run
from qibocal.protocols.characterization.classification import (
    SingleShotClassificationData,
    SingleShotClassificationParameters,
    SingleShotClassificationResults,
)
from qibocal.protocols.characterization.utils import (
    MESH_SIZE,
    evaluate_grid,
    plot_results,
)

COLUMNWIDTH = 600
LEGEND_FONT_SIZE = 20
TITLE_SIZE = 25
SPACING = 0.1
DEFAULT_CLASSIFIER = "naive_bayes"


@dataclass
class QutritClassificationParameters(SingleShotClassificationParameters):
    """SingleShotClassification runcard inputs."""

    classifiers_list: Optional[list[str]] = field(
        default_factory=lambda: [DEFAULT_CLASSIFIER]
    )
    """List of models to classify the qubit states"""


@dataclass
class QutritClassificationData(SingleShotClassificationData):
    classifiers_list: Optional[list[str]] = field(
        default_factory=lambda: [DEFAULT_CLASSIFIER]
    )
    """List of models to classify the qubit states"""


def _acquisition(
    params: QutritClassificationParameters,
    platform: Platform,
    qubits: Qubits,
) -> QutritClassificationData:
    """
    Args:
        nshots (int): number of times the pulse sequence will be repeated.
        classifiers (list): list of classifiers, the available ones are:
            - linear_svm
            - ada_boost
            - gaussian_process
            - naive_bayes
            - nn
            - qubit_fit
            - random_forest
            - rbf_svm
            - qblox_fit.
        The default value is `["qubit_fit"]`.
        savedir (str): Dumping folder of the classification results.
        If not given the dumping folder will be the report one.
        relaxation_time (float): Relaxation time.

        Example:
        .. code-block:: yaml

            - id: single_shot_classification_1
                priority: 0
                operation: single_shot_classification
                parameters:
                nshots: 5000
                savedir: "single_shot"
                classifiers_list: ["qubit_fit","naive_bayes", "linear_svm"]

    """

    # create two sequences of pulses:
    # state0_sequence: I  - MZ
    # state1_sequence: RX - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    states_sequences = [PulseSequence() for _ in range(3)]
    ro_pulses = {}
    hpars = {}
    for qubit in qubits:
        rx_pulse = platform.create_RX_pulse(qubit, start=0)
        rx12_pulse = platform.create_RX12_pulse(qubit, start=rx_pulse.finish)
        drive_pulses = [rx_pulse, rx12_pulse]
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=rx12_pulse.finish
        )
        hpars[qubit] = qubits[qubit].classifiers_hpars
        for i, sequence in enumerate(states_sequences):
            sequence.add(*drive_pulses[:i])
            sequence.add(ro_pulses[qubit])
    # create a DataUnits object to store the results
    data = QutritClassificationData(
        nshots=params.nshots,
        classifiers_list=params.classifiers_list,
        classifiers_hpars=hpars,
        savedir=params.savedir,
    )
    states_results = []
    for sequence in states_sequences:
        states_results.append(
            platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                ),
            )
        )

    for qubit in qubits:
        for state, state_result in enumerate(states_results):
            result = state_result[ro_pulses[qubit].serial]
            data.register_qubit(
                qubit=qubit, state=state, i=result.voltage_i, q=result.voltage_q
            )

    return data


def _fit(data: QutritClassificationData) -> SingleShotClassificationResults:
    qubits = data.qubits

    benchmark_tables = {}
    models_dict = {}
    y_tests = {}
    x_tests = {}
    hpars = {}
    y_test_predict = {}
    grid_preds_dict = {}
    for qubit in qubits:
        qubit_data = data.data[qubit]
        benchmark_table, y_test, x_test, models, names, hpars_list = run.train_qubit(
            data, qubit
        )
        benchmark_tables[qubit] = benchmark_table.values.tolist()
        models_dict[qubit] = models
        y_tests[qubit] = y_test.tolist()
        x_tests[qubit] = x_test.tolist()
        hpars[qubit] = {}
        y_preds = []
        grid_preds = []

        grid = evaluate_grid(qubit_data)
        for i, model_name in enumerate(names):
            hpars[qubit][model_name] = hpars_list[i]
            try:
                y_preds.append(models[i].predict_proba(x_test)[:, 1].tolist())
            except AttributeError:
                y_preds.append(models[i].predict(x_test).tolist())
            grid_preds.append(
                np.round(np.reshape(models[i].predict(grid), (MESH_SIZE, MESH_SIZE)))
                .astype(np.int64)
                .tolist()
            )
        y_test_predict[qubit] = y_preds
        grid_preds_dict[qubit] = grid_preds
    return SingleShotClassificationResults(
        benchmark_table=benchmark_tables,
        names=names,
        classifiers_hpars=hpars,
        models=models_dict,
        savedir=data.savedir,
        y_preds=y_test_predict,
        grid_preds=grid_preds_dict,
    )


def _plot(data: QutritClassificationData, qubit, fit: SingleShotClassificationResults):
    figures = plot_results(data, qubit, 3, fit)
    fitting_report = None
    return figures, fitting_report


def _update(
    results: SingleShotClassificationResults, platform: Platform, qubit: QubitId
):
    update.classifiers_hpars(
        results.classifiers_hpars[qubit], platform, qubit
    )  # TODO: implement a qutrit classifiers hpars (?)


qutrit_classification = Routine(_acquisition, _fit, _plot, _update)
