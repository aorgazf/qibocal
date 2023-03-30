from dataclasses import dataclass
from math import atan2

import numpy as np
import numpy.typing as npt

from .utils import identity


def constructor(_hyperparams):
    r"""Return the model class.

    Args:
        _hyperparams: Model hyperparameters.
    """
    return QubitFit()


def hyperopt(_x_train, _y_train, _path):
    r"""Perform an hyperparameter optimization and return the hyperparameters.

    Args:
        x_train: Training inputs.
        y_train: Training outputs.
        path (path): Model save path.

    Returns:
        Dictionary with model's hyperparameters.
    """
    return {}


normalize = identity


@dataclass
class QubitFit:
    r"""This class deploys a qubit state classifier.

    Args:
        iq_mean0 (np.ndarray): Center of the ground state cloud in the I-Q plane.
        iq_mean1 (np.ndarray): Center of the excited state cloud in the I-Q plane.
        threshold (float): Classifier's threshold.
        angle (float): Rotational angle.
    """  # TODO: add references

    iq_mean0: np.ndarray = np.array([0.0, 0.0])
    iq_mean1: np.ndarray = np.array([0.0, 0.0])
    threshold: float = 0.0
    angle: float = 0.0
    fidelity: float = None
    assignment_fidelity: float = None

    def fit(self, iq_coordinates, states: list):
        nshots = len(iq_coordinates)
        iq_state1 = iq_coordinates[(states == 1)]
        iq_state0 = iq_coordinates[(states == 0)]
        self.iq_mean0 = np.mean(iq_state0, axis=0)
        self.iq_mean1 = np.mean(iq_state1, axis=0)

        # translate
        iq_coordinates_translated = self.translate(iq_coordinates)
        iq_state1_trans = self.translate(self.iq_mean1)
        self.angle = -1 * atan2(iq_state1_trans[1], iq_state1_trans[0])

        # rotate
        iq_coord_rot = self.rotate(iq_coordinates_translated)

        x_values_state0 = iq_coord_rot[(states == 0)][:, 0]
        x_values_state1 = iq_coord_rot[(states == 1)][:, 0]

        x_values_state0 = np.sort(x_values_state0)
        x_values_state1 = np.sort(x_values_state1)

        # evaluate threshold and fidelity
        x_values = iq_coord_rot[:, 0]
        x_values.sort()
        cum_distribution_state0 = _eval_cumulative(x_values, x_values_state0)
        cum_distribution_state1 = _eval_cumulative(x_values, x_values_state1)

        cum_distribution_diff = np.abs(
            np.array(cum_distribution_state1) - np.array(cum_distribution_state0)
        )

        max_index = np.argmax(cum_distribution_diff)
        self.threshold = x_values[max_index]
        errors_state1 = nshots - cum_distribution_state1[max_index]
        errors_state0 = cum_distribution_state0[max_index]
        self.fidelity = cum_distribution_diff[max_index] / nshots
        self.assignment_fidelity = 1 - (errors_state1 + errors_state0) / nshots / 2

    def rotate(self, v):
        c, s = np.cos(self.angle), np.sin(self.angle)
        rot = np.array([[c, -s], [s, c]])
        return v @ rot.T

    def translate(self, v):
        return v - self.iq_mean0

    def predict(self, inputs: npt.NDArray):
        r"""Classify the `inputs`.

        Returns:
            List of predictions.
        """
        translated = self.translate(inputs)
        rotated = self.rotate(translated)
        return (rotated[:, 0] > self.threshold).astype(int)


def _check(value, sorted_list):
    for i, val in enumerate(sorted_list):
        if value < val:
            if i == 0:
                return 0
            else:
                return i - 1
    return len(sorted_list) - 1


def _eval_cumulative(data, points):
    # data and points sorted
    prob = np.zeros(len(data))
    app = 0

    for i, val in enumerate(data):
        app = app + _check(val, points[app::])
        prob[i] = app + 1

    return prob / len(points)
