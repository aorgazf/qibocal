# -*- coding: utf-8 -*-
import numpy as np


def variable_resolution_scanrange(
    lowres_width, lowres_step, highres_width, highres_step
):
    """Helper function for sweeps."""
    return np.concatenate(
        (
            np.arange(-lowres_width, -highres_width, lowres_step),
            np.arange(-highres_width, highres_width, highres_step),
            np.arange(highres_width, lowres_width, lowres_step),
        )
    )

def iq_to_prob(i, q, mean_gnd, mean_exc):
    state = i + 1j * q
    state = state - mean_gnd
    mean_exc = mean_exc - mean_gnd
    state = state * np.exp(-1j * np.angle(mean_exc))
    mean_exc = mean_exc * np.exp(-1j * np.angle(mean_exc))
    return np.real(state) / np.real(mean_exc)
