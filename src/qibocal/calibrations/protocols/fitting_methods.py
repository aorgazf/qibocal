import pdb
from typing import Tuple, Union

import numpy as np
from scipy.linalg import hankel, svd
from scipy.optimize import curve_fit

def generate_id(noise = "U"): 
    """noise = "U", "P", "T" """
    from datetime import datetime
    return noise + datetime.now().strftime("%H-%M")

def exp1_func(x: np.ndarray, A: float, f: float, B: float) -> np.ndarray:
    """ """
    return A * f**x + B


def exp2_func(x: np.ndarray, A1: float, A2: float, f1: float, f2: float) -> np.ndarray:
    """ """
    return A1 * f1**x + A2 * f2**x


def esprit(xdata, ydata, num_decays, hankel_dim=None):
    """Implements the ESPRIT algorithm for peak detection
    TODO write documentation of ESPRIT
    `"""

    # TODO the xdata has to be equally spaced, check that.
    sampleRate = 1 / (xdata[1] - xdata[0])
    # xdata has to be an array.
    xdata = np.array(xdata)
    if hankel_dim is None:
        hankel_dim = int(np.round(0.5 * xdata.size))
        hankel_dim = hankel_dim
    # Find the dimension of the hankel matrix such that the mulitplication
    # processes don't break.
    hankel_dim = max(num_decays + 1, hankel_dim)  # 5
    hankel_dim = min(hankel_dim, xdata.size - num_decays + 1)  # 5
    hankelMatrix = hankel(ydata[:hankel_dim], ydata[(hankel_dim - 1) :])
    # Calculate nontrivial (nonzero) singular vectors of the hankel matrix.
    U, _, _ = svd(hankelMatrix, full_matrices=False)
    # Cut off the columns to the amount which is needed.
    U_signal = U[:, :num_decays]
    # Calculte the solution.
    spectralMatrix = (
        np.linalg.pinv(
            U_signal[
                :-1,
            ]
        )
        @ U_signal[
            1:,
        ]
    )
    # Calculate the poles/eigenvectors and space them right.
    decays = np.linalg.eigvals(spectralMatrix) * sampleRate

    # estimate dimension of te environment
    dim_env = (np.linalg.norm(spectralMatrix, 1) ** 2) / (np.linalg.norm(hankelMatrix, 'fro') ** 2)
    print("\nDimenion of the enironment is:", dim_env, "\n")
    svds = np.linalg.eigvals(spectralMatrix)
    svds.sort()
    print('singular values =\n', np.round(svds, 3))
    # Create a report
    try:
        with open(f"/home/yelyzavetavodovozova/Documents/plots/{generate_id()}.txt", 'a') as f:
            f.write("\ndim_env = " + str(dim_env))
            f.write("\nsvds = " + str(svds))
    except FileNotFoundError:
        print("The directory does not exist")

    return decays


def fit_exp1_func(
    xdata: Union[np.ndarray, list], ydata: Union[np.ndarray, list], **kwargs
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """Calculate an single exponential fit to the given ydata."""

    # Get a guess for the exponential function.
    guess = kwargs.get("p0", [0.5, 0.9, 0.8])
    # If the search for fitting parameters does not work just return
    # fixed parameters where one can see that the fit did not work
    try:
        popt, _ = curve_fit(exp1_func, xdata, ydata, p0=guess, method="lm")
    except:
        popt = (0, 0, 0)
    # Build a finer spaces xdata array for plotting the fit.
    x_fit = np.linspace(np.sort(xdata)[0], np.sort(xdata)[-1], num=len(xdata) * 20)
    # Get the ydata for the fit with the calculated parameters.
    y_fit = exp1_func(x_fit, *popt)
    return x_fit, y_fit, popt


def fit_exp2_func(
    xdata: Union[np.ndarray, list], ydata: Union[np.ndarray, list], **kwargs
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """Calculate 2 exponentials on top of each other fit to the given ydata."""

    decays = esprit(xdata, ydata, 2)
    vandermonde = np.vander(decays, N=xdata[-1] + 1, increasing=True)
    vandermonde = np.take(vandermonde, xdata, axis=1)
    alphas = np.linalg.pinv(vandermonde.T) @ ydata.reshape(-1, 1).flatten()
    # Build a finer spaces xdata array for plotting the fit.
    if sum(decays < 0):
        dtype = complex
    else:
        dtype = float
    x_fit = np.linspace(
        np.sort(xdata)[0], np.sort(xdata)[-1], num=len(xdata) * 20, dtype=dtype
    )
    # Get the ydata for the fit with the calculated parameters.
    y_fit = exp2_func(x_fit, *alphas, *decays)
    return x_fit, y_fit, tuple([*alphas, *decays])
