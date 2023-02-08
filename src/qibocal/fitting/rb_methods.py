from typing import Tuple, Union

import numpy as np
from scipy.linalg import hankel, svd
from scipy.optimize import curve_fit


def exp1_func(x: np.ndarray, A: float, f: float, B: float) -> np.ndarray:
    """ """
    return A * f**x + B


def exp2_func(x: np.ndarray, A1: float, A2: float, f1: float, f2: float) -> np.ndarray:
    """ """
    return A1 * f1**x + A2 * f2**x

def expn_func(x: np.ndarray, As: list, fs: list) -> np.ndarray:
    """ """
    res = np.zeros(x.shape)
    for i in range(min(len(As), len(fs))):
        for j in range(len(x)):
            res[j] += As[i] * fs[i] ** x[j]
    return res


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

    # estimate dimension of the environment
    # print("[", end="")
    # for i in range(len(spectralMatrix)):
    #     print("[", end="")
    #     for j in range(len(spectralMatrix[i])):
    #         print(round(spectralMatrix[i][j], 3), end=", " if j < len(spectralMatrix[i])-1 else "")
    #     print("]," if i < len(spectralMatrix)-1 else "]]")
    dim_env = (np.linalg.norm(hankelMatrix, 'nuc')) / (np.linalg.norm(hankelMatrix, 'fro'))
    print("\nRank of the Hankel matrix:", dim_env, "\n")
    _, hankel_svds, _ = np.linalg.svd(hankelMatrix)
    hankel_svds.sort()
    print("Singular values =\n", np.round(hankel_svds, 3))

    # Calculate the poles/eigenvectors and space them right.
    decays = np.linalg.eigvals(spectralMatrix) * sampleRate
    return decays


def fit_exp1B_func(
    xdata: Union[np.ndarray, list], ydata: Union[np.ndarray, list], **kwargs
) -> Tuple[tuple, tuple, np.ndarray, np.ndarray]:
    """Calculate an single exponential fit to the given ydata.

    Args:
        xdata (Union[np.ndarray, list]): _description_
        ydata (Union[np.ndarray, list]): _description_

    Returns:
        Tuple[tuple, tuple]: _description_
    """
    # Check if all the values in ``ydata``are the same. That would make the
    # exponential fit unnecessary.
    if np.all(ydata == ydata[0]):
        popt, pcov = (ydata[0], 1.0, 0), (0, 0, 0)
    else:
        # Get a guess for the exponential function.
        guess = kwargs.get("p0", [0.5, 0.9, 0.8])
        # If the search for fitting parameters does not work just return
        # fixed parameters where one can see that the fit did not work
        try:
            popt, pcov = curve_fit(exp1_func, xdata, ydata, p0=guess, method="lm")
        except:
            popt, pcov = (0, 0, 0), (1, 1, 1)
    x_fit = np.linspace(np.sort(xdata)[0], np.sort(xdata)[-1], num=len(xdata) * 20)
    y_fit = exp1_func(x_fit, *popt)
    return popt, pcov, x_fit, y_fit


def fit_exp1_func(
    xdata: Union[np.ndarray, list], ydata: Union[np.ndarray, list], **kwargs
) -> Tuple[tuple, tuple, np.ndarray, np.ndarray]:
    """Calculate an single exponential fit to the given ydata.

    Args:
        xdata (Union[np.ndarray, list]): _description_
        ydata (Union[np.ndarray, list]): _description_

    Returns:
        Tuple[tuple, tuple]: _description_
    """
    # Check if all the values in ``ydata``are the same. That would make the
    # exponential fit unnecessary.
    if np.all(ydata == ydata[0]):
        popt, pcov = (ydata[0], 1.0), (0, 0)
    else:
        # Get a guess for the exponential function.
        guess = kwargs.get("p0", [0.5, 0.9, 0.8])
        # If the search for fitting parameters does not work just return
        # fixed parameters where one can see that the fit did not work
        try:
            popt, pcov = curve_fit(
                lambda x, A, f: exp1_func(x, A, f, 0),
                xdata,
                ydata,
                p0=guess[:-1],
                method="lm",
            )
        except:
            popt, pcov = (0, 0), (1, 1)
    x_fit = np.linspace(np.sort(xdata)[0], np.sort(xdata)[-1], num=len(xdata) * 20)
    y_fit = exp1_func(x_fit, *popt, 0)
    return (*popt, 0), (*pcov, 0), x_fit, y_fit


def fit_exp2_func(
    xdata: Union[np.ndarray, list], ydata: Union[np.ndarray, list], **kwargs
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """Calculate 2 exponentials on top of each other fit to the given ydata."""

    # TODO the data has to have a sufficiently big size, check that.
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
    return (*alphas, 0), (*decays, 0), np.real(x_fit), np.real(y_fit)

def fit_expn_func(
    xdata: Union[np.ndarray, list], ydata: Union[np.ndarray, list], **kwargs
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """Calculate n exponentials on top of each other fit to the given ydata."""

    # TODO the data has to have a sufficiently big size, check that.
    ndecays = 4
    decays = esprit(xdata, ydata, ndecays)
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
    y_fit = expn_func(x_fit, alphas, decays)
    return (*alphas, 0), (*decays, 0), np.real(x_fit), np.real(y_fit)
