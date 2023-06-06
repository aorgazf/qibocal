from collections import Counter
from dataclasses import dataclass, field
from numbers import Number
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from qibocal.auto.operation import Results
from qibocal.protocols.characterization.randomized_benchmarking.fitting import (
    exp1_func,
    exp1B_func,
    fit_exp1_func,
    fit_exp1B_func,
)


@dataclass
class DecayResult(Results):
    """Data being described by a single decay, Ap^x."""

    # x and y data.
    x: Union[List[Number], np.ndarray]
    y: Union[List[Number], np.ndarray]
    # Fitting parameters.
    A: Optional[Number] = None
    Aerr: Optional[Number] = None
    p: Optional[Number] = None
    perr: Optional[Number] = None
    model: Iterable = field(default=exp1_func)
    meta_data: Dict = field(default_factory=dict)
    """The result data should behave according to that to that model."""

    def __post_init__(self):
        """Do some checks if the data given is correct. If no initial fitting parameters are given,
        choose a wise guess based on the given data.
        """
        if len(self.y) != len(self.x):
            raise ValueError(
                "Lenght of y and x must agree. len(x)={} != len(y)={}".format(
                    len(self.x), len(self.y)
                )
            )
        self.y_scatter = None
        if isinstance(self.y[0], Iterable):
            self.y_scatter = self.y
            self.y = [np.mean(y_row) for y_row in self.y]
        if self.A is None:
            self.A = np.max(self.y) - np.min(self.y)
        if self.p is None:
            self.p = 0.9
        self.fig = None

    @property
    def fitting_params(self):
        return (self.A, self.p)

    def fit(self, **kwargs):
        """Fits the data, all parameters given through kwargs will be passed on to the
        optimization function.
        """

        kwargs.setdefault("bounds", ((0, 0), (1, 1)))
        kwargs.setdefault("p0", (self.A, self.p))
        params, errs = fit_exp1_func(self.x, self.y, **kwargs)
        self.A, self.p = params
        self.Aerr, self.perr = errs

    def plot(self):
        """Plots the histogram data for each point and the averges plus the fit."""
        self.fig = plot_decay_result(self)
        return self.fig

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters
        if there are any.
        """

        if self.perr is not None:
            return (
                "Fit: y=Ap^x<br>A: {:.3f}\u00B1{:.3f}<br>p: {:.3f}\u00B1{:.3f}".format(
                    self.A, self.Aerr, self.p, self.perr
                )
            )
        return "DecayResult: Ap^x"


@dataclass
class DecayWithOffsetResult(DecayResult):
    """Data being described by a single decay with offset, Ap^x + B."""

    B: Optional[Number] = None
    Berr: Optional[Number] = None
    model: Iterable = field(default=exp1B_func)
    """The result data should behave according to that to that model."""

    def __post_init__(self):
        super().__post_init__()
        if self.B is None:
            self.B = np.mean(np.array(self.y))

    @property
    def fitting_params(self):
        return (*super().fitting_params, self.B)

    def fit(self, **kwargs):
        """Fits the data, all parameters given through kwargs will be passed on to the
        optimization function.
        """

        kwargs.setdefault("bounds", ((0, 0, 0), (1, 1, 1)))
        kwargs.setdefault("p0", (self.A, self.p, self.B))
        params, errs = fit_exp1B_func(self.x, self.y, **kwargs)
        self.A, self.p, self.B = params
        self.Aerr, self.perr, self.Berr = errs

    def __str__(self):
        """Overwrite the representation of the object with the fitting parameters
        if there are any.
        """

        if self.perr is not None:
            return "Fit: y=Ap^x+B<br>A: {:.3f}\u00B1{:.3f}<br>p: {:.3f}\u00B1{:.3f}<br>B: {:.3f}\u00B1{:.3f}".format(
                self.A, self.Aerr, self.p, self.perr, self.B, self.Berr
            )
        return "DecayResult: Ap^x+B"


def plot_decay_result(result: DecayResult) -> go.Figure:
    """Plots the average and the fitting data from a `DecayResult`.

    Args:
        result (DecayResult): Data to plot.
        fig (Optional[go.Figure], optional): If given, traces. Defaults to None.

    Returns:
        go.Figure: Figure with at least two traces, one for the data, one for the fit.
    """

    if result.y_scatter is not None:
        fig = plot_hists_result(result)
    else:
        fig = go.Figure()

    # Plot the x and y data from the result, they are (normally) the averages.
    fig.add_trace(
        go.Scatter(
            x=result.x,
            y=result.y,
            line={"color": "#aa6464"},
            mode="markers",
            name="average",
        )
    )
    # Build the fit and plot the fit.
    x_fit = np.linspace(min(result.x), max(result.x), 100)
    y_fit = result.model(x_fit, *result.fitting_params)
    fig.add_trace(
        go.Scatter(x=x_fit, y=y_fit, name=str(result), line=go.scatter.Line(dash="dot"))
    )
    fig.update_layout(xaxis_title="x", yaxis_title="y")
    return fig


def plot_hists_result(result: DecayResult) -> go.Figure:
    """Plots the distribution of data around each point.

    Args:
        result (DecayResult): Where the histogramm data comes from.

    Returns:
        go.Figure: A plotly figure with one single trace with the distribution of
    """
    counts_list, bins_list = get_hists_data(result.y_scatter)
    counts_list = sum(counts_list, [])
    fig_hist = go.Figure(
        go.Scatter(
            x=np.repeat(result.x, [len(bins) for bins in bins_list]),
            y=sum(bins_list, []),
            mode="markers",
            marker={"symbol": "square"},
            marker_color=[
                f"rgba(101, 151, 170, {count/max(counts_list)})"
                for count in counts_list
            ],
            text=counts_list,
            hovertemplate="<br>x:%{x}<br>y:%{y}<br>count:%{text}",
            name="iterations",
        )
    )

    return fig_hist


def get_hists_data(signal: Union[List[Number], np.ndarray]) -> Tuple[list, list]:
    """From a dataframe extract for each point the histogram data.

    Args:
        signal (list or np.ndarray): The raw data for the histogram.

    Returns:
        Tuple[list, list]: Counts and bins for each point.
    """

    # Get the exact number of occurences
    counters = [Counter(np.round(x, 3)) for x in signal]
    bins_list = [list(counter_x.keys()) for counter_x in counters]
    counts_list = [list(counter_x.values()) for counter_x in counters]

    return counts_list, bins_list
