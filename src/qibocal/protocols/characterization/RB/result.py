from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from qibocal.auto.operation import Results
from qibocal.calibrations.niGSC.basics.fitting import (
    exp1_func,
    exp1B_func,
    fit_exp1_func,
    fit_exp1B_func,
)
from qibocal.protocols.characterization.RB.utils import extract_from_data

real_numeric = Union[int, float, np.number]
numeric = Union[int, float, complex, np.number]
NoneType = type(None)


@dataclass
class DecayResult(Results):
    """ """

    m: Union[List[numeric], np.ndarray]
    y: Union[List[numeric], np.ndarray]
    A: Optional[numeric] = field(default=None)
    Aerr: Optional[numeric] = field(default=None)
    p: Optional[numeric] = field(default=None)
    perr: Optional[numeric] = field(default=None)
    hists: Tuple[List[numeric], List[numeric]] = field(
        default_factory=lambda: (list(), list())
    )

    def __post_init__(self):
        if len(self.y) != len(self.m):
            raise ValueError(
                "Lenght of y and m must agree. len(m)={} != len(y)={}".format(
                    len(self.m), len(self.y)
                )
            )
        if self.A is None:
            self.A = np.max(self.y) - np.mean(self.y)
        if self.p is None:
            self.p = 0.9
        self.func = exp1_func

    @property
    def fitting_params(self):
        return (self.A, self.p)

    def reset_fitting_params(self, A=None, p=None):
        self.A: Optional[numeric] = (
            A if A is not None else np.max(self.y) - np.mean(self.y)
        )
        self.p: Optional[numeric] = p if p is not None else 0.9

    def fit(self, **kwargs):
        kwargs.setdefault("bounds", ((0, 0), (1, 1)))
        kwargs.setdefault("p0", (self.A, self.p))
        params, errs = fit_exp1_func(self.m, self.y, **kwargs)
        self.A, self.p = params
        self.Aerr, self.perr = errs

    def plot(self):
        if self.hists is not None:
            self.fig = plot_hists_result(self)
        self.fig = plot_decay_result(self, self.fig)
        return self.fig

    def __str__(self):
        if self.perr is not None:
            return "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m".format(
                self.A, self.Aerr, self.p, self.perr
            )
        else:
            return "DecayResult: Ap^m"

    def get_tables(self):
        pass

    def get_figures(self):
        return [self.fig]


@dataclass
class DecayWithOffsetResult(DecayResult):
    """
    y[i] = (A +- Aerr) (p +- perr)^m[i] + (B +- Berr)
    # for later: y= sum_i A_i p_i^m (needs m integer)
    """

    B: Optional[numeric] = field(default=None)
    Berr: Optional[numeric] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.B is None:
            self.B = np.mean(np.array(self.y))
        self.func = exp1B_func

    @property
    def fitting_params(self):
        return (*super().fitting_params, self.B)

    def reset_fitting_params(self, A=None, p=None, B=None):
        super().reset_fitting_params(A, p)
        self.B: Optional[numeric] = B if B is not None else np.mean(self.y)

    def fit(self, **kwargs):
        kwargs.setdefault("bounds", ((0, 0, 0), (1, 1, 1)))
        kwargs.setdefault("p0", (self.A, self.p, self.B))
        params, errs = fit_exp1B_func(self.m, self.y, **kwargs)
        self.A, self.p, self.B = params
        self.Aerr, self.perr, self.Berr = errs

    def __str__(self):
        if self.perr is not None:
            return "({:.3f}\u00B1{:.3f})({:.3f}\u00B1{:.3f})^m + ({:.3f}\u00B1{:.3f})".format(
                self.A, self.Aerr, self.p, self.perr, self.B, self.Berr
            )
        else:
            return "DecayResult: Ap^m+B"


def plot_decay_result(
    result: DecayResult, fig: Optional[go.Figure] = None
) -> go.Figure:
    if fig is None:
        fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.m,
            y=result.y,
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
        )
    )
    m_fit = np.linspace(min(result.m), max(result.m), 100)
    y_fit = result.func(m_fit, *result.fitting_params)
    fig.add_trace(
        go.Scatter(x=m_fit, y=y_fit, name=str(result), line=go.scatter.Line(dash="dot"))
    )
    return fig


def plot_hists_result(result: DecayResult) -> go.Figure:
    counts_list, bins_list = result.hists
    counts_list = sum(counts_list, [])
    fig_hist = go.Figure(
        go.Scatter(
            x=np.repeat(result.m, [len(bins) for bins in bins_list]),
            y=sum(bins_list, []),
            mode="markers",
            marker={"symbol": "square"},
            marker_color=[
                f"rgba(101, 151, 170, {count/max(counts_list)})"
                for count in counts_list
            ],
            text=[count for count in counts_list],
            hovertemplate="<br>x:%{x}<br>y:%{y}<br>count:%{text}",
            name="iterations",
        )
    )

    return fig_hist


def get_hists_data(data_agg: DecayResult):
    signal = extract_from_data(data_agg, "signal", "depth")[1].reshape(
        -1, data_agg.attrs["niter"]
    )
    counts_list, bins_list = zip(*[np.histogram(x, bins=12) for x in signal])
    counts_list, bins_list = list(counts_list), list(bins_list)
    for k in range(len(counts_list)):
        bins, counts = bins_list[k], counts_list[k]
        bins = bins[:-1] + (bins[1] - bins[0]) / 2
        bins_list[k] = list(bins[counts != 0])
        counts_list[k] = list(counts[counts != 0])
    return counts_list, bins_list
