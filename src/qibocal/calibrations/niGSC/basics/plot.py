import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import qibocal.calibrations.niGSC.basics.fitting as fitting_methods
from qibocal.calibrations.niGSC.basics.experiment import Experiment
from qibocal.calibrations.niGSC.basics.utils import number_to_str
from qibocal.plots.utils import get_data_subfolders


def plot_qq(folder: str, routine: str, qubit, format):
    """Load the module for which the plot has to be done.


    Args:
        folder (str): The folder path where the data was stored.
        routine (str): The routine name, here the module name.
        qubit (Any): Is not used here
        format (Any): Is not used here.

    Returns:
        plotly figure: plotly graphical object figure.
    """

    import importlib

    subfolders = get_data_subfolders(folder)
    report_n = 0
    for subfolder in subfolders:
        # Load the module, something like 'standardrb'.
        module = importlib.import_module(f"qibocal.calibrations.niGSC.{routine}")
        # Load the experiment with the class method ``load``.
        experiment = module.ModuleExperiment.load(f"{folder}/{subfolder}/{routine}/")
        # In this data frame the precomputed fitting parameters and other
        # parameters for fitting and plotting are stored.
        aggr_df = pd.read_pickle(f"{folder}/data/{routine}/fit_plot.pkl")
        # Build the figure/report using the responsible module.
        plotly_figure, fitting_report = module.build_report(experiment, aggr_df)

        report_n += 1
    return [plotly_figure], fitting_report


class Report:
    """Once initialized with the correct parameters an Report object can build
    reports to display results of a gate set characterization experiment.
    """

    def __init__(self) -> None:
        self.all_figures = []
        self.title = "Report"
        self.info_dict = {}

    def build(self):
        l = len(self.all_figures)
        if l < 3:
            divide_by = 1
        else:
            divide_by = 2
        subplot_titles = [figdict.get("subplot_title") for figdict in self.all_figures]
        fig = make_subplots(
            rows=int(l / divide_by) + l % divide_by + 1,
            cols=1 if l == 1 else divide_by,
            subplot_titles=subplot_titles,
            vertical_spacing=0.3 / (2 * (int(l / divide_by) + l % divide_by)),
        )
        total_legend_size = 0
        for count, fig_dict in enumerate(self.all_figures):
            plot_list = fig_dict["figs"]
            total_legend_size += (
                len(plot_list) + (0 if fig_dict.get("subplot_title") is None else 1)
            ) * 55
            print(fig_dict.get("subplot_title"))
            for plot in plot_list:
                plot["legendgrouptitle"] = {
                    "font": {"size": 16},
                    "text": fig_dict.get("subplot_title"),
                }
                fig.add_trace(
                    plot, row=count // divide_by + 1, col=count % divide_by + 1
                )

        fig.update_xaxes(title_font_size=18, tickfont_size=16)
        fig.update_yaxes(title_font_size=18, tickfont_size=16)
        figure_height = (
            500 * (int(l / divide_by) + l % divide_by) if l > divide_by else 1000
        )
        fig.update_layout(
            font_family="Averta",
            hoverlabel_font_family="Averta",
            title_text=self.title,
            title_font_size=24,
            legend_font_size=16,
            hoverlabel_font_size=16,
            showlegend=True,
            legend_tracegroupgap=200
            if l < 3
            else (
                (figure_height - total_legend_size) / (l - 1)
                if total_legend_size < figure_height
                else 0
            ),
            legend_groupclick="toggleitem",
            height=figure_height,
            width=1000,
        )

        return fig


def scatter_fit_fig(
    experiment: Experiment,
    df_aggr: pd.DataFrame,
    xlabel: str,
    index: str,
    fittingparam_label="popt",
    is_imag=False,
):
    """
    Generate a figure dictionary for plotly.

    Args:
        experiment (:class:`qibocal.calibrations.niGSC.basics.experiment.Experiment`):
            an RB experiment that contains a dataframe with the information for the figure.
        df_aggr (pd.DataFrame): DataFrame containing aggregational data about the RB experiment.
        xlabel (str): key for x values in experiment.dataframe[xlabel] and df_aggr[xlabel].
        index (str): key for y values in experiment.dataframe[index] and df_aggr[index].
        fittingparam_label (str): key in df_aggr with dictionary containing the parameters for the dfrow["fit_func"].
            Default is "popt".
        is_imag (bool): Plots imaginary values of the data when True. Default is False.

    Returns:
        dict: dictionary for the plotly figure.
    """
    fig_traces = []
    dfrow = df_aggr.loc[index]

    legend_group_title = ("Imaginary " if is_imag else "") + index

    fig_traces.append(
        go.Scatter(
            x=experiment.dataframe[xlabel],
            y=np.imag(experiment.dataframe[index])
            if is_imag
            else np.real(experiment.dataframe[index]),
            line=dict(color="#6597aa"),
            mode="markers",
            marker={"opacity": 0.2, "symbol": "square"},
            name="runs",
            legendgroup=legend_group_title,
        )
    )
    fig_traces.append(
        go.Scatter(
            x=dfrow[xlabel],
            y=np.imag(dfrow["data"]) if is_imag else np.real(dfrow["data"]),
            line=dict(color="#aa6464"),
            mode="markers",
            name="average",
            legendgroup=legend_group_title,
        )
    )
    x_fit = np.linspace(min(dfrow[xlabel]), max(dfrow[xlabel]), len(dfrow[xlabel]) * 20)

    y_fit = getattr(fitting_methods, dfrow["fit_func"])(
        x_fit, *(dfrow[fittingparam_label].values())
    )
    y_fit = np.imag(y_fit) if is_imag else np.real(y_fit)
    print(dfrow[fittingparam_label])
    name = "".join(
        [
            "{}{}:{}".format(
                "<br>" if len(dfrow[fittingparam_label]) > 4 and key == "p1" else " ",
                key,
                number_to_str(dfrow[fittingparam_label][key]),
            )
            for key in dfrow[fittingparam_label]
        ]
    )
    print(name, index + str(len(fig_traces)))
    fig_traces.append(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            name=name,
            legendgroup=legend_group_title,
            line=go.scatter.Line(dash="dot"),
        )
    )

    return {
        "figs": fig_traces,
        "xlabel": xlabel,
        "ylabel": index,
        "subplot_title": "Real" if not is_imag else "Imaginary",
    }
