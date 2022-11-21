import os.path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.fitting.utils import cos, exp, flipping, line, lorenzian, rabi, ramsey


def frequency_current_flux(folder, routine, qubit, format):
    """Plot of the experimental data for the flux resonator spectroscopy and its corresponding fit.
        Args:
        folder (str): Folder where the data files with the experimental and fit data are.
        routine (str): Routine name (resonator_flux_sample_matrix)
        qubit (int): qubit coupled to the resonator for which we want to plot the data.
        format (str): format of the data files.

    Returns:
        fig (Figure): Figure associated to data.

    """
    fluxes = []
    fluxes_fit = []
    for i in range(5):  # FIXME: 5 is hardcoded
        file1 = f"{folder}/data/{routine}/data_q{qubit}_f{i}.csv"
        file2 = f"{folder}/data/{routine}/fit_q{qubit}_f{i}.csv"
        if os.path.exists(file1):
            fluxes += [i]
        if os.path.exists(file2):
            fluxes_fit += [i]

    if len(fluxes) < 1:
        nb = 1
    else:
        nb = len(fluxes)
    fig = make_subplots(
        rows=1,
        cols=nb,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
        x_title="Current (A)",
        y_title="Frequency (GHz)",
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for j in fluxes:

        data_spec = DataUnits.load_data(folder, routine, format, f"data_q{qubit}_f{j}")
        fig.add_trace(
            go.Scatter(
                x=data_spec.get_values("current", "A"),
                y=data_spec.get_values("frequency", "GHz"),
                name=f"fluxline: {j}",
            ),
            row=1,
            col=j,
        )

        if j in fluxes_fit:

            data_fit = Data.load_data(folder, routine, format, f"fit_q{qubit}_f{j}")
            if len(data_spec) > 0 and len(data_fit) > 0:
                curr_range = np.linspace(
                    min(data_spec.get_values("current", "A")),
                    max(data_spec.get_values("current", "A")),
                    100,
                )
                params = [i for i in list(data_fit.df.keys()) if "popt" not in i]
                if j == qubit:
                    fig.add_trace(
                        go.Scatter(
                            x=curr_range,
                            y=cos(
                                curr_range,
                                data_fit.get_values("popt0"),
                                data_fit.get_values("popt1"),
                                data_fit.get_values("popt2"),
                                data_fit.get_values("popt3"),
                            ),
                            name=f"Fit fluxline {j}",
                            line=go.scatter.Line(dash="dot"),
                        ),
                        row=1,
                        col=j,
                    )

                else:
                    fig.add_trace(
                        go.Scatter(
                            x=curr_range,
                            y=line(
                                curr_range,
                                data_fit.get_values("popt0"),
                                data_fit.get_values("popt1"),
                            ),
                            name=f"Fit fluxline {j}",
                            line=go.scatter.Line(dash="dot"),
                        ),
                        row=1,
                        col=j,
                    )

                fig.update_layout(margin=dict(l=20, r=20, t=20, b=170))
                if j == qubit:
                    fig.add_annotation(
                        dict(
                            font=dict(color="black", size=12),
                            x=0,
                            y=-0.25 - 0.1 * j,
                            showarrow=False,
                            # text=f"Qubit: {qubit} Fluxline: {j} C_{qubit}{j} = {data_fit.df[params[3]][0]:.3f} +- {data_fit.df[params[2]][0]:.1f} GHz/A. freq_{qubit} = {data_fit.df[params[1]][0]:.5f} +- {data_fit.df[params[0]][0]:.1f} GHz. {params[7]} = {data_fit.df[params[7]][0]:.3} +- {data_fit.df[params[6]][0]:.1f} A. {params[5]} = {data_fit.df[params[5]][0]:.3f} +- {data_fit.df[params[4]][0]:.1f} GHz. ",
                            text=f"Qubit: {qubit} Fluxline: {j} C_{qubit}{j} = {data_fit.df[params[3]][0]:.3f} GHz/A. freq_{qubit} = {np.round_(data_fit.df[params[1]][0],4)} GHz. {params[7]} = {np.round_(data_fit.df[params[7]][0],3)} A. {params[5]} = {np.round(data_fit.df[params[5]][0],4)} GHz. ",
                            xanchor="left",
                            xref="paper",
                            yref="paper",
                        )
                    )
                else:
                    fig.add_annotation(
                        dict(
                            font=dict(color="black", size=12),
                            x=0,
                            y=-0.25 - 0.1 * j,
                            showarrow=False,
                            # text=f"Qubit: {qubit} Fluxline: {j} C_{qubit}{j} = {data_fit.df[params[1]][0]:.3f} +- {data_fit.df[params[0]][0]:.1f} GHz/A.",
                            text=f"Qubit: {qubit} Fluxline: {j} C_{qubit}{j} = {data_fit.df[params[1]][0]:.3f} GHz/A.",
                            xanchor="left",
                            xref="paper",
                            yref="paper",
                        )
                    )

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    return fig
