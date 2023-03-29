import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibocal.data import Data, DataUnits
from qibocal.plots.utils import get_color, get_data_subfolders


# Fast reset oscillations
def fast_reset_states(folder, routine, qubit, format):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("State",),
    )

    # iterate over multiple data folders
    subfolders = get_data_subfolders(folder)
    report_n = 0
    fitting_report = ""
    for subfolder in subfolders:
        try:
            data = DataUnits.load_data(folder, subfolder, routine, format, "data")
            data.df = data.df[data.df["qubit"] == qubit]
        except:
            data = DataUnits(options=["qubit", "iteration", "n"])

        iterations = data.df["iteration"]
        states = data.df["MSR"].pint.to("V").pint.magnitude

        opacity = 1

        state0_count = states.value_counts()[0]
        state1_count = states.value_counts()[1]
        error = (state1_count - state0_count) / len(states)

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=states,
                mode="markers",
                marker_color=get_color(report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}",
                showlegend=True,
                legendgroup=f"q{qubit}/r{report_n}",
            ),
            row=1,
            col=1,
        )

        title_text = ""
        title_text += f"q{qubit}/r{report_n} | state0 count: {state0_count:.0f}<br>"
        title_text += f"q{qubit}/r{report_n} | state1 count: {state1_count:.0f}<br>"
        title_text += f"q{qubit}/r{report_n} | Error: {error:.6f}<br>"
        title_text += (
            f"q{qubit}/r{report_n} | Fidelity(add 0 to 1 error): {(1 - error):.6f}<br>"
        )

        fitting_report = fitting_report + title_text
        report_n += 1

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Shot",
        yaxis_title="State",
    )

    figures.append(fig)

    return figures, fitting_report
