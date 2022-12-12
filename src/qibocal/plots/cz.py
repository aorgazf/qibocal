import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import Platform

from qibocal.calibrations.characterization.utils import iq_to_prob
from qibocal.data import DataUnits


def duration_amplitude_msr_flux_pulse(folder, routine, qubit, format):

    highfreq = 2
    qubit = int(qubit)
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    platform = Platform("qw5q_gold")

    # Mean and excited states
    mean_gnd = {
        str(highfreq): complex(
            platform.characterization["single_qubit"][highfreq]["mean_gnd_states"]
        ),
        str(lowfreq): complex(
            platform.characterization["single_qubit"][lowfreq]["mean_gnd_states"]
        ),
    }
    mean_exc = {
        str(highfreq): complex(
            platform.characterization["single_qubit"][highfreq]["mean_exc_states"]
        ),
        str(lowfreq): complex(
            platform.characterization["single_qubit"][lowfreq]["mean_exc_states"]
        ),
    }

    data = DataUnits.load_data(folder, routine, format, f"data_q{lowfreq}{highfreq}")

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V) - High Frequency",
            "MSR (V) - Low Frequency",
        ),
    )

    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "high"
            ].to_numpy(),
            z=iq_to_prob(
                data.get_values("i", "V")[data.df["q_freq"] == "high"].to_numpy(),
                data.get_values("q", "V")[data.df["q_freq"] == "high"].to_numpy(),
                mean_gnd[str(highfreq)],
                mean_exc[str(highfreq)],
            ),
            colorbar_x=0.45,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=data.get_values("flux_pulse_duration", "ns")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            y=data.get_values("flux_pulse_amplitude", "dimensionless")[
                data.df["q_freq"] == "low"
            ].to_numpy(),
            z=iq_to_prob(
                data.get_values("i", "V")[data.df["q_freq"] == "low"].to_numpy(),
                data.get_values("q", "V")[data.df["q_freq"] == "low"].to_numpy(),
                mean_gnd[str(lowfreq)],
                mean_exc[str(lowfreq)],
            ),
            colorbar_x=1.0,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="duration (ns)",
        yaxis_title="amplitude (dimensionless)",
        xaxis2_title="duration (ns)",
        yaxis2_title="amplitude (dimensionless)",
    )
    return fig


def landscape_2q_gate(folder, routine, qubit, format):

    qubit = int(qubit)
    highfreq = 2
    lowfreq = qubit
    if qubit > 2:
        highfreq = qubit
        lowfreq = 2

    platform = Platform("qw5q_gold")

    # Mean and excited states
    mean_gnd = {
        str(highfreq): complex(
            platform.characterization["single_qubit"][highfreq]["mean_gnd_states"]
        ),
        str(lowfreq): complex(
            platform.characterization["single_qubit"][lowfreq]["mean_gnd_states"]
        ),
    }
    mean_exc = {
        str(highfreq): complex(
            platform.characterization["single_qubit"][highfreq]["mean_exc_states"]
        ),
        str(lowfreq): complex(
            platform.characterization["single_qubit"][lowfreq]["mean_exc_states"]
        ),
    }

    data = DataUnits.load_data(folder, routine, format, f"data_q{lowfreq}{highfreq}")

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V) - High Frequency",
            "MSR (V) - Low Frequency",  # TODO: change this to <Z>
        ),
    )
    y = iq_to_prob(
        data.get_values("i", "V")[data.df["q_freq"] == "high"].to_numpy(),
        data.get_values("q", "V")[data.df["q_freq"] == "high"].to_numpy(),
        mean_gnd[str(highfreq)],
        mean_exc[str(highfreq)],
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("theta", "rad")[data.df["q_freq"] == "high"][
                data.df["setup"] == "I"
            ].to_numpy(),
            y=iq_to_prob(
                data.get_values("i", "V")[data.df["q_freq"] == "high"][
                    data.df["setup"] == "I"
                ].to_numpy(),
                data.get_values("q", "V")[data.df["q_freq"] == "high"][
                    data.df["setup"] == "I"
                ].to_numpy(),
                mean_gnd[str(highfreq)],
                mean_exc[str(highfreq)],
            ),
            name="I",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("theta", "rad")[data.df["q_freq"] == "high"][
                data.df["setup"] == "X"
            ].to_numpy(),
            y=iq_to_prob(
                data.get_values("i", "V")[data.df["q_freq"] == "high"][
                    data.df["setup"] == "X"
                ].to_numpy(),
                data.get_values("q", "V")[data.df["q_freq"] == "high"][
                    data.df["setup"] == "X"
                ].to_numpy(),
                mean_gnd[str(highfreq)],
                mean_exc[str(highfreq)],
            ),
            name="X",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("theta", "rad")[data.df["q_freq"] == "low"][
                data.df["setup"] == "I"
            ].to_numpy(),
            y=iq_to_prob(
                data.get_values("i", "V")[data.df["q_freq"] == "low"][
                    data.df["setup"] == "I"
                ].to_numpy(),
                data.get_values("q", "V")[data.df["q_freq"] == "low"][
                    data.df["setup"] == "I"
                ].to_numpy(),
                mean_gnd[str(lowfreq)],
                mean_exc[str(lowfreq)],
            ),
            name="I",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=data.get_values("theta", "rad")[data.df["q_freq"] == "low"][
                data.df["setup"] == "X"
            ].to_numpy(),
            y=iq_to_prob(
                data.get_values("i", "V")[data.df["q_freq"] == "low"][
                    data.df["setup"] == "X"
                ].to_numpy(),
                data.get_values("q", "V")[data.df["q_freq"] == "low"][
                    data.df["setup"] == "X"
                ].to_numpy(),
                mean_gnd[str(lowfreq)],
                mean_exc[str(lowfreq)],
            ),
            name="X",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="phase (rad)",
        yaxis_title="prob",
        xaxis2_title="phase (rad)",
        yaxis2_title="prob",
    )
    return fig
