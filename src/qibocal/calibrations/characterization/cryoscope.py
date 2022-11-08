# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, PulseType, Rectangular

from qibocal import plots
from qibocal.calibrations.characterization.utils import iq_to_prob
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("cryoscope_raw", plots.cryoscope_raw)
@plot("cryoscope_norm", plots.cryoscope_norm)
@plot("cryoscope_norm_heatmap", plots.cryoscope_norm_heatmap)
@plot("cryoscope_fft", plots.cryoscope_fft)
@plot("cryoscope_phase", plots.cryoscope_phase)
@plot("cryoscope_phase_heatmap", plots.cryoscope_phase_heatmap)
@plot("cryoscope_phase_unwrapped", plots.cryoscope_phase_unwrapped)
@plot("cryoscope_phase_unwrapped_heatmap", plots.cryoscope_phase_unwrapped_heatmap)
@plot("cryoscope_phase_amplitude_unwrapped_heatmap", plots.cryoscope_phase_amplitude_unwrapped_heatmap)
@plot("cryoscope_fft_phase_unwrapped", plots.cryoscope_fft_phase_unwrapped)
@plot("cryoscope_detuning_time", plots.cryoscope_detuning_time)
# @plot("cryoscope_detuning_amplitude", plots.cryoscope_detuning_amplitude)
def cryoscope(
    platform: AbstractPlatform,
    qubit: int,
    flux_pulse_duration_start,
    flux_pulse_duration_end,
    flux_pulse_duration_step,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    points=10,
):

    """Cryoscope calibration routine (arxiv ref: XXXXX)
    This characterization method allows the user to calibrate CPhase gates. Given a qubit,
    the method implements the following sequences for a range of different flux pulse durations and amplitudes:
           MX = Ry(pi/2) - (flux)(t) - Ry(pi/2)  - MZ
           MY = Ry(pi/2) - (flux)(t) - Rx(-pi/2) - MZ
    
    The flux pulse detunes the qubit and results in a rotation around the Z axis = atan(MY/MX)
    From the data analysis performed in the batch of functions for cryoscope analysis (see decorators)
    we parametrize the flux pulse used to implement CPhase gates in a given qubit.

    Args:
        platfform (AbstrcatPlatform): Qibolab object that allows the user to communicate with the experimental setup (QPU)
        flux_pulse_duration_start (int): initial duration of the flux pulse range values
        flux_pulse_duration_step (int): incremental duration of the flux pulse range values
        flux_pulse_duration_start (int): end duration of the flux pulse range values
        flux_pulse_amplitude_start (float): initial amplitude of the flux pulse range values (0 <= flux_pulse_amplitude_start <= 1)
        flux_pulse_amplitude_end (float): incremental amplitude of the flux pulse range values
        flux_pulse_amplitude_step (float): end amplitude of the flux pulse range values (0 <= flux_pulse_amplitude_start <= 1)
        points (int): Number of points obtained to executed the save method of the results in a file
    """

    platform.reload_settings()
    mean_gnd = complex(
        platform.characterization["single_qubit"][qubit]["mean_gnd_states"]
    )
    mean_exc = complex(
        platform.characterization["single_qubit"][qubit]["mean_exc_states"]
    )

    # start at |+> by rotating Ry(pi/2)
    initial_RY90_pulse = platform.create_RX90_pulse(
        qubit, start=0, relative_phase=np.pi / 2
    )
    # apply a detuning flux pulse
    flux_pulse = FluxPulse(
        start=initial_RY90_pulse.se_finish,
        duration=flux_pulse_duration_start,  # sweep to produce oscillations [300 to 400ns] in steps od 1ns? or 4?
        amplitude=flux_pulse_amplitude_start,  # fix for each run
        shape=Rectangular(),  # should be rectangular, but it gets distorted
        relative_phase=0,
        channel=platform.qubit_channel_map[qubit][2],
        qubit=qubit,
    )

    # rotate around the X asis Rx(-pi/2) to meassure Y component
    RX90_pulse = platform.create_RX90_pulse(
        qubit, start=flux_pulse.se_finish, relative_phase=np.pi
    )

    # rotate around the Y asis Ry(pi/2) to meassure X component
    RY90_pulse = platform.create_RX90_pulse(
        qubit, start=flux_pulse.se_finish, relative_phase=np.pi / 2
    )

    # add ro pulse at the end of each sequence
    MZ_ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX90_pulse.se_finish)

    # build the sequences
    MX_seq = initial_RY90_pulse + flux_pulse + RY90_pulse + MZ_ro_pulse
    MY_seq = initial_RY90_pulse + flux_pulse + RX90_pulse + MZ_ro_pulse

    MX_tag = "MX"
    MY_tag = "MY"

    data = DataUnits(
        name=f"data_q{qubit}",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
            "prob": "dimensionless",
        },
        options=["component"],
    )

    amplitudes = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    # durations_more = np.arange(
    #     flux_pulse_duration_start,
    #     int(
    #         flux_pulse_duration_start
    #         + (flux_pulse_duration_end - flux_pulse_duration_start)
    #         / 8
    #         // flux_pulse_duration_step
    #         * flux_pulse_duration_step
    #     ),
    #     flux_pulse_duration_step,
    # )
    # durations_less = np.arange(
    #     int(
    #         flux_pulse_duration_start
    #         + (
    #             (
    #                 (flux_pulse_duration_end - flux_pulse_duration_start)
    #                 / 8
    #                 // flux_pulse_duration_step
    #             )
    #             + 1
    #         )
    #         * flux_pulse_duration_step
    #     ),
    #     flux_pulse_duration_end,
    #     flux_pulse_duration_step * 4,
    # )
    # durations = np.concatenate((durations_more, durations_less))
    durations = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )
    count = 0

    for amplitude in amplitudes:
        for duration in durations:
            if count % points == 0:
                yield data
            flux_pulse.amplitude = amplitude
            flux_pulse.duration = duration

            MX_results = platform.execute_pulse_sequence(MX_seq)[MZ_ro_pulse.serial]
            results = {
                "MSR[V]": MX_results[0],
                "i[V]": MX_results[2],
                "q[V]": MX_results[3],
                "phase[rad]": MX_results[1],
                "prob[dimensionless]": iq_to_prob(
                    MX_results[2], MX_results[3], mean_gnd, mean_exc
                ),
                "component": MX_tag,
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
            }
            data.add(results)

            MY_results = platform.execute_pulse_sequence(MY_seq)[MZ_ro_pulse.serial]
            results = {
                "MSR[V]": MY_results[0],
                "i[V]": MY_results[2],
                "q[V]": MY_results[3],
                "phase[rad]": MY_results[1],
                "prob[dimensionless]": iq_to_prob(
                    MY_results[2], MY_results[3], mean_gnd, mean_exc
                ),
                "component": MY_tag,
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
            }
            data.add(results)

            count += 1
        yield data


# @plot("cryoscope", plots.cryoscope_amplitude)
def cryoscope_amplitude(
    platform: AbstractPlatform,
    qubit: int,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    points=10,
):
    # 1) from state |0> apply Ry(pi/2) to state |+>,
    # 2) apply a flux pulse of variable duration,
    # 3) measure in the X and Y axis
    #   MX = Ry(pi/2) - (flux)(t) - Ry(pi/2)  - MZ
    #   MY = Ry(pi/2) - (flux)(t) - Rx(-pi/2) - MZ
    #   MZ = Ry(pi/2) - (flux)(t) - wait      - MZ
    # The flux pulse detunes the qubit and results in a rotation around the Z axis = atan(MY/MX)

    platform.reload_settings()

    # start at |+> by rotating Ry(pi/2)
    initial_RY90_pulse = platform.create_RX90_pulse(
        qubit, start=0, relative_phase=np.pi / 2
    )
    # apply a detuning flux pulse
    flux_pulse = FluxPulse(
        start=initial_RY90_pulse.se_finish,
        duration=40,  # sweep to produce oscillations [300 to 400ns] in steps od 1ns? or 4?
        amplitude=flux_pulse_amplitude_start,  # fix for each run
        shape=Rectangular(),  # should be rectangular, but it gets distorted
        channel=platform.qubit_channel_map[qubit][2],
        qubit=qubit,
    )

    # rotate around the X asis Rx(-pi/2) to meassure Y component
    RX90_pulse = platform.create_RX90_pulse(
        qubit, start=flux_pulse.se_finish, relative_phase=np.pi
    )

    # rotate around the Y asis Ry(pi/2) to meassure X component
    RY90_pulse = platform.create_RX90_pulse(
        qubit, start=flux_pulse.se_finish, relative_phase=np.pi / 2
    )

    # add ro pulse at the end of each sequence
    MZ_ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX90_pulse.se_finish)

    # build the sequences
    MX_seq = initial_RY90_pulse + flux_pulse + RY90_pulse + MZ_ro_pulse
    MY_seq = initial_RY90_pulse + flux_pulse + RX90_pulse + MZ_ro_pulse
    MZ_seq = initial_RY90_pulse + flux_pulse + MZ_ro_pulse

    MX_tag = "MX"
    MY_tag = "MY"
    MZ_tag = "MZ"

    data = DataUnits(
        name=f"data_q{qubit}",
        quantities={
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
        },
        options=["component"],
    )

    amplitudes = np.arange(
        flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step
    )
    count = 0

    for amplitude in amplitudes:
        if count % points == 0:
            yield data
        flux_pulse.amplitude = amplitude

        MX_results = platform.execute_pulse_sequence(MX_seq)[MZ_ro_pulse.serial]
        results = {
            "MSR[V]": MX_results[0],
            "i[V]": MX_results[2],
            "q[V]": MX_results[3],
            "phase[rad]": MX_results[1],
            "component": MX_tag,
            "flux_pulse_amplitude[dimensionless]": amplitude,
        }
        data.add(results)

        MY_results = platform.execute_pulse_sequence(MY_seq)[MZ_ro_pulse.serial]
        results = {
            "MSR[V]": MY_results[0],
            "i[V]": MY_results[2],
            "q[V]": MY_results[3],
            "phase[rad]": MY_results[1],
            "component": MY_tag,
            "flux_pulse_amplitude[dimensionless]": amplitude,
        }
        data.add(results)

        MZ_results = platform.execute_pulse_sequence(MZ_seq)[MZ_ro_pulse.serial]
        results = {
            "MSR[V]": MZ_results[0],
            "i[V]": MZ_results[2],
            "q[V]": MZ_results[3],
            "phase[rad]": MZ_results[1],
            "component": MZ_tag,
            "flux_pulse_amplitude[dimensionless]": amplitude,
        }
        data.add(results)

        count += 1
    yield data
