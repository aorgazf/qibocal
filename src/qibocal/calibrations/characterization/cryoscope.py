# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import FluxPulse, Pulse, PulseSequence, PulseType, Rectangular

from qibocal import plots
from qibocal.calibrations.characterization.utils import iq_to_prob
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("cryoscope_raw", plots.cryoscope_raw)
# @plot("cryoscope_norm", plots.cryoscope_norm)
# @plot("cryoscope_norm_heatmap", plots.cryoscope_norm_heatmap)
@plot("cryoscope_fft", plots.cryoscope_fft)
#@plot("cryoscope_phase", plots.cryoscope_phase)
#@plot("cryoscope_phase_heatmap", plots.cryoscope_phase_heatmap)
#@plot("cryoscope_phase_unwrapped", plots.cryoscope_phase_unwrapped)
# @plot("cryoscope_phase_unwrapped_heatmap", plots.cryoscope_phase_unwrapped_heatmap)
# @plot("cryoscope_phase_amplitude_unwrapped_heatmap", plots.cryoscope_phase_amplitude_unwrapped_heatmap)
# @plot("cryoscope_fft_phase_unwrapped", plots.cryoscope_fft_phase_unwrapped)
#@plot("cryoscope_detuning_time", plots.cryoscope_detuning_time)
def cryoscope(
    platform: AbstractPlatform,
    qubit: int,
    flux_pulse_duration_start,
    flux_pulse_duration_end,
    flux_pulse_duration_step,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    flux_buffer,
    points=10,
):
    # 1) from state |0> apply Ry(pi/2) to state |+>,
    # 2) apply a flux pulse of variable duration,
    # 3) measure in the X and Y axis
    #   MX = Ry(pi/2) - (flux)(t) - Ry(pi/2)  - MZ
    #   MY = Ry(pi/2) - (flux)(t) - Rx(-pi/2) - MZ
    # The flux pulse detunes the qubit and results in a rotation around the Z axis = atan(MY/MX)

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
        # relative_phase=0,
        channel=platform.qubit_channel_map[qubit][2],
        qubit=qubit,
    )

    # rotate around the X asis Rx(-pi/2) to meassure Y component
    RX90_pulse = platform.create_RX90_pulse(
        qubit, start=flux_pulse.se_finish + flux_buffer, relative_phase=np.pi
    )

    # rotate around the Y asis Ry(pi/2) to meassure X component
    RY90_pulse = platform.create_RX90_pulse(
        qubit, start=flux_pulse.se_finish + flux_buffer, relative_phase=np.pi / 2
    )

    # add ro pulse at the end of each sequence
    MZ_ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX90_pulse.se_finish)

    # build the sequences
    MX_seq = initial_RY90_pulse + flux_pulse + RY90_pulse + MZ_ro_pulse
    MY_seq = initial_RY90_pulse + flux_pulse + RX90_pulse + MZ_ro_pulse

    MX_tag = "MX"
    MY_tag = "MY"

    # data = DataUnits(
    #     name=f"data_q{qubit}",
    #     quantities={
    #             "prob": "dimensionless",            
    #             "flux_pulse_duration": "ns",
    #             "flux_pulse_amplitude": "dimensionless",
    #             "component": "dimensionless"
    #     },
    # )


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
    durations = np.arange(
        flux_pulse_duration_start, flux_pulse_duration_end, flux_pulse_duration_step
    )
    count = 0

    for amplitude in amplitudes:
        for duration in durations:
            # while True:
            #     try:
            if count % points == 0:
                yield data
            flux_pulse.amplitude = amplitude
            flux_pulse.duration = duration
            prob_x = 1 - platform.execute_pulse_sequence(MX_seq, nshots=1024)["probability"][MZ_ro_pulse.serial]
            #MX_results = platform.execute_pulse_sequence(MX_seq)[MZ_ro_pulse.serial]
            results = {
                "prob[dimensionless]": prob_x,
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
                "component": MX_tag,
            }
            data.add(results)

            prob_y = 1 - platform.execute_pulse_sequence(MY_seq, nshots=1024)["probability"][MZ_ro_pulse.serial]
            #MY_results = platform.execute_pulse_sequence(MY_seq)[MZ_ro_pulse.serial]
            results = {
                "prob[dimensionless]": prob_y,
                "flux_pulse_duration[ns]": duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
                "component": MY_tag,
            }
            data.add(results)

                # except:
                #     continue
                # break
            count += 1
        yield data


@plot("cryoscope_delays", plots.cryoscope_delays)
def cryoscope_delays(
    platform: AbstractPlatform,
    qubit: int,
    flux_pulse_duration,
    flux_pulse_amplitude_start,
    flux_pulse_amplitude_end,
    flux_pulse_amplitude_step,
    flux_pulse_start, 
    flux_pulse_end, 
    flux_pulse_step,
    buffer,
    points=10,
):
    # 1) from state |0> apply Ry(pi/2) to state |+>,
    # 2) apply a flux pulse of variable duration,
    # 3) measure in the X and Y axis
    #   MX = Ry(pi/2) - (flux)(t) - Ry(pi/2)  - MZ
    #   MY = Ry(pi/2) - (flux)(t) - Rx(-pi/2) - MZ
    # The flux pulse detunes the qubit and results in a rotation around the Z axis = atan(MY/MX)

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
        start=initial_RY90_pulse.finish,
        duration=flux_pulse_duration,  # sweep to produce oscillations [300 to 400ns] in steps od 1ns? or 4?
        amplitude=flux_pulse_amplitude_start,  # fix for each run
        shape=Rectangular(),  # should be rectangular, but it gets distorted
        # relative_phase=0,
        channel=platform.qubit_channel_map[qubit][2],
        qubit=qubit,
    )

    # rotate around the X asis Rx(-pi/2) to meassure Y component
    RX90_pulse = platform.create_RX90_pulse(
        qubit, start=initial_RY90_pulse.finish + buffer, relative_phase=np.pi
    )

    # rotate around the Y asis Ry(pi/2) to meassure X component
    RY90_pulse = platform.create_RX90_pulse(
        qubit, start=initial_RY90_pulse.finish + buffer, relative_phase=-np.pi / 2
    )

    # add ro pulse at the end of each sequence
    MZ_ro_pulse = platform.create_qubit_readout_pulse(qubit, start=RX90_pulse.finish)

    MX_tag = "MX"
    MY_tag = "MY"

    data = DataUnits(
        name=f"data_q{qubit}",
        quantities={
            "flux_start": "ns",
            "flux_pulse_duration": "ns",
            "flux_pulse_amplitude": "dimensionless",
            "prob": "dimensionless",
        },
        options=["component"],
    )

    
    amplitudes = np.arange(flux_pulse_amplitude_start, flux_pulse_amplitude_end, flux_pulse_amplitude_step)
    flux_start_times = np.arange(flux_pulse_start, flux_pulse_end, flux_pulse_step)

    count = 0
    for amplitude in amplitudes:
        flux_pulse.amplitude = amplitude

        for flux_start in flux_start_times:  
            if count % points == 0:
                yield data
                
            # flux_pulse.start = flux_start

            # build the sequences
            if(flux_start >= 0):
                flux_pulse.start = flux_start 
                initial_RY90_pulse.start = 0
                RY90_pulse.start = initial_RY90_pulse.finish + buffer
                RX90_pulse.start = initial_RY90_pulse.finish + buffer
                MZ_ro_pulse.start = RY90_pulse.finish

            else:
                flux_pulse.start = 0
                initial_RY90_pulse.start = initial_RY90_pulse.start - flux_start
                RY90_pulse.start = initial_RY90_pulse.finish + buffer
                RX90_pulse.start = initial_RY90_pulse.finish + buffer
                MZ_ro_pulse.start = RY90_pulse.finish


            
            MX_seq = initial_RY90_pulse + flux_pulse + RY90_pulse + MZ_ro_pulse
            MY_seq = initial_RY90_pulse + flux_pulse + RX90_pulse + MZ_ro_pulse

            prob_x = 1 - platform.execute_pulse_sequence(MX_seq, nshots=1024)["probability"][MZ_ro_pulse.serial]
            #MX_results = platform.execute_pulse_sequence(MX_seq)[MZ_ro_pulse.serial]
            results = {
                "prob[dimensionless]": prob_x,
                "flux_start": flux_start,
                "flux_pulse_duration[ns]": flux_pulse_duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
                "component": MX_tag,
            }
            data.add(results)

            prob_y = 1 - platform.execute_pulse_sequence(MY_seq, nshots=1024)["probability"][MZ_ro_pulse.serial]
            #MY_results = platform.execute_pulse_sequence(MY_seq)[MZ_ro_pulse.serial]
            results = {
                "prob[dimensionless]": prob_y,
                "flux_start": flux_start,
                "flux_pulse_duration[ns]": flux_pulse_duration,
                "flux_pulse_amplitude[dimensionless]": amplitude,
                "component": MY_tag,
            }
            data.add(results)

                # except:
                #     continue
                # break
            count += 1
        yield data