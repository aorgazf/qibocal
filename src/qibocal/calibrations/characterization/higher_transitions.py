# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Dataset
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase)
def measure_f12(
    platform: AbstractPlatform,
    qubit: int,
    offset_start,
    offset_end,
    offset_step,
    software_averages,
    points=10,
):
    platform.reload_settings()

    qubit_frequency = platform.characterization["single_qubit"][qubit]["qubit_freq"]
    sequence = PulseSequence()
    qd_pulse = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    qd_pulse2 = platform.create_qubit_drive_pulse(qubit, start=0, duration=5000)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=5000)
    sequence.add(qd_pulse)
    sequence.add(qd_pulse2)
    sequence.add(ro_pulse)

    freqrange = qubit_frequency - np.arange(offset_start, offset_end, offset_step)

    data = Dataset(name=f"data_q{qubit}", quantities={"frequency": "Hz"})
    count = 0
    for _ in range(software_averages):
        for freq in freqrange:
            if count % points == 0 and count > 0:
                yield data
                yield lorentzian_fit(
                    data,
                    x="frequency[GHz]",
                    y="MSR[uV]",
                    qubit=qubit,
                    nqubits=platform.settings["nqubits"],
                    labels=["qubit_freq", "peak_voltage"],
                )

            qd_pulse2.frequency = -(qubit_frequency - freq) / 2
            platform.qd_port[qubit].lo_frequency = freq - qd_pulse2.frequency
            qd_pulse.frequency = qubit_frequency - platform.qd_port[qubit].lo_frequency

            msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                ro_pulse.serial
            ]
            results = {
                "MSR[V]": msr,
                "i[V]": i,
                "q[V]": q,
                "phase[rad]": phase,
                "frequency[Hz]": freq,
            }
            data.add(results)
            count += 1
    yield data


@plot("MSR vs length and amplitude", plots.offset_amplitude_msr_phase)
def rabi_ef(
    platform: AbstractPlatform,
    qubit,
    pulse_offset_frequency_start,
    pulse_offset_frequency_end,
    pulse_offset_frequency_step,
    pulse_amplitude_start,
    pulse_amplitude_end,
    pulse_amplitude_step,
    software_averages,
    points=10,
):
    """
    Calibration routine to excite the |1> to |2>

    Sequence run is: RX - Pulse - RX - M
    The Pulse is the pulse that is being changed to excite the |1> - |2> transition.
    One needs to be mindfull of the IF used for the RX pulse. The offset frequency is removed from it.
    Rabi oscillations should be observed if the |1> - |2> frequency is hit because the amplitude is varied.

    Parameters
    ----------
    platform : AbstractPlatform
        Platform containing instrument and runcard being used in the routine
    qubit :
        Qubit name
    pulse_offset_frequency_start : float
        Pulse frequency offset start in the np.arange method from the |0> and |1> frequency
    pulse_offset_frequency_end : float
        Pulse frequency offset end in the np.arange method from the |0> and |1> frequency
    pulse_offset_frequency_step : float
        Pulse frequency offset step in the np.arange method from the |0> and |1> frequency
    pulse_amplitude_start : float
        Pulse amplitude start in the np.arange method
    pulse_amplitude_end  : float
        Pulse amplitude end in the np.arange method
    pulse_amplitude_step : float
        Pulse amplitude step in the np.arange method
    software_averages : int
        Number of times to repeat the routine
    points : int
        Data is saved and plot every points
    """
    platform.reload_settings()

    data = Dataset(
        name=f"data_q{qubit}", quantities={"offset": "Hz", "amplitude": "dimensionless"}
    )

    sequence = PulseSequence()
    RX_pulse = platform.create_RX_pulse(qubit, start=0)
    qd_pulse = platform.create_qubit_drive_pulse(
        qubit, start=RX_pulse.se_finish, duration=40
    )
    RX_pulse2 = platform.create_RX_pulse(qubit, start=qd_pulse.se_finish)
    ro_pulse = platform.create_qubit_readout_pulse(qubit, start=qd_pulse.se_finish)
    sequence.add(RX_pulse)
    sequence.add(qd_pulse)
    sequence.add(RX_pulse2)
    sequence.add(ro_pulse)

    qd_pulse_frequency_range = np.arange(
        pulse_offset_frequency_start,
        pulse_offset_frequency_end,
        pulse_offset_frequency_step,
    )
    qd_pulse_amplitude_range = np.arange(
        pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
    )

    count = 0
    for _ in range(software_averages):
        for off in qd_pulse_frequency_range:
            for amp in qd_pulse_amplitude_range:
                qd_pulse.amplitude = amp
                qd_pulse.frequency = RX_pulse.frequency - off
                if count % points == 0 and count > 0:
                    yield data
                msr, phase, i, q = platform.execute_pulse_sequence(sequence)[
                    ro_pulse.serial
                ]
                results = {
                    "MSR[V]": msr,
                    "i[V]": i,
                    "q[V]": q,
                    "phase[rad]": phase,
                    "offset[Hz]": off,
                    "amplitude[dimensionless]": amp,
                }
                data.add(results)
                count += 1

    yield data
