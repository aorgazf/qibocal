# -*- coding: utf-8 -*-
import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import Dataset
from qibocal.decorators import plot
from qibocal.fitting.methods import lorentzian_fit


@plot("MSR and Phase vs Frequency", plots.frequency_msr_phase__fast_precision)
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

    data = Dataset(name=f"fast_sweep_q{qubit}", quantities={"frequency": "Hz"})
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
            platform.qd_port[qubit].lo_frequency = (qubit_frequency - freq) / 2
            qd_pulse2.frequency = freq - platform.qd_port[qubit].lo_frequency
            qd_pulse.frequency = platform.qd_port[qubit].lo_frequency - freq

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
