import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import minimize

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("MSR vs length and amplitude", plots.offset_amplitude_msr_phase)
def rabi_ef(
    platform: AbstractPlatform,
    qubits: dict,
    pulse_offset_frequency_start,
    pulse_offset_frequency_end,
    pulse_offset_frequency_step,
    pulse_amplitude_start,
    pulse_amplitude_end,
    pulse_amplitude_step,
    software_averages,
    points=10,
):
    r"""
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

    data = DataUnits(
        name=f"data",
        quantities={"offset": "Hz", "amplitude": "dimensionless"},
        options=["qubit", "iteration"],
    )
    qubit_frequencies = {}

    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    RX_pulses = {}
    RX_pulses2 = {}
    for qubit in qubits:
        qubit_frequencies[qubit] = platform.qubits[qubit].drive_frequency
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=RX_pulses[qubit].se_finish, duration=40
        )
        RX_pulses2[qubit] = platform.create_RX_pulse(
            qubit, start=qd_pulses[qubit].se_finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].se_finish
        )
        sequence.add(RX_pulses[qubit])
        sequence.add(qd_pulses[qubit])
        sequence.add(RX_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    qd_pulse_frequency_range = np.arange(
        pulse_offset_frequency_start,
        pulse_offset_frequency_end,
        pulse_offset_frequency_step,
    )
    qd_pulse_amplitude_range = np.arange(
        pulse_amplitude_start, pulse_amplitude_end, pulse_amplitude_step
    )

    count = 0
    for iteration in range(software_averages):
        for off in qd_pulse_frequency_range:
            for amp in qd_pulse_amplitude_range:
                for qubit in qubits:
                    qd_pulses[qubit].amplitude = amp
                    qd_pulses[
                        qubit
                    ].frequency = off  # FIXME: how do I deal with the other IF
                if count % points == 0 and count > 0:
                    yield data
                    # yield rabi_ef_fit()

                # execute sequence
                results = platform.execute_pulse_sequence(sequence)

                for ro_pulse in ro_pulses.values():
                    # average msr, phase, i and q over the number of shots defined in the runcard
                    r = results[ro_pulse.serial].to_dict()
                    r.update(
                        {
                            "amplitude[dimensionless]": amp,
                            "offset[Hz]": off,
                            "qubit": ro_pulse.qubit,
                            "iteration": iteration,
                        }
                    )
                    data.add(r)
                count += 1
    idx = np.argmax(abs(data.get_values("MSR", "V") - data.get_values("MSR", "V")[0]))
    qubit_12frequency = qubit_frequency - data.get_values("offset", "Hz")[idx]
    ec_ej_res = minimize(
        lambda x: fit_score(
            x[0], x[1], qubit_frequency * 1e-9, qubit_12frequency * 1e-9
        ),
        [0.4, 4.8],
    )
    print(
        f"EC = {ec_ej_res['x'][0]}, Ej = {ec_ej_res['x'][1]} for an unharmonicity of {qubit_frequency - qubit_12frequency}"
    )
    yield data


def fit_score(ec, ej, w01, w12):
    r = calculate_transmon_transitions(ec, ej)
    return (r[0] - w01) ** 2 + (r[1] - w12) ** 2


def calculate_transmon_transitions(
    EC, EJ, asym=0, reduced_flux=0, no_transitions=2, dim=None, ng=0, return_injs=False
):
    r"""
    Calculates transmon energy levels from the full transmon qubit Hamiltonian.

    Creds to Ramiro for the script

    Parameters
    ----------
    EC:
        Charging energy of the transmon
    EJ:
        Inductive energy
    asym:
        Asymetry between the two junctions
    reduced_flux:
        Reduced flux
    no_transitions:
        Number of transitions
    dim:
        Number of dimensions
    ng:
        Not so sure
    return_injs:
        Not so sure

    """
    if dim is None:
        dim = no_transitions * 10

    EJphi = EJ * np.sqrt(
        asym**2 + (1 - asym**2) * np.cos(np.pi * reduced_flux) ** 2
    )
    Ham = 4 * EC * np.diag(np.arange(-dim - ng, dim - ng + 1) ** 2) - EJphi / 2 * (
        np.eye(2 * dim + 1, k=+1) + np.eye(2 * dim + 1, k=-1)
    )

    if return_injs:
        HamEigs, HamEigVs = np.linalg.eigh(Ham)
        # HamEigs.sort()
        transitions = HamEigs[1:] - HamEigs[:-1]
        charge_number_operator = np.diag(np.arange(-dim - ng, dim - ng + 1))
        injs = np.zeros([dim, dim])
        for i in range(dim):
            for j in range(dim):
                vect_i = np.matrix(HamEigVs[:, i])
                vect_j = np.matrix(HamEigVs[:, j])
                injs[i, j] = vect_i * (charge_number_operator * vect_j.getH())
        return transitions[:no_transitions], injs

    else:
        HamEigs = np.linalg.eigvalsh(Ham)
        HamEigs.sort()
        transitions = HamEigs[1:] - HamEigs[:-1]
        return transitions[:no_transitions]
