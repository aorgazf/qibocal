import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot
from qibocal.fitting.methods import t1_fit


@plot("MSR vs Time", plots.t1_time_msr)
def t1_sweep(
    platform: AbstractPlatform,
    qubits: list,
    delay_before_readout_start,
    delay_before_readout_end,
    delay_before_readout_step,
    nshots=1024,
    relaxation_time=None,
    software_averages=1,
):
    r"""
    In a T1 experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        delay_before_readout_start (int): Initial time delay before ReadOut
        delay_before_readout_end (list): Maximum time delay before ReadOut
        delay_before_readout_step (int): Scan range step for the delay before ReadOut
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **wait[ns]**: Delay before ReadOut used in the current execution
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **labels[0]**: T1
            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment
    # RX - wait t - MZ
    qd_pulses = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # wait time before readout
    ro_wait_range = np.arange(
        delay_before_readout_start, delay_before_readout_end, delay_before_readout_step
    )

    sweeper = Sweeper(
        Parameter.delay,
        ro_wait_range,
        [qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the MSR, phase, i, q and the delay time
    data = DataUnits(
        name=f"data", quantities={"wait": "ns"}, options=["qubit", "iteration"]
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        # sweep the parameter
        results = platform.sweep(
            sequence, sweeper, nshots=nshots, relaxation_time=relaxation_time
        )
        for qubit in qubits:
            result = results[ro_pulses[qubit].serial]
            r = result.to_dict(average=False)
            r.update(
                {
                    "wait[ns]": ro_wait_range,
                    "qubit": len(ro_wait_range) * [qubit],
                    "iteration": len(ro_wait_range) * [iteration],
                }
            )
            data.add_data_from_dict(r)
        yield data

        yield t1_fit(
            data,
            x="wait[ns]",
            y="MSR[uV]",
            qubits=qubits,
            resonator_type=platform.resonator_type,
            labels=["T1"],
        )


@plot("MSR vs Time", plots.t1_time_msr)
def t1(
    platform: AbstractPlatform,
    qubits: list,
    delay_before_readout_start,
    delay_before_readout_end,
    delay_before_readout_step,
    software_averages=1,
):
    r"""
    In a T1 experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        delay_before_readout_start (int): Initial time delay before ReadOut
        delay_before_readout_end (list): Maximum time delay before ReadOut
        delay_before_readout_step (int): Scan range step for the delay before ReadOut
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **wait[ns]**: Delay before ReadOut used in the current execution
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **labels[0]**: T1
            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment
    # RX - wait t - MZ
    qd_pulses = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # wait time before readout
    ro_wait_range = np.arange(
        delay_before_readout_start, delay_before_readout_end, delay_before_readout_step
    )

    # create a DataUnits object to store the MSR, phase, i, q and the delay time
    data = DataUnits(
        name=f"data", quantities={"wait": "ns"}, options=["qubit", "iteration"]
    )

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(software_averages):
        # sweep the parameter
        for wait in ro_wait_range:
            # save data as often as defined by points
            if count % points == 0 and count > 0:
                # save data
                yield data
                # calculate and save fit
                yield t1_fit(
                    data,
                    x="wait[ns]",
                    y="MSR[uV]",
                    qubits=qubits,
                    resonator_type=platform.resonator_type,
                    labels=["T1"],
                )

            for qubit in qubits:
                ro_pulses[qubit].start = qd_pulses[qubit].duration + wait

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].average.raw
                r.update(
                    {
                        "wait[ns]": wait,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                    }
                )
                data.add(r)
            count += 1
    yield data
    yield t1_fit(
        data,
        x="wait[ns]",
        y="MSR[uV]",
        qubits=qubits,
        resonator_type=platform.resonator_type,
        labels=["T1"],
    )
