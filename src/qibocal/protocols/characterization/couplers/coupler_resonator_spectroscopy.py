from typing import Optional

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Qubits, Routine

from ..flux_dependence.utils import flux_dependence_plot
from ..two_qubit_interaction.utils import order_pair
from .utils import (
    CouplerSpectroscopyData,
    CouplerSpectroscopyParameters,
    CouplerSpectroscopyResults,
)


class CouplerSpectroscopyParameters(CouplerSpectroscopyParameters):
    readout_delay: Optional[int] = 1000
    """Readout delay before the measurament is done to let the flux coupler pulse act"""


def _acquisition(
    params: CouplerSpectroscopyParameters, platform: Platform, qubits: Qubits
) -> CouplerSpectroscopyData:
    """
    Data acquisition for CouplerResonator spectroscopy.

    This consist on a frequency sweep on the readout frequency while we change the flux coupler pulse amplitude of
    the coupler pulse. We expect to enable the coupler during the amplitude sweep and detect an avoided crossing
    that will be followed by the frequency sweep. No need to have the qubits at resonance. This should be run after
    resonator_spectroscopy to detect couplers and adjust the coupler sweetspot if needed and get some information
    on the flux coupler pulse amplitude requiered to enable 2q interactions.

    """

    """
    We may want to measure both qubits on the pair,
    that will require a different acquisition, for now I suggest to only measure one and reduce possible crosstalk.
    Zurich crashes on several pairs due to the single oscillator issue.
    """

    # create a sequence of pulses for the experiment:
    # Coupler pulse while MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    fx_pulses = {}
    couplers = []

    for i, pair in enumerate(qubits):
        qubit = platform.qubits[params.measured_qubit[i]].name
        # TODO: Qubit pair patch
        ordered_pair = order_pair(pair, platform.qubits)
        coupler = platform.pairs[tuple(sorted(ordered_pair))].coupler
        couplers.append(coupler)

        # TODO: Does it need time or can it start at 0 ???
        # TODO: May measure both qubits on the pair
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=params.readout_delay
        )
        if params.amplitude is not None:
            ro_pulses[qubit].amplitude = params.amplitude

        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )

    sweeper_freq = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in params.measured_qubit],
        type=SweeperType.OFFSET,
    )

    # define the parameter to sweep and its range:
    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )

    """This sweeper is implemented in the flux pulse amplitude and we need it to be that way. """
    sweeper_bias = Sweeper(
        Parameter.bias,
        delta_bias_range,
        couplers=couplers,
        type=SweeperType.ABSOLUTE,
    )

    data = CouplerSpectroscopyData(
        resonator_type=platform.resonator_type,
    )

    results = platform.sweep(
        sequence,
        ExecutionParameters(
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        ),
        sweeper_bias,
        sweeper_freq,
    )

    # retrieve the results for every qubit
    for i, pair in enumerate(qubits):
        # TODO: May measure both qubits on the pair
        qubit = platform.qubits[params.measured_qubit[i]].name
        # average msr, phase, i and q over the number of shots defined in the runcard
        result = results[ro_pulses[qubit].serial]
        # store the results
        data.register_qubit(
            qubit,
            msr=result.magnitude,
            phase=result.phase,
            freq=delta_frequency_range + ro_pulses[qubit].frequency,
            bias=delta_bias_range,
        )
    return data


def _fit(data: CouplerSpectroscopyData) -> CouplerSpectroscopyResults:
    """Post-processing function for CouplerResonatorSpectroscopy."""
    qubits = data.qubits
    pulse_amp = {}
    sweetspot = {}
    fitted_parameters = {}

    for qubit in qubits:
        # TODO: Implement fit
        """It should get two things:
        Coupler sweetspot: the value that makes both features centered and symmetric
        Pulse_amp: That turn on the feature taking into account the shift introduced by the coupler sweetspot

        Issues:  Coupler sweetspot it measured in volts while pulse_amp is a pulse amplitude, this routine just sweeps pulse amplitude
        and relies on manual shifting of that sweetspot by repeated scans as current chips are already symmetric for this feature.
        Maybe another routine sweeping the bias in volts would be needed and that sweeper implement on Zurich driver.
        """
        # spot, amp, fitted_params = coupler_fit(data[qubit])

        sweetspot[qubit] = 0
        pulse_amp[qubit] = 0
        fitted_parameters[qubit] = {}

    return CouplerSpectroscopyResults(
        pulse_amp=pulse_amp,
        sweetspot=sweetspot,
        fitted_parameters=fitted_parameters,
    )


def _plot(
    data: CouplerSpectroscopyData,
    qubit,
    fit: CouplerSpectroscopyResults,
):
    """
    We may want to measure both qubits on the pair,
    that will require a different plotting that takes both.
    """
    qubit_pair = qubit  # TODO: Patch for 2q gate routines

    for qubit in qubit_pair:
        if qubit in data.data.keys():
            return flux_dependence_plot(data, fit, qubit)


def _update(results: CouplerSpectroscopyResults, platform: Platform, qubit: QubitId):
    pass


coupler_resonator_spectroscopy = Routine(_acquisition, _fit, _plot, _update)
"""CouplerResonatorSpectroscopy Routine object."""
