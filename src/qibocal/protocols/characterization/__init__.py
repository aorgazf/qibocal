from enum import Enum

from .classification import single_shot_classification
from .qubit_spectroscopy import qubit_spectroscopy
from .rabi.amplitude import rabi_amplitude
from .rabi.length import rabi_length
from .ramsey import ramsey
from .resonator_flux_dependence import resonator_flux
from .resonator_punchout import resonator_punchout
from .resonator_spectroscopy import resonator_spectroscopy
from .t1 import t1


class Operation(Enum):
    resonator_spectroscopy = resonator_spectroscopy
    resonator_punchout = resonator_punchout
    resonator_flux = resonator_flux
    qubit_spectroscopy = qubit_spectroscopy
    rabi_amplitude = rabi_amplitude
    rabi_length = rabi_length
    ramsey = ramsey
    t1 = t1
    single_shot_classification = single_shot_classification
