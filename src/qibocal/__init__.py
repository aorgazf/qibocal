"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

import importlib.metadata as im

from .cli import acquire, command, fit, report, upload

__version__ = im.version(__package__)
