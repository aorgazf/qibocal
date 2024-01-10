"""Validation module."""
from dataclasses import dataclass, field
from typing import Callable, NewType, Optional, Union

from qibolab.qubits import QubitId, QubitPairId

from ..config import log, raise_error
from .operation import Results
from .status import Failure, Normal, Status
from .validators import VALIDATORS

ValidatorId = NewType("ValidatorId", str)
"""Identifier for validator object."""
Target = Union[QubitId, QubitPairId, list[QubitId]]
"""Protocol target."""
Outcome = tuple[str, Optional[dict]]
"""Validation outcome tuple of nodes with parameters or Status."""


@dataclass
class Validator:
    """Generic validator object."""

    scheme: ValidatorId
    """Validator present in validators module."""
    parameters: Optional[dict] = field(default_factory=dict)
    """"Validator parameters."""
    outcomes: Optional[list[Outcome]] = field(default_factory=list)
    """Depending on the validation we jump into one of the possible nodes."""

    # TODO: think of a better name
    @property
    def method(self) -> Callable[[Results, Target], Union[Status, str]]:
        """Validation function in validators module."""
        try:
            return VALIDATORS[self.scheme]
        except KeyError:
            raise_error(KeyError, f"Validator {self.scheme} not available.")

    def validate(
        self, results: Results, target: Target
    ) -> Union[Outcome, tuple[Status, None]]:
        """Perform validation of target in results.

        Possible Returns are:
            - (Failure, None) which stops the execution.
            - (Normal, None) which corresponds to the normal flow
            - (task, dict) which moves the head to task using parameters in dict.
        """
        index = self.method(results=results, target=target, **self.parameters)

        if index == -1:
            # -1 denotes Normal()
            return Normal(), None
        else:
            try:
                return self.outcomes[index]
            except (TypeError, IndexError):
                # TypeError to handle the case where index is None
                # IndexError to handle the case where index not in outcomes
                log.error("Stopping execution due to error in validation.")
                return Failure(), None
