"""Tasks execution."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union

from qibolab import create_platform
from qibolab.platform import Platform

from qibocal import protocols
from qibocal.config import log

from .history import History
from .mode import ExecutionMode
from .operation import Routine
from .runcard import Action, Id, Runcard, Targets
from .task import Completed, Task


@dataclass
class Executor:
    """Execute a tasks' graph and tracks its history."""

    actions: list[Action]
    """List of actions."""
    history: History
    """The execution history, with results and exit states."""
    output: Path
    """Output path."""
    targets: Targets
    """Qubits/Qubit Pairs to be calibrated."""
    platform: Platform
    """Qubits' platform."""
    update: bool = True
    """Runcard update mechanism."""

    # TODO: find a more elegant way to pass everything
    @classmethod
    def load(
        cls,
        card: Runcard,
        output: Path,
        platform: Platform,
        targets: Targets,
        update: bool = True,
    ):
        """Load execution graph and associated executor from a runcard."""

        return cls(
            actions=card.actions,
            history=History({}),
            output=output,
            platform=platform,
            targets=targets,
            update=update,
        )

    @property
    def _actions_dict(self):
        """Helper dict to find protocol."""
        return {action.id: action for action in self.actions}

    @classmethod
    def create(
        cls,
        platform: Union[Platform, str] = None,
        output: Union[str, bytes, os.PathLike] = None,
    ):
        """Load list of protocols."""
        platform = (
            platform if isinstance(platform, Platform) else create_platform(platform)
        )
        return cls(
            actions=[],
            history=History({}),
            output=Path(output),
            platform=platform,
            targets=list(platform.qubits),
            update=True,
        )

    def run_protocol(
        self,
        protocol: Routine,
        parameters: Union[dict, Action],
        mode: ExecutionMode = ExecutionMode.ACQUIRE | ExecutionMode.FIT,
    ) -> Completed:
        """Run single protocol in ExecutionMode mode."""
        if isinstance(parameters, dict):
            parameters["operation"] = str(protocol)
            action = Action(**parameters)
        else:
            action = parameters
        task = Task(action, protocol)
        if isinstance(mode, ExecutionMode):
            log.info(
                f"Executing mode {mode.name if mode.name is not None else 'AUTOCALIBRATION'} on {id}."
            )
        completed = task.run(
            platform=self.platform,
            targets=self.targets,
            folder=self.output,
            mode=mode,
        )

        if ExecutionMode.FIT in mode and self.platform is not None:
            completed.update_platform(platform=self.platform, update=self.update)

        self.history.push(completed)

        return completed

    def run(self, mode: ExecutionMode) -> Iterator[Id]:
        """Actual execution.

        The platform's update method is called if:
        - self.update is True and task.update is None
        - task.update is True
        """
        for id, params in self._actions_dict.items():
            completed = self.run_protocol(
                protocol=getattr(protocols, params.operation),
                parameters=params,
                mode=mode,
            )
            yield completed.task.id
