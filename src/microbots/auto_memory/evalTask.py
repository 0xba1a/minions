"""Defines the abstract eval task interface for the train <-> eval loop.

An ``EvalTask`` describes one unit of work: how to prepare a repo, what
prompt to give the agent, how to verify the agent's output, and how to
clean up afterward.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalOutcome:
    """Result of verifying whether an eval task was completed correctly.

    Attributes
    ----------
    passed : bool
        Whether the agent's output satisfies the task's check.
    score : float
        A numeric score representing the quality of the agent's output.
    feedback : str
        A short human-readable explanation of the pass/fail verdict.
    """

    passed: bool
    score: float
    feedback: str

class EvalTask(ABC):
    """Base class for a single evaluation task in the train <-> eval loop.

    Subclasses must implement ``run``. ``parse_config``, ``setup``,
    ``check``, and ``teardown`` are optional hooks
    subclasses may use to structure their own ``run`` implementation
    (see ``SweBenchVerifiedTask`` for an example), but nothing in this
    base class calls them automatically.
    """

    def __init__(self, config_file: Path) -> None:
        super().__init__()

    @abstractmethod
    def repo_url(self) -> str:
        """Return the URL of the repo for the training agent."""

    @property
    def task_id(self) -> str:
        """Identifier for this task instance, used to name its output folder.

        Defaults to the class name, which is fine for tasks with only
        one instance per run. Override for tasks with several distinct
        instances per class (e.g. ``SweBenchVerifiedTask``, where each
        dataset row needs its own folder).

        Returns
        -------
        str
            This task instance's identifier.
        """
        return type(self).__name__

    def setup(self) -> None:
        """Optional. Prepare repo/environment before the agent runs.

        Not called automatically; only useful if your ``run``
        implementation calls it.

        Parameters
        ----------
        repo_path : str
            Absolute path to the repo to prepare.
        """
        pass

    def teardown(self, eval_repo_path: Path) -> None:
        """Optional. Clean up anything setup() created.

        Parameters
        ----------
        eval_repo_path : Path
            Absolute path to the repo that was prepared by ``setup``.
        """
        pass

    @abstractmethod
    def parse_config(self, config_file: Path) -> None:
        """Parse the task-specific config file. Importantly it
        parses the config file and get the repo for the training
        agent.

        Parameters
        ----------
        config_file : Path
            Path to the config file to parse.
        """

    @abstractmethod
    def eval(self, memory_dir: str, model: str, log_path: str) -> EvalOutcome:
        """Required. Run one eval iteration and return its outcome.

        Parameters
        ----------
        memory_dir : str
            Directory containing memory files to give the agent via
            ``MemoryTool``.
        model : str
            The model to use, in the format ``<provider>/<model_name>``.
        log_path : str
            Path to write this round's log to. Caller-provided (e.g. a
            workdir-managed path) so logs persist under the run's
            layout instead of each task inventing its own temp file.

        Returns
        -------
        EvalOutcome
            The result of this eval round, including the agent's output,
            the check verdict.
        """
