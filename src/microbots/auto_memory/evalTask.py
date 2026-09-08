"""Defines the abstract eval task interface for the train <-> eval loop.

An ``EvalTask`` describes one unit of work: how to prepare a repo, what
prompt to give the agent, how to verify the agent's output, and how to
clean up afterward.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import yaml


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
        self.parse_config(config_file=config_file)

    def repo_url(self) -> str:
        """Return the URL of the repo for the training agent."""
        if not self._repo_url:
            raise ValueError("Repo URL is not set in the config file."
                             " Or you didn't override the base method.")
        return self._repo_url

    def parse_config(self, config_file: Path) -> None:
        """Parse the task-specific config file. Importantly it
        parses the config file and get the repo for the training
        agent.

        Parameters
        ----------
        config_file : Path
            Path to the config file to parse.
        """
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            self._repo_url = config.get("repo")

    @abstractmethod
    def eval(self, memory_dir: str, model: str, eval_dir: str) -> EvalOutcome:
        """Required. Run one eval iteration and return its outcome.

        Parameters
        ----------
        memory_dir : str
            Directory containing memory files to give the agent via
            ``MemoryTool``.
        model : str
            The model to use, in the format ``<provider>/<model_name>``.
        eval_dir: str
            Path to run this round's eval. This directory is managed by
            the eval task itself. It can have its cloned repo, logs, etc.

        Returns
        -------
        EvalOutcome
            The result of this eval round, including the agent's output,
            the check verdict.
        """
