"""Defines the abstract eval task interface for the train <-> eval loop.

An ``EvalTask`` owns its own config, names the repo the training agent
should learn from, and runs one complete evaluation per round.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class EvalOutcome:
    """Result of one round's evaluation.

    Attributes
    ----------
    passed : bool
        Whether every unit of work in the round passed.
    score : float
        Fraction of units that passed, or ``-1`` if the round errored.
    feedback : str
        Text describing what went wrong, fed to the next round's
        training pass.
    """

    passed: bool
    score: float
    feedback: str

class EvalTask(ABC):
    """Base class for an evaluation task in the train <-> eval loop.

    Subclasses must implement ``eval``. ``parse_config`` and
    ``repo_url`` have working defaults driven by the config file's
    ``repo`` key, and may be overridden by tasks that derive the repo
    some other way (see ``SweBenchVerified``).
    """

    _repo_url: str | None = None

    def __init__(self, config_file: Path) -> None:
        # NOTE: Don't call this from child class unless you need to reuse
        # the parse_config logic from here.
        super().__init__()
        self.parse_config(config_file=config_file)

    def repo_url(self) -> str:
        """Return the URL of the repo the training agent should learn from.

        Returns
        -------
        str
            The training repo's clone URL.

        Raises
        ------
        ValueError
            If the config had no ``repo`` key and the subclass did not
            override this method.
        """
        if not self._repo_url:
            raise ValueError("Repo URL is not set in the config file."
                             " Or you didn't override the base method.")
        return self._repo_url

    def parse_config(self, config_file: Path) -> None:
        """Read the task's config file, recording the training repo URL.

        Called from ``__init__``. Subclasses that need more than the
        ``repo`` key override this to load their own settings too.

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
        """Required. Run one full evaluation and return its outcome.

        Parameters
        ----------
        memory_dir : str
            Directory containing memory files to give the agent via
            ``MemoryTool``.
        model : str
            The model to use, in the format ``<provider>/<model_name>``.
        eval_dir: str
            Directory this round's eval owns. The task decides what
            goes in it (cloned repo, logs, and so on).

        Returns
        -------
        EvalOutcome
            Whether the round passed, its score, and the feedback to
            retrain on.
        """
