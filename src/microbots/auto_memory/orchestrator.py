"""Orchestrates the train <-> eval loop for repo-learning agents.

Repeatedly runs an ``EvalTask`` against a repo, and on failure builds
feedback and retrains via the training agent, looping until the task
passes or ``max_rounds`` is exhausted.
"""

import dataclasses
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from microbots.auto_memory.evalTask import EvalOutcome, EvalTask
from microbots.auto_memory.training.runner import run_training
from microbots.auto_memory.workdir import (
    RESULT_FILENAME,
    get_eval_dir,
    memory_dir,
    repo_dir,
    take_memory_snapshot,
)

logger = getLogger(__name__)

@dataclass
class LoopResult:
    """Result of running ``run_train_eval_loop``.

    Attributes
    ----------
    passed : bool
        Whether the task passed within ``max_rounds``.
    rounds_run : int
        Number of eval rounds actually run.
    final_outcome : EvalOutcome
        The outcome of the last round run.
    outcomes : list[EvalOutcome]
        The outcome of every round run, in order.
    """

    passed: bool
    rounds_run: int
    final_outcome: EvalOutcome
    outcomes: list[EvalOutcome] = field(default_factory=list)

def clone_repo(url: str, repo_path: Path) -> None:
    """Clone ``url`` into ``repo_path``, or reuse it if already cloned from ``url``.

    Existence alone isn't enough to trust ``repo_path``: it could be an
    empty/partial directory left by a previous failed clone, or a
    reused workdir whose config now points at a different ``url``. So
    if ``repo_path`` exists, its ``origin`` remote is checked against
    ``url`` first. Only an exact match is reused as-is; anything else
    (mismatched origin, or not a git checkout at all) is removed and
    re-cloned, so training never silently runs against missing or
    wrong code.

    Parameters
    ----------
    url : str
        Git URL (or local path) to clone from.
    repo_path : Path
        Destination directory for the clone.
    """
    if repo_path.exists():
        origin = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path, capture_output=True, text=True,
        )
        if origin.returncode == 0 and origin.stdout.strip() == url:
            return

        logger.warning(
            "clone_repo: %s exists but isn't a checkout of %s (origin=%r); "
            "removing and re-cloning",
            repo_path, url, origin.stdout.strip(),
        )
        shutil.rmtree(repo_path)

    subprocess.run(["git", "clone", url, str(repo_path)], check=True)

def write_eval_result(eval_dir: Path, outcome: EvalOutcome) -> None:
    """Write a round's eval result to ``result.json``.

    Parameters
    ----------
    eval_dir : Path
        The directory for this round's evaluation.
    outcome : EvalOutcome
        The round's outcome to persist.
    """
    path = eval_dir / RESULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(outcome), indent=2))


def run_train_eval_loop(
    training_repo_path: str,
    workdir: Path,
    model: str,
    task: EvalTask,
    max_rounds: int = 5,
) -> LoopResult:
    """Run an eval task in a loop, retraining on failure until it passes.

    Memory lives in one place (``workdir/memory``) and is mutated in
    place: each round snapshots it to
    ``rounds/round_N/starting_memory_snapshot`` before the eval agent
    reads it, so what the round began with stays recoverable. The task
    then evaluates against that memory in its own
    ``rounds/round_N/eval`` directory. Passing returns immediately;
    failing feeds ``outcome.feedback`` to ``run_training``, which
    rewrites memory for the next round. Either way the round's outcome
    is written to ``result.json``.

    An eval that raises is logged and recorded as a failed outcome so
    one bad round cannot discard the rounds before it.

    Parameters
    ----------
    training_repo_path : str
        Absolute path to the persistent checkout the training agent
        reads. Separate from the eval checkout, which the task clones
        and resets itself every round.
    workdir : Path
        This run's workdir (see ``microbots.auto_memory.workdir``).
    model : str
        The model to use, in the format ``<provider>/<model_name>``.
    task : EvalTask
        The eval task to run each round.
    max_rounds : int
        Maximum number of train/eval rounds to attempt. Defaults to 5.

    Returns
    -------
    LoopResult
        Whether the task passed, how many rounds ran, and every round's
        outcome.

    Raises
    ------
    ValueError
        If ``max_rounds`` is less than 1.
    """
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")

    outcomes: list[EvalOutcome] = []

    #TODO: Logs need to be saved to appropriate log files
    for round_idx in range(1, max_rounds+1):
        logger.info(
            "run_train_eval_loop: round %d/%d starting", round_idx, max_rounds
        )
        mem_dir = memory_dir(Path(workdir))
        eval_dir = get_eval_dir(workdir, round_idx)
        take_memory_snapshot(mem_dir, round_idx)
        try:
            outcome = task.eval(str(mem_dir), model, str(eval_dir))
            outcomes.append(outcome)
        except Exception as e:
            logger.warning(
                "run_train_eval_loop: round %d failed during evaluation;\n"
                "Exception: %s\n"
                "continuing to next round",
                round_idx, e
            )
            # Store the failure in outcomes
            outcomes.append(EvalOutcome(passed=False, score=-1, feedback=str(e)))
            continue

        try:
            if outcome.passed:
                logger.info(
                    "run_train_eval_loop: passed on round %d/%d", round_idx, max_rounds
                )
                return LoopResult(
                    passed=True,
                    rounds_run=round_idx,
                    final_outcome=outcome,
                    outcomes=outcomes,
                )

            logger.info(
                "run_train_eval_loop: round %d failed (%s), retraining",
                round_idx,
                outcome.feedback,
            )

            try:
                run_training(
                    repo_path=training_repo_path,
                    feedback=outcome.feedback,
                    memory_dir=str(mem_dir),
                    model=model,
            )
            except Exception:
                logger.exception(
                    "run_train_eval_loop: round %d failed to build feedback/retrain; "
                    "continuing to next round without retraining",
                    round_idx,
                )
        finally:
            write_eval_result(eval_dir, outcome)

    logger.info(
        "run_train_eval_loop: exhausted %d rounds without passing", max_rounds
    )
    return LoopResult(
        passed=False,
        rounds_run=max_rounds,
        final_outcome=outcomes[-1],
        outcomes=outcomes,
    )

def run(
    workdir: Path,
    model: str,
    task: EvalTask,
    max_rounds: int = 5,
) -> LoopResult:
    """Clone the task's training repo, then run the full train/eval loop.

    Parameters
    ----------
    workdir : Path
        This run's workdir (see ``microbots.auto_memory.workdir``),
        holding the training clone, memory, and all round output.
    model : str
        The model to use, in the format ``<provider>/<model_name>``.
    task : EvalTask
        The eval task to run each round. It also supplies the training
        repo's clone URL via ``repo_url()``.
    max_rounds : int
        Maximum number of train/eval rounds to attempt. Defaults to 5.

    Returns
    -------
    LoopResult
        The eval loop's result.
    """
    training_repo_dir = repo_dir(workdir)
    clone_repo(task.repo_url(), training_repo_dir)
    training_repo_path = str(training_repo_dir)

    # TODO: train-only mode will be implemented if required after proper design
    # if task is None:
    #     # Train-only mode has no rounds of its own; round 1 is just a
    #     # scratch dir seeded from (and saved back to) top-level memory.
    #     memory_dir = str(load_round_memory(workdir, 1))
    #     run_training_loop(
    #         repo_path=training_repo_path,
    #         feedback="",
    #         memory_dir=memory_dir,
    #         model=model,
    #         iterations=training_iterations,
    #     )
    #     save_round_memory(workdir, 1)
    #     return None

    return run_train_eval_loop(
        training_repo_path=training_repo_path,
        workdir=workdir,
        model=model,
        task=task,
        max_rounds=max_rounds,
    )