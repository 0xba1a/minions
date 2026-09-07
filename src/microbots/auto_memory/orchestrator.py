"""Orchestrates the train <-> eval loop for repo-learning agents.

Repeatedly runs an ``EvalTask`` against a repo, and on failure builds
feedback and retrains via the training agent, looping until the task
passes or ``max_rounds`` is exhausted.
"""

from dataclasses import dataclass, field
import dataclasses
from logging import getLogger
from pathlib import Path
import json
import shutil
import subprocess

from microbots.auto_memory.evalTask import EvalOutcome, EvalTask
from microbots.auto_memory.training.runner import run_training
from microbots.auto_memory.workdir import (
    eval_log_path,
    eval_result_path,
    load_round_memory,
    repo_dir,
    save_round_memory,
    snapshot_seed_memory,
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

def write_eval_result(workdir: Path, round_num: int, task: EvalTask, outcome: EvalOutcome) -> None:
    """Write a round's eval result to ``result.json``.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number this outcome belongs to.
    task : EvalTask
        The task that produced ``outcome``, used for both its
        ``task_id`` (folder name) and ``build_result`` (file content).
    outcome : EvalOutcome
        The round's outcome to persist.
    """
    path = eval_result_path(workdir, round_num, task.task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(outcome), indent=2))

def run_training_loop(
    repo_path: str,
    feedback: str,
    memory_dir: str,
    model: str,
    iterations: int = 10,
) -> None:
    """Run ``run_training`` ``iterations`` times, reusing the same memory dir.

    Shared by the eval-loop's retrain step and any training-only entry
    point (e.g. a CLI) that needs to run training without an eval task.

    Parameters
    ----------
    repo_path : str
        Absolute path to the repo to train against.
    feedback : str
        Feedback from a prior failed eval attempt, or ``""`` if none.
    memory_dir : str
        Directory where the training agent reads/writes memory files.
    model : str
        The model to use, in the format ``<provider>/<model_name>``.
    iterations : int
        Number of training passes to run, each reusing the same
        ``memory_dir``. Defaults to 10.
    """
    for iteration in range(1, iterations + 1):
        logger.info(
            "run_training_loop: training iteration %d/%d",
            iteration,
            iterations,
        )
        run_training(
            repo_path=repo_path,
            feedback=feedback,
            memory_dir=memory_dir,
            model=model,
        )

def run_train_eval_loop(
    training_repo_path: str,
    workdir: Path,
    model: str,
    task: EvalTask,
    max_rounds: int = 5,
    training_iterations: int = 10,
) -> LoopResult:
    """Run an eval task in a loop, retraining on failure until it passes.

    Each round loads the current top-level memory into its own
    ``rounds_<task_id>/round_N/memory`` (carried forward from the
    previous round, or empty on round 1), then runs ``task.run(...)``
    against it, writing its log to a workdir-managed path
    (``rounds_<task_id>/round_N/eval/eval.log``) so it persists. Since
    each eval task instance gets its own ``rounds_<task_id>`` dir,
    different instances sharing the same ``workdir`` never collide on
    round numbers, and each instance's per-round memory is preserved
    individually. If the task passes, the loop returns immediately. If
    it fails, feedback is built from the round's log and used to
    retrain via ``run_training`` (called ``training_iterations`` times,
    each pass reusing the same round memory dir) before the next round.
    Either way, the round's result is written to ``result.json`` and
    its memory is saved back to the top-level memory dir before the
    next round starts.

    Parameters
    ----------
    training_repo_path : str
        Absolute path to the persistent repo checkout used only for
        retraining (``run_training_loop``). Kept separate from
        ``eval_repo_path`` since the task manages the latter's
        lifecycle itself (clone/teardown each round).
    eval_repo_path : str
        Absolute path to the repo the task clones/manages itself (via
        its own ``setup``) and runs/checks the agent against each
        round.
    workdir : Path
        This run's workdir, used to carry memory forward between rounds
        (see ``microbots.auto_memory.workdir``).
    model : str
        The model to use, in the format ``<provider>/<model_name>``.
    task : EvalTask
        The eval task to run each round.
    max_rounds : int
        Maximum number of train/eval rounds to attempt. Defaults to 5.
    training_iterations : int
        Number of training passes to run per retraining round, each
        reusing the same round memory dir. Defaults to 10.

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

    for round_idx in range(1, max_rounds+1):
        logger.info(
            "run_train_eval_loop: round %d/%d starting", round_idx, max_rounds
        )
        # TODO: Instead of loading new memory dir on every iteration, snapshot the memory.
        memory_dir = str(load_round_memory(workdir, round_idx, instance_id=task.task_id))
        log_path = str(eval_log_path(workdir, round_idx, task.task_id))
        outcome = task.eval(memory_dir, model, log_path)
        outcomes.append(outcome)

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
                run_training_loop(
                    repo_path=training_repo_path,
                    feedback=outcome.feedback,
                    memory_dir=memory_dir,
                    model=model,
                    iterations=training_iterations,
                )
            except Exception:
                logger.exception(
                    "run_train_eval_loop: round %d failed to build feedback/retrain; "
                    "continuing to next round without retraining",
                    round_idx,
                )
        finally:
            write_eval_result(workdir, round_idx, task, outcome)
            save_round_memory(workdir, round_idx, instance_id=task.task_id)

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
    training_iterations: int = 10,
) -> LoopResult:
    """Run full train/eval loop, depending on ``task``.

    Parameters
    ----------
    workdir : Path
        This run's workdir (see ``microbots.auto_memory.workdir``),
        holding ``task_config.yaml``, the shared repo clone, and all output.
    model : str
        The model to use, in the format ``<provider>/<model_name>``.
    task : EvalTask
        The eval task to run each round.
    max_rounds : int
        Maximum number of train/eval rounds to attempt. Defaults to 5.
    training_iterations : int
        Number of training passes to run per retraining round, each
        reusing the same round memory dir. Defaults to 10.

    Returns
    -------
    LoopResult
        The eval loop's result.

    Raises
    ------
    ValueError
        If ``config`` has no ``repo`` entry. Every run needs a training
        checkout (``run_training_loop`` always mounts
        ``training_repo_path``, regardless of ``task``), so ``repo``
        must be configured even for tasks like ``SweBenchVerifiedTask``
        that manage their own separate eval checkout.
    """
    clone_repo(task.repo_url(), repo_dir(workdir))

    snapshot_seed_memory(workdir)

    training_repo_path = str(repo_dir(workdir))

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
        training_iterations=training_iterations,
    )
