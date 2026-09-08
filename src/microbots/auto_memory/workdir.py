"""Path/layout helpers for a training run's workdir.

Centralizes every path this package reads or writes under a run's
``workdir`` (config, repo clone, logs, memory, and per-round/per-eval
outputs), so callers never hard-code layout details themselves.
"""

import os
import shutil
import time
from pathlib import Path

WORKDIR_NAME = "workdir"
CONFIG_FILENAME = "task_config.yaml"
REPO_DIRNAME = "repo"
MEMORY_DIRNAME = "memory"
"""
At the beginning of the round, memory from memory_dir will be snapshotted
inside the workdir/rounds/rounds_n/starting_memory_snapshot directory.
So, the final memory after that round should be found at rounds_(n+1)
directory. The memory of the final round will be available at the
main memory dir workdir/memory
"""
STARTING_MEMORY_SNAPSHOT_DIR = "starting_memory_snapshot"
ROUNDS_DIRNAME = "rounds"
ROUND_LOG_DIR= "logs"
EVAL_DIRNAME = "eval"
RESULT_FILENAME = "result.json"

"""
Expected workdir structure:

  workdir/
  |
   workdir/
   ├── task_config.yaml
   ├── repo/
   ├── memory/
   ├── rounds/round_n/
   │   └── logs/  <-- Contains only training log. Eval logs can be found inside the eval task
   │   └── eval/  <-- Managed by the eval task
   │   └── starting_memory_snapshot/
   └── logs/
"""


def resolve_workdir(base: Path | None = None) -> Path:
    """Resolve the fixed workdir path relative to ``base``.

    Parameters
    ----------
    base : Path | None
        Directory to resolve ``workdir/`` relative to. Defaults to the
        current working directory.

    Returns
    -------
    Path
        ``workdir`` resolved relative to ``base`` (or ``Path.cwd()``).
    """
    workdir = (base or Path.cwd()) / WORKDIR_NAME
    return workdir


def require_workdir(workdir: Path) -> None:
    """Validate that ``workdir`` exist. Create if not exist

    Parameters
    ----------
    workdir : Path
        The workdir to validate.

    """
    # create workdir if not existing
    if not workdir.exists():
        mem_dir = workdir / MEMORY_DIRNAME
        mem_dir.mkdir(parents=True)
        return

    workdir_parent = workdir.parent
    workdir_backup = workdir_parent / f"{workdir.name}_backup_{int(time.time())}"
    shutil.move(str(workdir), str(workdir_backup))
    workdir.mkdir(parents=True)

    task_config = workdir_backup / CONFIG_FILENAME
    if task_config.exists():
        shutil.copy(task_config, workdir / CONFIG_FILENAME)
    else:
        raise FileNotFoundError(f"Task config file does not exist: {task_config}")

    mem_dir = workdir_backup / MEMORY_DIRNAME
    if mem_dir.exists():
        shutil.copytree(mem_dir, workdir / MEMORY_DIRNAME)
    else:
        mem_dir.mkdir(parents=True)

    repo_dir = workdir_backup / REPO_DIRNAME
    if repo_dir.exists():
        shutil.copytree(repo_dir, workdir / REPO_DIRNAME)


def config_path(workdir: Path) -> Path:
    """Return the path to ``workdir``'s config file.

    Parameters
    ----------
    workdir : Path
        The run's workdir.

    Returns
    -------
    Path
        ``workdir/config.yaml``.
    """
    return workdir / CONFIG_FILENAME


def repo_dir(workdir: Path) -> Path:
    """Return the path to the single cloned repo shared across rounds.

    Used only for training (both train-only mode and the eval loop's
    retrain step): a persistent checkout that stays in place across
    rounds. Eval tasks that manage their own repo checkout (e.g.
    ``SweBenchVerifiedTask``, which clones a different repo/commit per
    dataset instance) use ``eval_repo_dir`` instead, so the two never
    collide.

    Parameters
    ----------
    workdir : Path
        The run's workdir.

    Returns
    -------
    Path
        ``workdir/repo``.
    """
    return workdir / REPO_DIRNAME


def memory_dir(workdir: Path) -> Path:
    """Return the path to the current top-level (latest) memory directory.

    Parameters
    ----------
    workdir : Path
    The run's workdir.

    Returns
    -------
    Path
        ``workdir/memory``.
    """
    return workdir / MEMORY_DIRNAME


def take_memory_snapshot(mem_dir: Path, round_idx: int) -> None:
    """Take a snapshot of the current memory directory for a specific round.

    Parameters
    ----------
    mem_dir : Path
        The path to the current top-level memory directory.
    round_idx : int
        The 1-based round number.
    """
    if not mem_dir.exists():
        # At this point, mem_dir can be empty but it should exist
        raise FileNotFoundError(f"Memory directory does not exist: {mem_dir}")

    workdir = Path(mem_dir).parent
    snapshot_dir = round_dir(workdir, round_idx) / STARTING_MEMORY_SNAPSHOT_DIR
    if os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir)
    shutil.copytree(mem_dir, snapshot_dir)


def round_dir(
    workdir: Path, round_num: int) -> Path:
    """Create and return the directory for a training round.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number.

    Returns
    -------
    Path
        ``workdir/rounds/round_{round_num}``
    """
    path = workdir / ROUNDS_DIRNAME / f"round_{round_num}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_log_dir(workdir: Path, round_num: int) -> Path:
    """Return the path to a round's training log.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number.

    Returns
    -------
    Path
        This round's log directory. Both training and eval logs
        can be found inside with appropriate file names
    """
    return round_dir(workdir, round_num) / ROUND_LOG_DIR


def get_eval_dir(
    workdir: Path, round_num: int) -> Path:
    """Return the eval task instance's eval directory. Creates it if missing.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number this eval instance belongs to.

    Returns
    -------
    Path
        ``workdir/rounds/round_{round_num}/eval``.
    """
    path = round_dir(workdir, round_num) / EVAL_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path
