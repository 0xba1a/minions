"""Path/layout helpers for a training run's workdir.

Centralizes every path this package reads or writes under a run's
``workdir`` (config, repo clone, logs, memory, and per-round/per-eval
outputs), so callers never hard-code layout details themselves.
"""

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
EVAL_DIRNAME = "eval"
EVAL_LOG_DIRNAME = "logs"
LOG_FILENAME = "log.txt"
TRAINING_LOG_FILENAME = "training_log.txt"
RESULT_FILENAME = "result.json"

"""
Expected workdir structure:

   workdir/
   ├── task_config.yaml
   ├── log.txt                                <-- CLI and orchestrator logs
   ├── repo/                                  <-- Training checkout, reused across rounds
   ├── memory/                                <-- Mutated in place; the run's living memory
   └── rounds/round_n/
       ├── training_log.txt                   <-- Training agent logs for this round
       ├── eval/                              <-- Managed by the eval task
       │   ├── result.json
       │   ├── eval_repo/
       │   └── logs/
       │       └── <instance_id>_log.txt      <-- One log per eval instance
       └── starting_memory_snapshot/          <-- memory/ as it looked when the round began
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
    """Prepare ``workdir`` for a fresh run, archiving any previous one.

    A run must start from clean round output, so an existing workdir is
    archived to ``<name>_backup_<timestamp>`` and recreated. The config,
    memory, and training checkout are carried over from the archive so
    the next run resumes from what the last one learned instead of
    re-cloning and re-learning from scratch.

    Parameters
    ----------
    workdir : Path
        The workdir to prepare. Created if it does not exist.
    """
    if workdir.exists():
        backup = workdir.parent / f"{workdir.name}_backup_{int(time.time())}"
        shutil.move(str(workdir), str(backup))
        workdir.mkdir(parents=True)
        # Moved rather than copied; repo/ can be hundreds of MB.
        for name in (CONFIG_FILENAME, MEMORY_DIRNAME, REPO_DIRNAME):
            carried_over = backup / name
            if carried_over.exists():
                shutil.move(str(carried_over), str(workdir / name))

    # Every round snapshots this, so it must exist even when empty.
    (workdir / MEMORY_DIRNAME).mkdir(parents=True, exist_ok=True)


def config_path(workdir: Path) -> Path:
    """Return the path to ``workdir``'s config file.

    Parameters
    ----------
    workdir : Path
        The run's workdir.

    Returns
    -------
    Path
        ``workdir/task_config.yaml``.
    """
    return workdir / CONFIG_FILENAME


def log_path(workdir: Path) -> Path:
    """Return the path to the run's top-level log file.

    Parameters
    ----------
    workdir : Path
        The run's workdir.

    Returns
    -------
    Path
        ``workdir/log.txt``.
    """
    return workdir / LOG_FILENAME


def repo_dir(workdir: Path) -> Path:
    """Return the path to the training checkout shared across rounds.

    Used only by the retraining step. Eval tasks clone and manage their
    own checkout under the round's eval directory, so the two never
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
    """Snapshot the memory dir as this round's starting point.

    Memory is mutated in place across rounds, so this preserves what
    the round started from before training rewrites it.

    Parameters
    ----------
    mem_dir : Path
        The run's top-level memory directory.
    round_idx : int
        The 1-based round number.

    Raises
    ------
    FileNotFoundError
        If ``mem_dir`` does not exist. It may be empty, but the run's
        layout must already have created it.
    """
    if not mem_dir.exists():
        raise FileNotFoundError(f"Memory directory does not exist: {mem_dir}")

    snapshot_dir = round_dir(mem_dir.parent, round_idx) / STARTING_MEMORY_SNAPSHOT_DIR
    if snapshot_dir.exists():
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


def training_log_path(workdir: Path, round_num: int) -> Path:
    """Return the path to a round's training log file.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number.

    Returns
    -------
    Path
        ``workdir/rounds/round_{round_num}/training_log.txt``.
    """
    return round_dir(workdir, round_num) / TRAINING_LOG_FILENAME


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


def eval_log_dir(eval_dir: Path) -> Path:
    """Return the directory holding one log file per eval instance. Creates it if missing.

    Parameters
    ----------
    eval_dir : Path
        The round's eval directory, as returned by ``get_eval_dir``.

    Returns
    -------
    Path
        ``<eval_dir>/logs``.
    """
    path = eval_dir / EVAL_LOG_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path
