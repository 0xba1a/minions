"""Path/layout helpers for a training run's workdir.

Centralizes every path this package reads or writes under a run's
``workdir`` (config, repo clone, logs, memory, and per-round/per-eval
outputs), so callers never hard-code layout details themselves.
"""

from pathlib import Path
import shutil

import yaml

WORKDIR_NAME = "workdir"
CONFIG_FILENAME = "config.yaml"
REPO_DIRNAME = "repo"
EVAL_REPO_DIRNAME = "eval_repo"
MEMORY_DIRNAME = "memory"
MEMORY_SEED_DIRNAME = "memory_seed"
ROUNDS_DIRNAME = "rounds"
ROUND_LOG_FILENAME = "round.log"
ROUND_PATCH_FILENAME = "repo.patch"
EVAL_DIRNAME = "eval"
RESULT_FILENAME = "result.json"
EVAL_LOG_FILENAME = "eval.log"


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
    if base is not None and not base.is_absolute():
        raise ValueError(f"base must be an absolute path: {base}")
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
        workdir.mkdir(parents=True)


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


def load_config(workdir: Path) -> dict:
    """Load and parse ``workdir``'s config file.

    Parameters
    ----------
    workdir : Path
        The workdir whose config file should be loaded.

    Returns
    -------
    dict
        The parsed config, or ``{}`` if the config file doesn't exist
        or is empty.
    """
    path = config_path(workdir)
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


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


def snapshot_seed_memory(workdir: Path) -> Path:
    """Snapshot the current top-level memory dir as the run's restorable baseline.

    ``memory_dir`` is shared and mutated in place across every
    training/eval round and every eval task instance (so later
    instances benefit from what earlier ones learned), which means the
    original, pre-run memory is otherwise overwritten and lost with no
    way to get back to it. Call this once, before anything trains,
    to preserve that original state at ``workdir/memory_seed``. A
    no-op if a snapshot already exists, so later calls (e.g. once per
    eval task instance in the same run) never clobber the very first
    snapshot with already-mutated memory.

    Parameters
    ----------
    workdir : Path
        The run's workdir.

    Returns
    -------
    Path
        ``workdir/memory_seed``, containing a copy of whatever
        ``memory_dir`` held the first time this was called (or empty,
        if there was no pre-existing memory).
    """
    dst = workdir / MEMORY_SEED_DIRNAME
    if dst.exists():
        return dst
    src = memory_dir(workdir)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)
    return dst


def round_dir(
    workdir: Path, round_num: int, *, instance_id: str | None = None, create: bool = False
) -> Path:
    """Return (and optionally create) the directory for a training round.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number.
    instance_id : str | None
        If given, rounds are kept under a per-instance rounds dir
        (``rounds_{instance_id}``) instead of the shared ``rounds`` dir,
        so different eval task instances sharing the same ``workdir``
        don't collide on round numbers. Pass the eval task's
        ``task_id`` when running an eval task; omit for training-only
        mode.
    create : bool
        If True, create the directory (and parents) if missing.

    Returns
    -------
    Path
        ``workdir/rounds/round_{round_num}`` (no ``instance_id``), or
        ``workdir/rounds_{instance_id}/round_{round_num}``.
    """
    rounds_dirname = f"{ROUNDS_DIRNAME}_{instance_id}" if instance_id else ROUNDS_DIRNAME
    path = workdir / rounds_dirname / f"round_{round_num}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def round_memory_dir(workdir: Path, round_num: int, *, instance_id: str | None = None) -> Path:
    """Return the path to a round's own memory snapshot (a directory).

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number.
    instance_id : str | None
        The eval task's ``task_id``, if running an eval task (see
        ``round_dir``). Omit for training-only mode.

    Returns
    -------
    Path
        This round's own memory directory.
    """
    return round_dir(workdir, round_num, instance_id=instance_id) / MEMORY_DIRNAME


def load_round_memory(workdir: Path, round_num: int, *, instance_id: str | None = None) -> Path:
    """Copy the current top-level memory into this round's own memory dir.

    Called before a round's training pass, so it starts from whatever
    memory the previous round left behind (or empty, on round 1). This
    round's memory dir is replaced, not merged into: any stale files
    left behind by a previous attempt at this same round (e.g. a
    crashed/re-run process) are discarded first, so the round always
    starts from an exact snapshot of the current top-level memory.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number to load memory into.
    instance_id : str | None
        The eval task's ``task_id``, if running an eval task (see
        ``round_dir``). Omit for training-only mode.

    Returns
    -------
    Path
        This round's own memory dir, ready for the round to use.
    """
    src = memory_dir(workdir)
    dst = round_memory_dir(workdir, round_num, instance_id=instance_id)
    shutil.rmtree(dst, ignore_errors=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)
    return dst


def save_round_memory(workdir: Path, round_num: int, *, instance_id: str | None = None) -> Path:
    """Copy this round's memory back up to the top-level memory dir.

    Called after a round's training pass, so later rounds (and the
    final saved memory) see what this round learned. The top-level
    memory dir is replaced, not merged into: files the round deleted
    (e.g. via the agent's ``memory delete`` command) are gone from
    the top level too, instead of surviving from a previous save.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number whose memory should be saved.
    instance_id : str | None
        The eval task's ``task_id``, if running an eval task (see
        ``round_dir``). Omit for training-only mode.

    Returns
    -------
    Path
        The top-level ``memory`` dir, now updated with this round's changes.
    """
    src = round_memory_dir(workdir, round_num, instance_id=instance_id)
    dst = memory_dir(workdir)
    shutil.rmtree(dst, ignore_errors=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)
    return dst


def round_log_path(workdir: Path, round_num: int, *, instance_id: str | None = None) -> Path:
    """Return the path to a round's training log.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number.
    instance_id : str | None
        The eval task's ``task_id``, if running an eval task (see
        ``round_dir``). Omit for training-only mode.

    Returns
    -------
    Path
        This round's ``round.log``.
    """
    return round_dir(workdir, round_num, instance_id=instance_id) / ROUND_LOG_FILENAME


def eval_dir(
    workdir: Path, round_num: int, instance_id: str, *, create: bool = False
) -> Path:
    """Return (and optionally create) an eval task instance's eval directory.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number this eval instance belongs to.
    instance_id : str
        The eval task instance identifier.
    create : bool
        If True, create the directory (and parents) if missing.

    Returns
    -------
    Path
        ``workdir/rounds_{instance_id}/round_{round_num}/eval``.
    """
    path = round_dir(workdir, round_num, instance_id=instance_id) / EVAL_DIRNAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def eval_result_path(workdir: Path, round_num: int, instance_id: str) -> Path:
    """Return the path to an eval instance's result file.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number this eval instance belongs to.
    instance_id : str
        The eval task instance identifier.

    Returns
    -------
    Path
        This eval instance's ``result.json``.
    """
    return eval_dir(workdir, round_num, instance_id) / RESULT_FILENAME


def eval_log_path(workdir: Path, round_num: int, instance_id: str) -> Path:
    """Return the path to an eval instance's log file.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number this eval instance belongs to.
    instance_id : str
        The eval task instance identifier.

    Returns
    -------
    Path
        This eval instance's ``eval.log``.
    """
    return eval_dir(workdir, round_num, instance_id) / EVAL_LOG_FILENAME


def eval_patch_path(workdir: Path, round_num: int, instance_id: str) -> Path:
    """Return the path to an eval instance's captured repo diff.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    round_num : int
        1-based round number this eval instance belongs to.
    instance_id : str
        The eval task instance identifier.

    Returns
    -------
    Path
        This eval instance's ``repo.patch``.
    """
    return eval_dir(workdir, round_num, instance_id) / ROUND_PATCH_FILENAME