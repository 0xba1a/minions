"""Routes this package's existing log records into a run's workdir files.

Three sinks, matching the workdir layout: the package's own records go
to ``workdir/log.txt``, while ``log_to_file`` temporarily captures the
agent/tool records emitted underneath a specific step (a training pass,
or one eval instance) into that step's own file.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from microbots.auto_memory.workdir import log_path

PACKAGE_LOGGER_NAME = "microbots.auto_memory"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _file_handler(path: Path) -> logging.FileHandler:
    """Build a formatted file handler appending to ``path``.

    Parameters
    ----------
    path : Path
        File to append records to. Parent directories are created.

    Returns
    -------
    logging.FileHandler
        The configured handler.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def configure_run_logging(workdir: Path) -> None:
    """Send this package's own records to ``workdir/log.txt``.

    Propagation is disabled so these records never also land in the
    per-step files that ``log_to_file`` attaches to the root logger.
    Call once per run, after ``require_workdir`` has prepared the
    workdir, otherwise the log file is archived out from under the
    open handle.

    Parameters
    ----------
    workdir : Path
        The run's workdir.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(_file_handler(log_path(workdir)))


@contextmanager
def log_to_file(path: Path) -> Iterator[None]:
    """Capture root-logger records into ``path`` for the duration of the block.

    Used to give a single training pass or eval instance its own file
    without changing what any of the underlying agents/tools log.

    Parameters
    ----------
    path : Path
        File to append records to for the duration of the block.
    """
    logger = logging.getLogger()
    handler = _file_handler(path)
    previous_level = logger.level
    if not previous_level or previous_level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()
