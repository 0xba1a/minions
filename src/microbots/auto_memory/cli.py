"""Command-line entry point for the auto-memory train/eval loop.

Two modes, selected by ``--task``:

- ``--task <name>`` given: run the full train <-> eval loop for that
  task.

Both modes are dispatched via ``orchestrator.run``.
"""

import argparse
import logging
from pathlib import Path

from microbots.auto_memory.orchestrator import run
from microbots.auto_memory.task_registry import TASK_REGISTRY, discover_tasks
from microbots.auto_memory.workdir import require_workdir, resolve_workdir

logger = logging.getLogger(__name__)

# Import every task module so their @register_task decorators fire.
discover_tasks()

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI's top-level args.

    Task-specific values (e.g. an eval task's instance ID) are not
    parsed here; they come from the workdir's config file instead.

    Parameters
    ----------
    argv : list[str] | None
        Args to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    argparse.Namespace
        The parsed args.
    """
    parser = argparse.ArgumentParser(description="Run the auto-memory train/eval loop.")
    parser.add_argument("--model", required=True, help='Model, e.g. "azure-openai/gpt-5.5".')
    parser.add_argument(
        "--workdir",
        help="Directory holding this run's files (repo clone, logs, memory, "
        "config). Defaults to './workdir' relative to the current directory.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_REGISTRY),
        help="Eval task to run.",
    )
    parser.add_argument(
        "--config-file", type=Path, help="Path to the task configuration file.",
    )
    parser.add_argument("--max-rounds", type=int, default=5)

    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the full train/eval loop.

    Parameters
    ----------
    argv : list[str] | None
        Args to parse. Defaults to ``sys.argv[1:]`` when ``None``.
    """
    args = parse_args(argv)

    # The user can pass either an existing workdir containing a
    # task_config.yml file or a task_config.yml file using --config
    # option. In the later case, the workdir will be created in
    # the default location.
    # If both workdir and config options are provided and there exists
    # a task_yaml.yml inside the workdir, that file will be ignored and
    # the provided --config-file will take precedence.
    workdir = Path(args.workdir) if args.workdir else resolve_workdir()
    require_workdir(workdir)

    if not args.config_file:
        config_file = workdir / "task_config.yaml"
    else:
        config_file = args.config_file
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    result = run(
        workdir=workdir,
        model=args.model,
        task=TASK_REGISTRY[args.task](config_file=config_file),
        max_rounds=args.max_rounds,
    )
    if result is not None:
        logger.info(
            "task=%s passed=%s rounds_run=%d", args.task, result.passed, result.rounds_run
        )

if __name__ == "__main__":
    main()
