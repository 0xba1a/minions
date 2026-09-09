"""Unit tests for microbots.auto_memory.cli."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from microbots.auto_memory.cli import main, parse_args

MODULE = "microbots.auto_memory.cli"


@pytest.mark.unit
def test_parse_args_defaults():
    args = parse_args(["--model", "azure-openai/gpt-4o", "--task", "swebenchverified"])

    assert args.model == "azure-openai/gpt-4o"
    assert args.task == "swebenchverified"
    assert args.max_rounds == 5
    assert args.workdir is None


@pytest.mark.unit
def test_main_builds_the_task_from_the_workdir_config_and_runs_the_loop(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "task_config.yaml").write_text("repo: django/django")

    task_cls = MagicMock()

    with patch.dict(f"{MODULE}.TASK_REGISTRY", {"swebenchverified": task_cls}, clear=True), \
         patch(f"{MODULE}.run") as mock_run:
        main([
            "--model", "azure-openai/gpt-4o",
            "--task", "swebenchverified",
            "--workdir", str(workdir),
            "--max-rounds", "2",
        ])

    task_cls.assert_called_once_with(config_file=workdir / "task_config.yaml")
    assert mock_run.call_args.kwargs["max_rounds"] == 2
    assert mock_run.call_args.kwargs["task"] is task_cls.return_value


@pytest.mark.unit
def test_main_prefers_an_explicit_config_file(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    config_file = tmp_path / "elsewhere.yaml"
    config_file.write_text("repo: django/django")

    task_cls = MagicMock()

    with patch.dict(f"{MODULE}.TASK_REGISTRY", {"swebenchverified": task_cls}, clear=True), \
         patch(f"{MODULE}.run"):
        main([
            "--model", "azure-openai/gpt-4o",
            "--task", "swebenchverified",
            "--workdir", str(workdir),
            "--config-file", str(config_file),
        ])

    task_cls.assert_called_once_with(config_file=config_file)


@pytest.mark.unit
def test_main_fails_fast_when_no_config_file_exists(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    with patch.dict(f"{MODULE}.TASK_REGISTRY", {"swebenchverified": MagicMock()}, clear=True), \
         patch(f"{MODULE}.run"):
        with pytest.raises(FileNotFoundError):
            main([
                "--model", "azure-openai/gpt-4o",
                "--task", "swebenchverified",
                "--workdir", str(workdir),
            ])
