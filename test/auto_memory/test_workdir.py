"""Unit tests for microbots.auto_memory.workdir.

These are pure filesystem-layout tests; nothing here touches git, an
LLM, or Docker.
"""

import pytest

from microbots.auto_memory.workdir import (
    config_path,
    get_eval_dir,
    memory_dir,
    repo_dir,
    require_workdir,
    resolve_workdir,
    round_dir,
    take_memory_snapshot,
)


@pytest.mark.unit
def test_layout_paths_hang_off_the_workdir(tmp_path):
    assert resolve_workdir(tmp_path) == tmp_path / "workdir"
    assert config_path(tmp_path) == tmp_path / "task_config.yaml"
    assert repo_dir(tmp_path) == tmp_path / "repo"
    assert memory_dir(tmp_path) == tmp_path / "memory"
    assert round_dir(tmp_path, 2) == tmp_path / "rounds" / "round_2"
    assert get_eval_dir(tmp_path, 2) == tmp_path / "rounds" / "round_2" / "eval"


@pytest.mark.unit
def test_require_workdir_creates_memory_dir_for_a_new_workdir(tmp_path):
    workdir = tmp_path / "workdir"

    require_workdir(workdir)

    assert memory_dir(workdir).is_dir()


@pytest.mark.unit
def test_require_workdir_archives_previous_run_and_carries_state_over(tmp_path):
    workdir = tmp_path / "workdir"
    (workdir / "memory").mkdir(parents=True)
    (workdir / "memory" / "notes.md").write_text("learned")
    (workdir / "repo").mkdir()
    (workdir / "repo" / "code.py").write_text("x = 1")
    (workdir / "task_config.yaml").write_text("repo: acme/widget")
    (workdir / "rounds" / "round_1").mkdir(parents=True)

    require_workdir(workdir)

    # Config, memory and the training checkout survive.
    assert (workdir / "task_config.yaml").read_text() == "repo: acme/widget"
    assert (workdir / "memory" / "notes.md").read_text() == "learned"
    assert (workdir / "repo" / "code.py").read_text() == "x = 1"
    # Previous round output does not.
    assert not (workdir / "rounds").exists()
    assert any(p.name.startswith("workdir_backup_") for p in tmp_path.iterdir())


@pytest.mark.unit
def test_require_workdir_provides_memory_dir_even_when_previous_run_had_none(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "task_config.yaml").write_text("repo: acme/widget")

    require_workdir(workdir)

    assert memory_dir(workdir).is_dir()


@pytest.mark.unit
def test_round_dir_is_idempotent(tmp_path):
    first = round_dir(tmp_path, 1)
    (first / "eval").mkdir()

    second = round_dir(tmp_path, 1)

    assert second == first
    assert (second / "eval").is_dir(), "re-resolving a round must not wipe it"


@pytest.mark.unit
def test_take_memory_snapshot_copies_memory_into_the_round(tmp_path):
    mem_dir = memory_dir(tmp_path)
    mem_dir.mkdir(parents=True)
    (mem_dir / "notes.md").write_text("round one")

    take_memory_snapshot(mem_dir, 1)
    (mem_dir / "notes.md").write_text("round two")

    snapshot = round_dir(tmp_path, 1) / "starting_memory_snapshot" / "notes.md"
    assert snapshot.read_text() == "round one"


@pytest.mark.unit
def test_take_memory_snapshot_requires_the_memory_dir_to_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        take_memory_snapshot(memory_dir(tmp_path), 1)
