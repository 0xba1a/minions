"""Unit tests for microbots.auto_memory.task_registry."""

from pathlib import Path

import pytest

from microbots.auto_memory.evalTask import EvalOutcome, EvalTask
from microbots.auto_memory.task_registry import (
    TASK_REGISTRY,
    discover_tasks,
    register_task,
)


class _StubTask(EvalTask):
    """Minimal concrete task, enough to be registrable."""

    def parse_config(self, config_file: Path) -> None:
        self._repo_url = "https://github.com/acme/widget.git"

    def eval(self, memory_dir: str, model: str, eval_dir: str) -> EvalOutcome:
        return EvalOutcome(passed=True, score=1.0, feedback="ok")


@pytest.fixture(autouse=True)
def _restore_registry():
    original = dict(TASK_REGISTRY)
    yield
    TASK_REGISTRY.clear()
    TASK_REGISTRY.update(original)


@pytest.mark.unit
def test_register_task_stores_the_class_under_its_name():
    register_task("stub")(_StubTask)

    assert TASK_REGISTRY["stub"] is _StubTask


@pytest.mark.unit
def test_registering_the_same_class_twice_is_allowed():
    register_task("stub")(_StubTask)
    register_task("stub")(_StubTask)

    assert TASK_REGISTRY["stub"] is _StubTask


@pytest.mark.unit
def test_registering_a_second_class_under_one_name_raises():
    class _OtherTask(_StubTask):
        pass

    register_task("stub")(_StubTask)

    with pytest.raises(ValueError, match="already registered"):
        register_task("stub")(_OtherTask)


@pytest.mark.unit
def test_discover_tasks_registers_the_shipped_eval_tasks():
    discover_tasks()

    assert "swebenchverified" in TASK_REGISTRY
