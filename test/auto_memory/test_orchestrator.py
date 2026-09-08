"""Unit tests for microbots.auto_memory.orchestrator.

The training bot is mocked out; these tests only cover how the loop
sequences rounds, reacts to outcomes, and records results.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from microbots.auto_memory.evalTask import EvalOutcome
from microbots.auto_memory.orchestrator import (
    run_train_eval_loop,
    write_eval_result,
)

MODULE = "microbots.auto_memory.orchestrator"


def _outcome(passed: bool, score: float = 0.0, feedback: str = "needs work") -> EvalOutcome:
    return EvalOutcome(passed=passed, score=score, feedback=feedback)


def _task(*outcomes: EvalOutcome) -> MagicMock:
    task = MagicMock()
    task.eval.side_effect = list(outcomes)
    return task


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "memory").mkdir()
    return tmp_path


@pytest.mark.unit
def test_write_eval_result_serializes_the_outcome(tmp_path):
    write_eval_result(tmp_path, _outcome(True, score=1.0, feedback="all good"))

    assert json.loads((tmp_path / "result.json").read_text()) == {
        "passed": True,
        "score": 1.0,
        "feedback": "all good",
    }


@pytest.mark.unit
def test_loop_stops_on_the_first_passing_round(workdir):
    task = _task(_outcome(True, score=1.0))

    with patch(f"{MODULE}.run_training") as mock_training:
        result = run_train_eval_loop("/repo", workdir, "azure-openai/gpt-4o", task, max_rounds=3)

    assert result.passed
    assert result.rounds_run == 1
    assert task.eval.call_count == 1
    mock_training.assert_not_called()


@pytest.mark.unit
def test_loop_retrains_with_the_round_feedback_then_passes(workdir):
    task = _task(_outcome(False, feedback="cover the settings module"), _outcome(True, score=1.0))

    with patch(f"{MODULE}.run_training") as mock_training:
        result = run_train_eval_loop("/repo", workdir, "azure-openai/gpt-4o", task, max_rounds=3)

    assert result.passed
    assert result.rounds_run == 2
    mock_training.assert_called_once()
    assert mock_training.call_args.kwargs["feedback"] == "cover the settings module"


@pytest.mark.unit
def test_loop_gives_up_after_max_rounds(workdir):
    task = _task(_outcome(False), _outcome(False))

    with patch(f"{MODULE}.run_training"):
        result = run_train_eval_loop("/repo", workdir, "azure-openai/gpt-4o", task, max_rounds=2)

    assert not result.passed
    assert result.rounds_run == 2
    assert len(result.outcomes) == 2


@pytest.mark.unit
def test_a_raising_eval_is_recorded_and_the_loop_continues(workdir):
    task = MagicMock()
    task.eval.side_effect = [RuntimeError("harness exploded"), _outcome(True, score=1.0)]

    with patch(f"{MODULE}.run_training"):
        result = run_train_eval_loop("/repo", workdir, "azure-openai/gpt-4o", task, max_rounds=2)

    assert result.passed
    assert result.outcomes[0].score == -1
    assert "harness exploded" in result.outcomes[0].feedback


@pytest.mark.unit
def test_each_round_records_its_result_and_memory_snapshot(workdir):
    (workdir / "memory" / "notes.md").write_text("what I know")
    task = _task(_outcome(False), _outcome(False))

    with patch(f"{MODULE}.run_training"):
        run_train_eval_loop("/repo", workdir, "azure-openai/gpt-4o", task, max_rounds=2)

    for round_num in (1, 2):
        round_path = workdir / "rounds" / f"round_{round_num}"
        assert (round_path / "eval" / "result.json").is_file()
        assert (round_path / "starting_memory_snapshot" / "notes.md").is_file()


@pytest.mark.unit
def test_max_rounds_must_be_at_least_one(workdir):
    with pytest.raises(ValueError, match="max_rounds"):
        run_train_eval_loop("/repo", workdir, "azure-openai/gpt-4o", MagicMock(), max_rounds=0)
