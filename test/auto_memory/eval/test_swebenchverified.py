"""Unit tests for microbots.auto_memory.eval.swebenchverified.

The SWE-bench dataset, git, the evaluation harness and every bot are
mocked, so these run without network access, Docker or an LLM.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from microbots.auto_memory.eval.swebenchverified import (
    SweBenchInstance,
    SweBenchVerified,
    SweBenchVerifiedTask_one,
)
from microbots.MicroBot import BotRunResult

MODULE = "microbots.auto_memory.eval.swebenchverified"

INSTANCE = SweBenchInstance(
    instance_id="django__django-11099",
    repo="django/django",
    base_commit="abc123",
    problem_statement="UsernameValidator allows trailing newline",
)


def _config(tmp_path: Path, **body) -> Path:
    path = tmp_path / "task_config.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def _task(tmp_path: Path, **body) -> SweBenchVerified:
    with patch(f"{MODULE}.load_instance_using_id", return_value=INSTANCE):
        return SweBenchVerified(_config(tmp_path, **body))


# ---------------------------------------------------------------------------
# parse_config / repo_url
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_parse_config_loads_the_listed_instances(tmp_path):
    task = _task(tmp_path, instance_id_list=["django__django-11099"])

    assert [i.instance_id for i in task.dataset] == ["django__django-11099"]
    assert task.repo_url() == "https://github.com/django/django.git"


@pytest.mark.unit
def test_parse_config_loads_every_instance_of_a_repo(tmp_path):
    with patch(f"{MODULE}.load_instances_of_repo", return_value=[INSTANCE, INSTANCE]) as loader:
        task = SweBenchVerified(_config(tmp_path, repo="django/django"))

    loader.assert_called_once_with(repo="django/django")
    assert len(task.dataset) == 2


@pytest.mark.unit
def test_parse_config_rejects_instances_from_different_repos(tmp_path):
    other = SweBenchInstance("flask__flask-1", "pallets/flask", "def456", "boom")

    with patch(f"{MODULE}.load_instance_using_id", side_effect=[INSTANCE, other]):
        with pytest.raises(ValueError, match="Conflicting repos"):
            SweBenchVerified(_config(tmp_path, instance_id_list=["a", "b"]))


@pytest.mark.unit
def test_parse_config_rejects_an_empty_selection(tmp_path):
    with pytest.raises(ValueError, match="No instances loaded"):
        SweBenchVerified(_config(tmp_path, note="nothing selected"))


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def _fake_harness(report: dict, test_output: str):
    """Stand in for the SWE-bench harness, writing its usual artifacts."""

    def run(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "diff"] or cmd[:2] == ["git", "add"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="diff --git a b\n", stderr="")

        report_dir = Path(kwargs["cwd"])
        run_id = cmd[cmd.index("--run_id") + 1]
        log_dir = (
            report_dir / "logs" / "run_evaluation" / run_id
            / "microbots-eval-agent" / INSTANCE.instance_id
        )
        log_dir.mkdir(parents=True)
        (log_dir / "report.json").write_text(json.dumps(report))
        (log_dir / "test_output.txt").write_text(test_output)
        return subprocess.CompletedProcess(cmd, 0, stdout="harness done\n", stderr="")

    return run


@pytest.mark.unit
def test_check_passes_when_the_harness_resolves_the_instance(tmp_path):
    log_path = tmp_path / "instance.log"
    log_path.write_text("")
    harness = _fake_harness({INSTANCE.instance_id: {"resolved": True}}, "OK")

    with patch(f"{MODULE}.subprocess.run", side_effect=harness):
        result = SweBenchVerifiedTask_one(INSTANCE).check("/repo", "", str(log_path))

    assert result.status
    assert result.result == "resolved"
    assert result.error is None


@pytest.mark.unit
def test_check_reports_the_test_output_when_the_instance_is_unresolved(tmp_path):
    log_path = tmp_path / "instance.log"
    log_path.write_text("")
    harness = _fake_harness(
        {INSTANCE.instance_id: {"resolved": False}},
        "FAILED tests/test_validators.py::test_trailing_newline",
    )

    with patch(f"{MODULE}.subprocess.run", side_effect=harness):
        result = SweBenchVerifiedTask_one(INSTANCE).check("/repo", "", str(log_path))

    assert not result.status
    assert "test_trailing_newline" in result.error
    assert "test_trailing_newline" in log_path.read_text()


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_eval_scores_the_fraction_of_resolved_instances(tmp_path):
    task = _task(tmp_path, instance_id_list=["django__django-11099"])
    task.dataset = [INSTANCE, INSTANCE]

    agent_ran = BotRunResult(status=True, result="patched", error=None)
    verdicts = [
        BotRunResult(status=True, result="resolved", error=None),
        BotRunResult(status=False, result="not resolved", error="tests failed"),
    ]

    with patch.object(SweBenchVerifiedTask_one, "setup"), \
         patch.object(SweBenchVerifiedTask_one, "check", side_effect=verdicts), \
         patch(f"{MODULE}.WritingBot") as writing_bot, \
         patch(f"{MODULE}.MemoryTool"), \
         patch(f"{MODULE}.ReadingBot") as reading_bot:
        writing_bot.return_value.run.return_value = agent_ran
        reading_bot.return_value.run.return_value = BotRunResult(
            status=True, result="one instance still fails", error=None
        )
        outcome = task.eval(str(tmp_path / "memory"), "azure-openai/gpt-4o", str(tmp_path / "eval"))

    assert outcome.score == 0.5
    assert not outcome.passed
    assert outcome.feedback == "one instance still fails"


@pytest.mark.unit
def test_eval_passes_only_when_every_instance_resolves(tmp_path):
    task = _task(tmp_path, instance_id_list=["django__django-11099"])
    resolved = BotRunResult(status=True, result="resolved", error=None)

    with patch.object(SweBenchVerifiedTask_one, "setup"), \
         patch.object(SweBenchVerifiedTask_one, "check", return_value=resolved), \
         patch(f"{MODULE}.WritingBot") as writing_bot, \
         patch(f"{MODULE}.MemoryTool"):
        writing_bot.return_value.run.return_value = BotRunResult(
            status=True, result="patched", error=None
        )
        outcome = task.eval(str(tmp_path / "memory"), "azure-openai/gpt-4o", str(tmp_path / "eval"))

    assert outcome.passed
    assert outcome.score == 1
    assert outcome.feedback == "All evaluations passed."


@pytest.mark.unit
def test_eval_skips_the_harness_when_the_agent_itself_failed(tmp_path):
    task = _task(tmp_path, instance_id_list=["django__django-11099"])

    with patch.object(SweBenchVerifiedTask_one, "setup"), \
         patch.object(SweBenchVerifiedTask_one, "check") as check, \
         patch(f"{MODULE}.WritingBot") as writing_bot, \
         patch(f"{MODULE}.MemoryTool"), \
         patch(f"{MODULE}.ReadingBot") as reading_bot:
        writing_bot.return_value.run.return_value = BotRunResult(
            status=False, result=None, error="agent timed out"
        )
        reading_bot.return_value.run.return_value = BotRunResult(
            status=True, result="the agent never produced a patch", error=None
        )
        outcome = task.eval(str(tmp_path / "memory"), "azure-openai/gpt-4o", str(tmp_path / "eval"))

    check.assert_not_called()
    assert outcome.score == 0


@pytest.mark.unit
def test_combined_feedback_falls_back_to_raw_results_when_the_bot_fails(tmp_path):
    task = _task(tmp_path, instance_id_list=["django__django-11099"])
    results = [BotRunResult(status=False, result="not resolved", error="assertion failed")]

    with patch(f"{MODULE}.ReadingBot", side_effect=RuntimeError("no model configured")):
        feedback = task._combine_result_feedback(results, "azure-openai/gpt-4o", "/repo")

    assert "assertion failed" in feedback
