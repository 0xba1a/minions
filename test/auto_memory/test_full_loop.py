"""End-to-end test of the train <-> eval loop on one SWE-bench instance.

Only the outside world is mocked: the SWE-bench dataset, git, the
evaluation harness, and the three bots. Everything inside
``microbots.auto_memory`` runs for real, so this exercises the whole
path a live run takes:

    cli.main
      -> require_workdir            (workdir laid out)
      -> SweBenchVerified           (config parsed, instance loaded)
      -> orchestrator.run           (training repo cloned)
        -> round 1: eval -> harness says unresolved -> feedback -> training
        -> round 2: eval -> harness says resolved   -> loop returns

The instance is unresolved on the first round and resolved on the
second, so both the failure/retrain branch and the success branch are
covered in a single run.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from microbots.auto_memory.cli import main
from microbots.auto_memory.eval.swebenchverified import (
    SweBenchInstance,
    SweBenchVerifiedTask_one,
)
from microbots.MicroBot import BotRunResult

SWE_MODULE = "microbots.auto_memory.eval.swebenchverified"

INSTANCE = SweBenchInstance(
    instance_id="django__django-11099",
    repo="django/django",
    base_commit="abc123",
    problem_statement="UsernameValidator allows trailing newline in usernames",
)


@pytest.fixture
def workdir(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "task_config.yaml").write_text(
        yaml.safe_dump({"instance_id_list": [INSTANCE.instance_id]})
    )
    return workdir


@pytest.mark.integration
def test_full_loop_retrains_after_a_failed_round_then_passes(workdir):
    training_bot = MagicMock()
    training_bot.run.return_value = BotRunResult(status=True, result="notes written", error=None)

    eval_bot = MagicMock()
    eval_bot.run.return_value = BotRunResult(status=True, result="patch applied", error=None)

    feedback_bot = MagicMock()
    feedback_bot.run.return_value = BotRunResult(
        status=True, result="Memory must cover the username validator regex.", error=None
    )

    # The agent gets it wrong the first round and right the second.
    harness_verdicts = [
        BotRunResult(status=False, result="not resolved", error="FAILED test_validators.py"),
        BotRunResult(status=True, result="resolved", error=None),
    ]

    def fake_clone(url, repo_path):
        Path(repo_path).mkdir(parents=True, exist_ok=True)

    def fake_setup(self, repo_path):
        Path(repo_path).mkdir(parents=True, exist_ok=True)

    with patch(f"{SWE_MODULE}.load_instance_using_id", return_value=INSTANCE), \
         patch("microbots.auto_memory.orchestrator.clone_repo", side_effect=fake_clone) as clone, \
         patch.object(SweBenchVerifiedTask_one, "setup", fake_setup), \
         patch.object(SweBenchVerifiedTask_one, "check", side_effect=harness_verdicts), \
         patch(f"{SWE_MODULE}.WritingBot", return_value=eval_bot), \
         patch(f"{SWE_MODULE}.ReadingBot", return_value=feedback_bot), \
         patch(f"{SWE_MODULE}.MemoryTool"), \
         patch("microbots.auto_memory.training.runner.ReadingBot", return_value=training_bot), \
         patch("microbots.auto_memory.training.runner.MemoryTool"):
        main([
            "--model", "azure-openai/gpt-4o",
            "--task", "swebenchverified",
            "--workdir", str(workdir),
            "--max-rounds", "3",
        ])

    # The training repo is derived from the instance, not from the config.
    assert clone.call_args.args[0] == "https://github.com/django/django.git"

    # Two rounds ran, and only the failing one triggered retraining.
    assert eval_bot.run.call_count == 2
    training_bot.run.assert_called_once()
    assert not (workdir / "rounds" / "round_3").exists()

    # The failed round's feedback is what the training agent was given.
    training_prompt = training_bot.run.call_args.args[0]
    assert "Memory must cover the username validator regex." in training_prompt

    # Round 1 recorded a failure, round 2 a pass.
    round_1 = _result(workdir, 1)
    assert round_1["passed"] is False
    assert round_1["score"] == 0

    round_2 = _result(workdir, 2)
    assert round_2["passed"] is True
    assert round_2["score"] == 1

    # Each round kept the memory it started from, and the per-instance log.
    for round_num in (1, 2):
        round_path = workdir / "rounds" / f"round_{round_num}"
        assert (round_path / "starting_memory_snapshot").is_dir()
        assert (
            round_path / "eval" / "logs" / f"{INSTANCE.instance_id}_log.txt"
        ).is_file()


def _result(workdir: Path, round_num: int) -> dict:
    import json

    return json.loads(
        (workdir / "rounds" / f"round_{round_num}" / "eval" / "result.json").read_text()
    )
