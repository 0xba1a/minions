"""SWE-bench-verified eval task.

Loads instances from the SWE-bench-verified dataset, checks out each
instance's repo at its base commit, has the agent attempt a fix, and
verifies the result via ``swebench.harness.run_evaluation``.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from functools import cache
from logging import getLogger
from pathlib import Path

import yaml

from microbots.auto_memory.evalTask import EvalOutcome, EvalTask
from microbots.auto_memory.run_logging import log_to_file
from microbots.auto_memory.task_registry import register_task
from microbots.auto_memory.workdir import eval_log_dir
from microbots.bot.ReadingBot import ReadingBot
from microbots.bot.WritingBot import WritingBot
from microbots.MicroBot import BotRunResult
from microbots.tools.tool_definitions.memory_tool import MemoryTool

logger = getLogger(__name__)

SWE_BENCH_VERIFIED = "SWE-bench/SWE-bench_Verified"
EVAL_AGENT_MODEL_NAME = "microbots-eval-agent"


@dataclass
class SweBenchInstance:
    """A single SWE-bench-verified dataset row.

    Attributes
    ----------
    instance_id : str
        Unique identifier for the instance, e.g. ``"django__django-11099"``.
    repo : str
        The GitHub repo this instance belongs to, e.g. ``"django/django"``.
    base_commit : str
        Commit hash representing the repo state before the issue's fix.
    problem_statement : str
        The GitHub issue title and body describing the bug to fix.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str


@cache
def _load_dataset_rows(dataset_name: str):
    """Load and cache ``dataset_name``'s ``test`` split for the process's lifetime.

    ``load_dataset`` caches the downloaded files on disk, but still
    re-reads and rebuilds the in-memory ``Dataset`` object on every
    call. Since ``load_instances_of_repo``/``load_instance_using_id``
    may each be called many times,
    this wraps ``load_dataset`` with an in-memory cache keyed by
    ``dataset_name``, so the dataset is only loaded once per process.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset name to load.

    Returns
    -------
    datasets.Dataset
        The loaded ``test`` split.

    Raises
    ------
    ImportError
        If the optional ``datasets`` package (the ``training`` extra)
        isn't installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "SWE-bench-verified evaluation requires the 'training' extra: "
            "pip install 'microbots[training]'"
        ) from exc
    return load_dataset(dataset_name, split="test")


def load_instances_of_repo(
    dataset_name: str = SWE_BENCH_VERIFIED,
    repo: str | None = None,
) -> list[SweBenchInstance]:
    """Load all dataset instances, optionally filtered to a single repo.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset name to load. Defaults to
        ``SWE_BENCH_VERIFIED``.
    repo : str | None
        If given, only instances whose ``repo`` matches this value are
        returned, e.g. ``"django/django"``. If ``None``, all instances
        are returned.

    Returns
    -------
    list[SweBenchInstance]
        The matching instances.
    """
    rows = _load_dataset_rows(dataset_name)
    instances = [
        SweBenchInstance(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
        )
        for row in rows
        if repo is None or row["repo"] == repo
    ]
    return instances

def load_instance_using_id(instance_id: str, dataset_name: str = SWE_BENCH_VERIFIED) -> SweBenchInstance:
    """Load a single dataset instance by its instance ID.

    Parameters
    ----------
    instance_id : str
        The instance ID to look up, e.g. ``"django__django-11099"``.
    dataset_name : str
        Hugging Face dataset name to load. Defaults to
        ``SWE_BENCH_VERIFIED``.

    Returns
    -------
    SweBenchInstance
        The matching instance.

    Raises
    ------
    ValueError
        If no instance with the given ``instance_id`` exists in the
        dataset.
    """
    rows = _load_dataset_rows(dataset_name)
    for row in rows:
        if row["instance_id"] == instance_id:
            return SweBenchInstance(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row["problem_statement"],
            )
    raise ValueError(f"instance_id not found: {instance_id}")

class SweBenchVerifiedTask_one:
    """Runs and grades a single SWE-bench-verified instance.

    Checks out the instance's repo at its base commit, gives the agent
    the issue's problem statement, and verifies the resulting patch
    with the official SWE-bench evaluation harness.

    Parameters
    ----------
    instance : SweBenchInstance
        The dataset instance this task evaluates against.
    """

    def __init__(self, instance: SweBenchInstance):
        self.instance = instance

    def setup(self, repo_path: str) -> None:
        """Clone the instance's repo, or reset it, to its base commit.

        Parameters
        ----------
        repo_path : str
            Absolute path to clone (or reset) the repo into.
        """
        expected_url = f"https://github.com/{self.instance.repo}.git"

        if Path(repo_path).exists():
            origin = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=repo_path, capture_output=True, text=True,
            )
            if origin.returncode == 0 and origin.stdout.strip() == expected_url:
                subprocess.run(
                    ["git", "reset", "--hard", self.instance.base_commit],
                    cwd=repo_path, check=True,
                )
                subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True)
                return

            logger.warning(
                "SweBenchVerifiedTask.setup: %s exists but isn't a checkout of %s "
                "(origin=%r); removing and re-cloning",
                repo_path, expected_url, origin.stdout.strip(),
            )
            shutil.rmtree(repo_path)

        subprocess.run(["git", "clone", expected_url, repo_path], check=True)
        subprocess.run(
            ["git", "checkout", self.instance.base_commit], cwd=repo_path, check=True
        )

    def build_prompt(self) -> str:
        """Return the instance's issue text as the agent's prompt.

        Returns
        -------
        str
            The instance's ``problem_statement``.
        """
        return self.instance.problem_statement

    def check(self, repo_path: str, agent_output: str, log_path: str) -> BotRunResult:
        """Verify the agent's patch using the SWE-bench evaluation harness.

        Captures the agent's changes as a git diff (after marking any
        untracked new files intent-to-add, so files the agent newly
        created are included in the diff rather than silently
        dropped), submits it as a prediction to
        ``swebench.harness.run_evaluation``, and checks whether the
        harness marked this instance as resolved.

        Parameters
        ----------
        repo_path : str
            Absolute path to the repo the agent operated on.
        agent_output : str
            The agent's raw output/result text. Unused here, since
            verification is based on the repo's git diff, not the
            agent's textual output.
        log_path : str
            Path to a log file to append the harness's output to,
            including the per-instance ``run_instance.log`` and
            ``test_output.txt`` artifacts if the harness produced them
            (read before the harness's ``report_dir`` is cleaned up).

        Returns
        -------
        BotRunResult
            ``status`` is whether the harness marked this instance as
            resolved. On failure, ``error`` carries the harness's
            ``test_output.txt`` (or its console output, if the harness
            died before producing one) so the feedback bot can see why
            the tests failed.
        """
        subprocess.run(
            ["git", "add", "--intent-to-add", "."], cwd=repo_path, check=True
        )
        # Diffed against the base commit, so the patch is the same whether or
        # not the agent committed its work.
        diff = subprocess.run(
            ["git", "diff", "--binary", self.instance.base_commit],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

        run_id = f"microbots-{uuid.uuid4().hex[:8]}"
        model_name_or_path = EVAL_AGENT_MODEL_NAME
        #will need to update when upgraded to ~5.0.2 , removed this flag in the new version
        #https://github.com/SWE-bench/SWE-bench/commit/e2c13307b6cf7764a50958b9c8bfbfb3f72cb70a
        report_dir = Path(tempfile.mkdtemp())
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as pred_file:
            pred_path = Path(pred_file.name)
            pred_file.write(json.dumps([{
                "instance_id": self.instance.instance_id,
                "model_patch": diff,
                "model_name_or_path": model_name_or_path,
            }]))

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "swebench.harness.run_evaluation",
                 "--dataset_name", SWE_BENCH_VERIFIED,
                 "--max_workers", "1",
                 "--predictions_path", str(pred_path),
                 "--run_id", run_id,
                 "--report_dir", str(report_dir),
                 "--instance_ids", self.instance.instance_id],
                 #can add timeout if needed
                capture_output=True, text=True,
                cwd=report_dir,
            )
            #need to update this instance_log_dir path when swebench is upgraded
            instance_log_dir = (
                report_dir / "logs" / "run_evaluation" / run_id
                / model_name_or_path / self.instance.instance_id
            )
            # Read while report_dir still exists; the finally block deletes it.
            test_output = ""
            with open(log_path, "a") as f:
                f.write(proc.stdout + proc.stderr)
                for log_filename in ("run_instance.log", "test_output.txt"):
                    log_file = instance_log_dir / log_filename
                    if log_file.exists():
                        content = log_file.read_text()
                        if log_filename == "test_output.txt":
                            test_output = content
                        f.write(f"\n--- {log_filename} ---\n{content}\n")

            report_file = instance_log_dir / "report.json"
            passed = False
            if report_file.exists():
                report = json.loads(report_file.read_text())
                passed = report.get(self.instance.instance_id, {}).get("resolved", False)
        finally:
            pred_path.unlink(missing_ok=True)
            shutil.rmtree(report_dir, ignore_errors=True)

        return BotRunResult(
            status = passed,
            result = "resolved" if passed else "not resolved",
            # Harness can fail before producing test_output.txt; fall back to its console output.
            error = None if passed else (test_output or proc.stdout + proc.stderr)
        )

    def eval(self, repo_path: str, memory_dir: str, model: str, log_path: str) -> BotRunResult:
        """Check out the repo and let the agent attempt the issue.

        Grading is deliberately left to ``check``, which the caller runs
        afterwards against the same checkout.

        Parameters
        ----------
        repo_path : str
            Absolute path to check the instance's repo out into.
        memory_dir : str
            Directory containing memory files to give the agent via
            ``MemoryTool``.
        model : str
            The model to use, in the format ``<provider>/<model_name>``.
        log_path : str
            Path to write this instance's log to. Truncated on entry.

        Returns
        -------
        BotRunResult
            The agent's run result, or a failed result carrying the
            exception if the attempt raised.
        """
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("")

        try:
            self.setup(repo_path)
            prompt = self.build_prompt()
            bot = WritingBot(
                model=model,
                folder_to_mount=repo_path,
                additional_tools=[MemoryTool(memory_dir=memory_dir)],
            )
            bot_result = bot.run(
                prompt,
                max_iterations=40,
                timeout_in_seconds=1800
                )

            with open(log_path, "a") as f:
                f.write(f"Agent output:\n{bot_result.result}\n")

            return bot_result

        except Exception as exc:
            logger.exception(
                "SweBenchVerifiedTask.eval: iteration raised %s", type(exc).__name__
            )
            with open(log_path, "a") as f:
                f.write(f"\nException during eval iteration: {type(exc).__name__}: {exc}\n")
            return BotRunResult(
                status=False,
                result=None,
                error=f"{type(exc).__name__}: {exc}"
            )

@register_task("swebenchverified")
class SweBenchVerified(EvalTask):
    """Evaluates memory against a set of SWE-bench-verified instances.

    Every instance in the configured set is attempted with the same
    memory, and the round's score is the fraction that the harness
    marks resolved.
    """

    def __init__(self, config_file: Path) -> None:
        # dataset must exist before parse_config populates it.
        self.dataset: list[SweBenchInstance] = []
        self.parse_config(config_file=config_file)

    def repo_url(self) -> str:
        """Return the clone URL of the repo the instances belong to.

        Returns
        -------
        str
            The training repo's clone URL. ``parse_config`` guarantees
            every instance shares one repo.
        """
        return f"https://github.com/{self.dataset[0].repo}.git"

    def parse_config(self, config_file: Path) -> None:
        """Load the instances this task evaluates from ``config_file``.

        The YAML file selects instances either by an
        ``instance_id_list`` of dataset IDs, or by a ``repo`` naming a
        SWE-bench repo such as ``django/django``.

        Parameters
        ----------
        config_file : Path
            Path to the task's YAML config file.

        Raises
        ------
        ValueError
            If the selected instances span more than one repo, or if
            the config selects no instances at all.
        """

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        instance_ids = config.get("instance_id_list", [])
        repo = config.get("repo", None)

        if instance_ids:
            repo = None
            for instance_id in instance_ids:
                dataset = load_instance_using_id(instance_id)
                if not repo:
                    repo = dataset.repo
                elif repo != dataset.repo:
                    raise ValueError(
                        f"Conflicting repos for instance_id {instance_id}: {repo} vs {dataset.repo}"
                    )

                self.dataset.append(dataset)

        elif repo:
            self.dataset = load_instances_of_repo(repo=repo)

        if len(self.dataset) == 0:
            raise ValueError("No instances loaded for evaluation.")

    def eval(self, memory_dir: str, model: str, eval_dir: str) -> EvalOutcome:
        """Attempt every configured instance and combine the results.

        Parameters
        ----------
        memory_dir : str
            Directory containing the memory notes to evaluate.
        model : str
            The model to use, in the format ``<provider>/<model_name>``.
        eval_dir : str
            Directory this round's eval owns; holds one checkout and one
            log file per instance.

        Returns
        -------
        EvalOutcome
            ``score`` is the fraction of instances resolved, and
            ``passed`` is true only when every one of them was.
        """
        eval_path = Path(eval_dir)
        # One checkout per instance: the agent's container writes to it as
        # root, leaving it unresettable for any instance that came after.
        eval_repos_path = eval_path / "eval_repo"
        log_dir = eval_log_dir(eval_path)
        results = []

        for instance in self.dataset:
            inst_log_path = log_dir / f"{instance.instance_id}_log.txt"
            inst_repo_path = eval_repos_path / instance.instance_id
            task = SweBenchVerifiedTask_one(instance)

            with log_to_file(inst_log_path):
                res = task.eval(str(inst_repo_path), memory_dir, model, str(inst_log_path))

                if not res.status:
                    logger.info(f"Evaluation failed for instance {instance.instance_id}: {res.error if res.error else 'Unknown error'}")
                    results.append(res)
                else:
                    res = task.check(str(inst_repo_path), "", str(inst_log_path))
                    results.append(res)

        score = 0
        for result in results:
            if result.status:
                score += 1

        score = score / len(self.dataset)

        if score == 1:
            feedback = "All evaluations passed."
        else:
            feedback = self._combine_result_feedback(results, model, str(eval_repos_path))

        # NOTE: Let's not teardown the repository as it will be useful for debugging

        return EvalOutcome(
            passed = score == 1,
            score = score,
            feedback = feedback
        )

    def _combine_result_feedback(self, results: list[BotRunResult], model: str, eval_repo: str) -> str:
        """Summarize every instance's result into one feedback string.

        Parameters
        ----------
        results : list[BotRunResult]
            One result per attempted instance.
        model : str
            The model to use, in the format ``<provider>/<model_name>``.
        eval_repo : str
            Path to the evaluation checkout, mounted for the bot.

        Returns
        -------
        str
            The bot's summary, falling back to the raw concatenated
            results if the bot is unavailable or fails.
        """

        serialized_str = f"Total {len(results)} tests ran and their result and feedback:\n"

        for res in results:
            serialized_str += f"\nResult: {'Passed' if res.status else 'Failed'}\n"
            serialized_str += f"Optional Feedback: {res.result if res.result else 'None'}\n"
            serialized_str += f"Error if there are any: {res.error if res.error else 'None'}\n"

        try:
            bot = ReadingBot(
                model = model,
                folder_to_mount=eval_repo
            )
            task = f"""
            Combine the results of the eval runs into single feedback.
            This feedback will be given to the next iteration.
            You just combine the results with minimal efforts.
            Avoid referring to code whenever possible.

            {serialized_str}
            """
            bot_result = bot.run(task=task)
        except Exception as e:
            logger.warning(f"Combining results failed with exception: {e}")
            return f"Combining results failed. raw combined output:\n\n{serialized_str}"

        if bot_result.status:
            return bot_result.result if bot_result.result else 'No feedback provided'
        else:
            return f"Combining results failed. raw combined output:\n\n{serialized_str}"
