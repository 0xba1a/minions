"""Run repository training with a reading bot and persistent memory tool."""

from pathlib import Path

from microbots.bot.ReadingBot import ReadingBot
from microbots.MicroBot import BotRunResult
from microbots.tools.tool_definitions.memory_tool import MemoryTool

_INSTRUCTIONS_PATH = Path(__file__).parent / "training_instructions.md"


def run_training(
    repo_path: str,
    feedback: str,
    memory_dir: str,
    model: str,
    max_iterations: int = 200,
    timeout_in_seconds: int = 3600,
) -> BotRunResult:
    """Run one training pass over a repository and update its memory.

    Parameters
    ----------
    repo_path : str
        Absolute path to a local repository to learn from. The caller
        (e.g. the orchestrator or an ``EvalTask``'s ``setup``) is
        responsible for ensuring this is a ready, checked-out local
        directory; this function does not clone or otherwise manage
        the repo.
    feedback : str
        Optional feedback to include in the training prompt.
    memory_dir : str
        Directory in which the memory tool stores its memory.
    model : str
        Model identifier used by the reading bot.
    max_iterations : int, default=200
        Maximum number of bot iterations.
    timeout_in_seconds : int, default=3600
        Maximum duration of the bot run in seconds.

    Returns
    -------
    microbots.MicroBot.BotRunResult
        Result of the training bot run.
    """
    instructions = _INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    feedback_section = feedback.strip() or "No feedback provided for this run."
    prompt = f"{instructions}\n\n## Feedback\n{feedback_section}\n"

    bot = ReadingBot(
        model=model,
        folder_to_mount=repo_path,
        additional_tools=[MemoryTool(memory_dir=memory_dir)],
    )

    return bot.run(
        prompt,
        max_iterations=max_iterations,
        timeout_in_seconds=timeout_in_seconds,
    )