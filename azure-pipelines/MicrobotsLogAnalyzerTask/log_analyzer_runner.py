import argparse
import logging
import os
import sys
import textwrap

from azure.identity import AzureCliCredential, get_bearer_token_provider
from microbots import LogAnalysisBot


def is_docker_access_error(error):
    return "docker" in type(error).__module__.lower() or "docker" in str(error).lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Microbots LogAnalysisBot.")
    parser.add_argument("--codebase-path", required=True)
    parser.add_argument("--log-file-path", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--output-file")
    parser.add_argument("--debug-log-file")
    parser.add_argument("--user-prompt")
    parser.add_argument("--max-iterations", type=int)
    return parser.parse_args()


def log(message, *, file=sys.stdout):
    print(message, file=file, flush=True)


def safe_analysis_line(message):
    if message.startswith("##vso[") or message.startswith("##["):
        return " " + message
    return message


def write_text_file(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write(content)


def configure_debug_logging(debug_log_file):
    """Capture Microbots runtime logs (thoughts, tool calls, command output) to a file."""
    os.makedirs(os.path.dirname(debug_log_file), exist_ok=True)
    file_handler = logging.FileHandler(debug_log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)


def main():
    args = parse_args()
    codebase_path = os.path.abspath(args.codebase_path)
    log_file_path = args.log_file_path
    timeout_seconds = args.timeout_seconds
    max_iterations = args.max_iterations

    os.chdir(codebase_path)
    if args.debug_log_file:
        configure_debug_logging(args.debug_log_file)
    log(
        f"MicrobotsLogAnalyzer: analyzing {log_file_path} with deployment "
        f"{os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']}"
    )
    log(f"MicrobotsLogAnalyzer: timeout is {timeout_seconds} seconds")
    if max_iterations is not None:
        log(f"MicrobotsLogAnalyzer: max iterations is {max_iterations}")
    if args.output_file:
        log(f"MicrobotsLogAnalyzer: analysis output file is {args.output_file}")
    if args.debug_log_file:
        log(f"MicrobotsLogAnalyzer: debugging log file is {args.debug_log_file}")
    if args.user_prompt:
        log("MicrobotsLogAnalyzer: additional user context was provided")

    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    run_kwargs = {
        "file_name": log_file_path,
        "timeout_in_seconds": timeout_seconds,
    }
    if max_iterations is not None:
        run_kwargs["max_iterations"] = max_iterations
    if args.user_prompt:
        run_kwargs["user_prompt"] = args.user_prompt

    try:
        bot = LogAnalysisBot(
            model=f"azure-openai/{os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']}",
            folder_to_mount=codebase_path,
            token_provider=token_provider,
        )
        result = bot.run(**run_kwargs)
    except Exception as error:
        if not is_docker_access_error(error):
            raise
        log(
            "MicrobotsLogAnalyzer: Docker-compatible daemon was not accessible "
            "while starting the Microbots sandbox.",
            file=sys.stderr,
        )
        log(f"Details: {error}", file=sys.stderr)
        return 1

    message = result.result or result.error or ""
    if args.output_file:
        write_text_file(args.output_file, str(message))
        log(f"MicrobotsLogAnalyzer: wrote analysis output to {args.output_file}")

    log("##[section]MicrobotsLogAnalyzer: LLM analysis")
    log("============================================================")
    log("MICROBOTS LOG ANALYSIS")
    log("============================================================")
    for paragraph in str(message).splitlines() or [""]:
        if paragraph.strip():
            for line in textwrap.wrap(paragraph, width=125) or [""]:
                log(safe_analysis_line(line))
        else:
            log("")
    log("============================================================")

    return 0 if result.status else 1


if __name__ == "__main__":
    sys.exit(main())
