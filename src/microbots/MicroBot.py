from collections.abc import Iterable
import json
import os
from pprint import pformat
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Optional

from microbots.constants import ModelProvider
from microbots.environment.local_docker.LocalDockerEnvironment import (
    LocalDockerEnvironment,
)
from microbots.llm.anthropic_api import AnthropicApi
from microbots.llm.azure_openai_api import AzureOpenAIApi
from microbots.llm.openai_api import OpenAIApi
from microbots.llm.ollama_local import OllamaLocal
from microbots.llm.llm import llm_output_format_str
from microbots.tools.tool import ToolAbstract, get_tool_from_call
from microbots.extras.mount import Mount, MountType
from microbots.utils.logger import LogLevelEmoji
from microbots.utils.network import get_free_port

logger = getLogger(" MicroBot ")

system_prompt_common = f"""
You are a helpful agent well versed in software development and debugging.

You will be provided with a coding or debugging task to complete inside a sandboxed shell environment.
There is a shell session open for you.
You will be provided with a task and you should achieve it using the shell commands.
All your response must be in the following json format:
{llm_output_format_str}
The properties ( task_done, thoughts, command ) are mandatory on each response.
Give the command one at a time to solve the given task. As long as you're not done with the task, set task_done to false.
When you are sure that the task is completed, set task_done to true, set command to empty string and provide your final thoughts in the thoughts field.
Don't add any chat or extra messages outside the json format. Because the system will parse only the json response.
Any of your thoughts must be in the 'thoughts' field.

after each command, the system will execute the command and respond to you with the output.
Ensure to run only one command at a time.
NEVER use commands that produce large amounts of output or take a long time to run to avoid exceeding context limits.
Use specific patterns: 'find <path> -name "*.c" -maxdepth 2' instead of recursive exploration.
No human is involved in the task. So, don't seek human intervention.

Remember following important points
1. If a command fails, analyze the error message and provide an alternative command in your next response. Same command will not pass again.
2. Avoid using recursive commands like 'ls -R', 'rm -rf', 'tree', or 'find' without depth limits as they can produce excessive output or be destructive.
3. You cannot run any interactive commands like vim, nano, etc.
"""


class BotType(StrEnum):
    READING_BOT = "READING_BOT"
    WRITING_BOT = "WRITING_BOT"
    BROWSING_BOT = "BROWSING_BOT"
    CUSTOM_BOT = "CUSTOM_BOT"
    LOG_ANALYSIS_BOT = "LOG_ANALYSIS_BOT"


@dataclass
class BotRunResult:
    status: bool
    result: str | None
    error: Optional[str]


class MicroBot:
    """
    The core Microbot class.

    MicroBot class is the core class representing the autonomous agent. Other bots are extensions of this class.
    If you want to create a custom bot, you can directly use this class or extend it into your own bot class.

    Attributes
    ----------
        model : str
            The model to use for the bot, in the format <provider>/<model_name>.
        bot_type : BotType
            The type of bot being created. It's unused. Will be removed soon.
        system_prompt : Optional[str]
            The system prompt to guide the bot's behavior.
        environment : Optional[any]
            The execution environment for the bot. If not provided, a default
            LocalDockerEnvironment will be created.
        additional_tools : Optional[list[ToolAbstract]]
            A list of additional tools to install in the bot's environment.
        folder_to_mount : Optional[Mount]
            A folder to mount into the bot's environment. The bot will be given
            access to this folder based on the specified permissions. This will
            be the main code folder where the bot will work. Additional folders
            can be mounted during the run() method. Refer to `Mount` class
            regarding the directory structure and permission details. Defaults
            to None.
    """

    def __init__(
        self,
        model: str,
        bot_type: BotType = BotType.CUSTOM_BOT,
        system_prompt: Optional[str] = None,
        environment: Optional[any] = None,
        additional_tools: Optional[list[ToolAbstract]] = None,
        folder_to_mount: Optional[Mount] = None,
        token_provider: Optional[any] = None,
    ):
        """
        Init function for MicroBot class.

        Parameters
        ----------
            model :str
                The model to use for the bot, in the format <provider>/<model_name>.
            bot_type :BotType
                The type of bot being created. It's unused. Will be removed soon.
            system_prompt :Optional[str]
                The system prompt to guide the bot's behavior. Defaults to None.
            environment :Optional[any]
                The execution environment for the bot. If not provided, a default
                LocalDockerEnvironment will be created.
            additional_tools :Optional[list[ToolAbstract]]
                A list of additional tools to install in the bot's environment.
                Defaults to None (treated as an empty list).
            folder_to_mount :Optional[Mount]
                A folder to mount into the bot's environment. The bot will be given
                access to this folder based on the specified permissions. This will
                be the main code folder where the bot will work. Additional folders
                can be mounted using the run() method. Refer to `Mount` class
                regarding the directory structure and permission details. Defaults
                to None.

                Note: Supports only mount type MountType.MOUNT for now.
        """

        self.folder_to_mount = folder_to_mount

        # TODO : Need to check on the purpose of variable `mounted`
        # 1. If we allow user to mount multiple directories,
        # we should able to get it as an argument and store them in self.mounted.
        # This require changes in _create_environment to handle multiple mount directories or files.
        #
        # 2. We should let user to mount only one directory. In that case self.mounted may not be required.
        # Just one self.folder_to_mount and necessary extra mounts at the derived class similar to LogAnalyticsBot.

        self.mounted = []
        if folder_to_mount is not None:
            self._validate_folder_to_mount(folder_to_mount)
            self.mounted.append(folder_to_mount)

        self.system_prompt = system_prompt
        self.model = model
        self.bot_type = bot_type
        self.environment = environment
        self.additional_tools = additional_tools or []

        # TODO: Replace iteration_count and max_iterations with cost management.
        # Iteration count represents overall LLM interactions including interactions
        # done by sub agents.
        self.iteration_count = 0
        self.max_iterations = 0
        self.step_count = 0

        self._validate_model_and_provider(model)
        self.model_provider = model.split("/")[0]
        self.deployment_name = model.split("/")[1]

        # Only auto-create token provider from env for providers that support Azure AD tokens.
        if token_provider is not None:
            self.token_provider = token_provider
        elif (
            os.getenv("AZURE_AUTH_METHOD", "").strip().lower() == "azure_ad"
            and self.model_provider == ModelProvider.AZURE_OPENAI
        ):
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError:
                raise ImportError(
                    "Azure AD authentication requires the 'azure-identity' package. "
                    "Install it with: pip install microbots[azure_ad]"
                )
            credential = DefaultAzureCredential()
            self.token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
        else:
            self.token_provider = None

        if not self.environment:
            self._create_environment(self.folder_to_mount)

        self._create_llm()

        for tool in self.additional_tools:
            tool.install_tool(self.environment)
            tool.verify_tool_installation(self.environment)

    def run(
        self,
        task: str,
        additional_mounts: Optional[list[Mount]] = None,
        max_iterations: int = 20,
        timeout_in_seconds: int = 200
    ) -> BotRunResult:

        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")

        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.step_count = 0

        for tool in self.additional_tools:
            tool.setup_tool(self.environment)

        for mount in additional_mounts or []:
            self._mount_additional(mount)

        # start timer
        start_time = time.time()
        timeout = timeout_in_seconds
        llm_response = self.llm.ask(task)
        return_value = BotRunResult(
            status=False,
            result=None,
            error="Did not complete",
        )
        logger.info("%s TASK STARTED : %s...", LogLevelEmoji.INFO, task)
        if llm_response.task_done is True:
            if llm_response.thoughts:
                logger.info(
                    f" LLM final thoughts: {llm_response.thoughts}",
                )
            logger.info("TASK COMPLETED : %s...", task[0:15])
            return BotRunResult(
                status=True, 
                result=llm_response.thoughts, 
                error=None
            )
        while llm_response.task_done is False:
            # increment iteration count
            self.iteration_count += 1
            self.step_count += 1
            if self.iteration_count >= max_iterations:
                logger.error(
                    "Max iterations %d reached. Exiting without completing the task. Last response: %s",
                    max_iterations,
                    llm_response,
                )
                return_value.error = f"Max iterations {max_iterations} reached"
                return return_value

            # check if timeout has reached
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > timeout:
                logger.error(
                    "Iteration %d with response %s - Exiting without running command as timeout reached",
                    self.iteration_count,
                    llm_response,
                )
                return_value.error = f"Timeout of {timeout} seconds reached"
                return return_value


            logger.info("%s Step-%d %s", "-" * 20, self.step_count, "-" * 20)
            if llm_response.thoughts:
                logger.info(
                    f" 💭  LLM thoughts: {llm_response.thoughts}",
                )
            logger.info(
                f" ➡️  LLM tool call : {json.dumps(llm_response.command)}",
            )

            # Validate command for dangerous operations
            is_safe, explanation = self._is_safe_command(llm_response.command)
            if not is_safe:
                error_msg = f"Dangerous command detected and blocked: {llm_response.command}\n{explanation}"
                logger.info("%s %s", LogLevelEmoji.WARNING, error_msg)
                llm_response = self.llm.ask(f"COMMAND_ERROR: {error_msg}\nPlease provide a safer alternative command.")
                continue

            tool = get_tool_from_call(llm_response.command, self.additional_tools)
            if tool:
                llm_command_output = tool.invoke(llm_response.command, self)
            else:
                llm_command_output = self.environment.execute(llm_response.command)

            logger.debug(
                    " 🔧  Command executed.\nReturn Code: %d\nStdout:\n%s\nStderr:\n%s",
                    llm_command_output.return_code,
                    llm_command_output.stdout,
                    llm_command_output.stderr,
                )

            if llm_command_output.return_code == 0:
                if llm_command_output.stdout:
                    output_text = llm_command_output.stdout
                    # HACK: anthropic-text-editor tool extra formats the output
                    try:
                        output_json = json.loads(llm_command_output.stdout)
                        if isinstance(output_json, Iterable) and "content" in output_json:
                            output_text = pformat(output_json["content"])
                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        logger.warning("Failed to parse command output as JSON, using raw stdout")
                        logger.debug("Error details: %s", str(e))
                else:
                    output_text = f"Command executed successfully with no output\nreturn code: {llm_command_output.return_code}\nstdout: {llm_command_output.stdout}\nstderr: {llm_command_output.stderr}"
            else:
                output_text = f"COMMAND EXECUTION FAILED\nreturn code: {llm_command_output.return_code}\nstdout: {llm_command_output.stdout}\nstderr: {llm_command_output.stderr}"

            logger.info(" ⬅️  Command output:\n%s", output_text)
            llm_response = self.llm.ask(output_text)

        if llm_response.thoughts:
            logger.info(
                f" 💭  LLM final thoughts: {llm_response.thoughts}",
            )
        logger.info("🔚 TASK COMPLETED : %s...", task[0:15])
        return BotRunResult(status=True, result=llm_response.thoughts, error=None)

    def _mount_additional(self, mount: Mount):
        if mount.mount_type != MountType.COPY:
            logger.error(
                "%s Only COPY mount type is supported for additional mounts for now",
                LogLevelEmoji.ERROR,
            )
            raise ValueError(
                "Only COPY mount type is supported for additional mounts for now"
            )

        self.mounted.append(mount)
        copy_to_container_result = self.environment.copy_to_container(
            mount.host_path_info.abs_path, mount.sandbox_path
        )
        if copy_to_container_result is False:
            raise ValueError(
                f"Failed to copy additional mount to container: {mount.host_path_info.abs_path} -> {mount.sandbox_path}"
            )

    # TODO : pass the sandbox path
    def _create_environment(self, folder_to_mount: Optional[Mount]):
        free_port = get_free_port()

        self.environment = LocalDockerEnvironment(
            port=free_port,
            folder_to_mount=folder_to_mount,
        )

    def _create_llm(self):
        # Append tool usage instructions to system prompt
        system_prompt_with_tools = self.system_prompt if self.system_prompt else ""
        if self.additional_tools:
            for tool in self.additional_tools:
                if tool.usage_instructions_to_llm:
                    system_prompt_with_tools += f"\n\n{tool.usage_instructions_to_llm}"

        if self.model_provider == ModelProvider.AZURE_OPENAI:
            self.llm = AzureOpenAIApi(
                system_prompt=system_prompt_with_tools, deployment_name=self.deployment_name,
                token_provider=self.token_provider,
            )
        elif self.model_provider == ModelProvider.OPENAI:
            self.llm = OpenAIApi(
                system_prompt=system_prompt_with_tools, deployment_name=self.deployment_name,
            )
        elif self.model_provider == ModelProvider.OLLAMA_LOCAL:
            self.llm = OllamaLocal(
                system_prompt=system_prompt_with_tools, model_name=self.deployment_name
            )
        elif self.model_provider == ModelProvider.ANTHROPIC:
            self.llm = AnthropicApi(
                system_prompt=system_prompt_with_tools, deployment_name=self.deployment_name,
                token_provider=self.token_provider,
            )
        # No Else case required as model provider is already validated using _validate_model_and_provider

    def _validate_model_and_provider(self, model):
        # Ensure it has only only slash
        if model.count("/") != 1:
            raise ValueError("Model should be in the format <provider>/<model_name>")
        provider = model.split("/")[0]
        if provider not in [e.value for e in ModelProvider]:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _validate_folder_to_mount(self, folder_to_mount: Mount):
        if folder_to_mount.mount_type != MountType.MOUNT:
            logger.error(
                "%s Only MOUNT mount type is supported for folder_to_mount",
                LogLevelEmoji.ERROR,
            )
            raise ValueError(
                "Only MOUNT mount type is supported for folder_to_mount"
            )

    def _get_dangerous_command_explanation(self, command: str) -> Optional[str]:
        """Provides detailed explanation for why a command is dangerous and suggests alternatives.

        Args:
            command: The shell command to analyze

        Returns:
            str: Explanation with reason and alternative, or None if command is safe
        """
        # Handle invalid commands (empty, None, or non-string)
        if not command or not isinstance(command, str):
            return "REASON: Empty or invalid command provided\nALTERNATIVE: Provide a valid shell command"

        stripped_command = command.strip()
        if not stripped_command:
            return "REASON: Empty or whitespace-only command provided\nALTERNATIVE: Provide a valid shell command"

        # Define dangerous command patterns with detailed explanations
        # Note: Don't convert to lowercase before checking, as we need case-sensitive pattern matching
        dangerous_checks = [
            {
                'pattern': r'\bls\s+(?:[^-]*\s+)?-[a-z]*[rR](?:[a-z]*\b|\s|$)',
                'reason': 'Recursive ls commands (ls -R) can generate massive output in large repositories, exceeding context limits',
                'alternative': 'Use targeted paths like "ls drivers/block/" or "ls -la <specific-directory>" instead'
            },
            {
                'pattern': r'\btree\b',
                'reason': 'Tree command recursively lists entire directory structures, which can exceed context limits',
                'alternative': 'Use "ls -la <specific-directory>" or "find <path> -maxdepth 2 -type d" for controlled exploration'
            },
            {
                'pattern': r'\brm\s+(?:[^-]*\s+)?-[a-z]*[rR](?:[a-z]*\b|\s|$)',
                'reason': 'Recursive rm commands (rm -r/-rf) can delete entire directory trees, which is destructive',
                'alternative': 'Delete specific files individually or use "rm <specific-file>" to avoid accidental data loss'
            },
            {
                'pattern': r'\brm\s+--recursive\b',
                'reason': 'Recursive rm commands can delete entire directory trees, which is destructive',
                'alternative': 'Delete specific files individually or use "rm <specific-file>" to avoid accidental data loss'
            },
            {
                'pattern': r'\bfind\b(?!.*-maxdepth)',
                'reason': 'Find command without -maxdepth can recursively search entire filesystems, causing excessive output',
                'alternative': 'Use "find <path> -name "*.ext" -maxdepth 2" to limit search depth and control output size'
            },
        ]

        for check in dangerous_checks:
            if re.search(check['pattern'], stripped_command, re.IGNORECASE):
                return f"REASON: {check['reason']}\nALTERNATIVE: {check['alternative']}"

        return None

    def _is_safe_command(self, command: str) -> tuple[bool, Optional[str]]:
        """Validates if a command is safe to execute.

        A command is considered safe if it:
        - Is not a recursive command (ls -R, rm -rf, tree, find without -maxdepth)
        - Does not risk generating excessive output or destructive actions

        Args:
            command: The shell command to validate

        Returns:
            tuple[bool, Optional[str]]: A tuple of (is_safe, explanation) where:
                - is_safe: True if command is safe to execute, False otherwise
                - explanation: Detailed explanation if dangerous, None if safe
        """
        explanation = self._get_dangerous_command_explanation(command)
        is_safe = explanation is None
        return is_safe, explanation
