# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Literal

from ...code_utils import (
    PYTHON_VARIANTS,
    UNKNOWN,
    execute_code,
    extract_code,
    infer_lang,
)
from ...events.agent_events import (
    ExecuteCodeBlockEvent,
    GenerateCodeExecutionReplyEvent,
)
from ...io.base import IOStream
from ..agent import Agent

if TYPE_CHECKING:
    from .base import ConversableAgentBase


class CodeExecutionMixin:
    """Mixin class for code execution functionality"""

    def _generate_code_execution_reply_using_executor(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: dict[str, Any] | Literal[False] | None = None,
    ):
        """Generate a reply using code executor."""
        iostream = IOStream.get_default()

        if config is not None:
            raise ValueError("config is not supported for _generate_code_execution_reply_using_executor.")
        if self._code_execution_config is False:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        last_n_messages = self._code_execution_config.get("last_n_messages", "auto")

        if not (isinstance(last_n_messages, (int, float)) and last_n_messages >= 0) and last_n_messages != "auto":
            raise ValueError("last_n_messages must be either a non-negative integer, or the string 'auto'.")

        num_messages_to_scan = last_n_messages
        if last_n_messages == "auto":
            # Find when the agent last spoke
            num_messages_to_scan = 0
            for message in reversed(messages):
                if "role" not in message or message["role"] != "user":
                    break
                else:
                    num_messages_to_scan += 1
        num_messages_to_scan = min(len(messages), num_messages_to_scan)
        messages_to_scan = messages[-num_messages_to_scan:]

        # iterate through the last n messages in reverse
        # if code blocks are found, execute the code blocks and return the output
        # if no code blocks are found, continue
        for message in reversed(messages_to_scan):
            if not message["content"]:
                continue
            code_blocks = self._code_executor.code_extractor.extract_code_blocks(message["content"])
            if len(code_blocks) == 0:
                continue

            iostream.send(GenerateCodeExecutionReplyEvent(code_blocks=code_blocks, sender=sender, recipient=self))

            # found code blocks, execute code.
            code_result = self._code_executor.execute_code_blocks(code_blocks)
            exitcode2str = "execution succeeded" if code_result.exit_code == 0 else "execution failed"
            return True, f"exitcode: {code_result.exit_code} ({exitcode2str})\nCode output: {code_result.output}"

        return False, None

    def generate_code_execution_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: dict[str, Any] | Literal[False] | None = None,
    ):
        """Generate a reply using code execution."""
        code_execution_config = config if config is not None else self._code_execution_config
        if code_execution_config is False:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        last_n_messages = code_execution_config.pop("last_n_messages", "auto")

        if not (isinstance(last_n_messages, (int, float)) and last_n_messages >= 0) and last_n_messages != "auto":
            raise ValueError("last_n_messages must be either a non-negative integer, or the string 'auto'.")

        messages_to_scan = last_n_messages
        if last_n_messages == "auto":
            # Find when the agent last spoke
            messages_to_scan = 0
            for i in range(len(messages)):
                message = messages[-(i + 1)]
                if "role" not in message or message["role"] != "user":
                    break
                else:
                    messages_to_scan += 1

        # iterate through the last n messages in reverse
        # if code blocks are found, execute the code blocks and return the output
        # if no code blocks are found, continue
        for i in range(min(len(messages), messages_to_scan)):
            message = messages[-(i + 1)]
            if not message["content"]:
                continue
            code_blocks = extract_code(message["content"])
            if len(code_blocks) == 1 and code_blocks[0][0] == UNKNOWN:
                continue

            # found code blocks, execute code and push "last_n_messages" back
            exitcode, logs = self.execute_code_blocks(code_blocks)
            code_execution_config["last_n_messages"] = last_n_messages
            exitcode2str = "execution succeeded" if exitcode == 0 else "execution failed"
            return True, f"exitcode: {exitcode} ({exitcode2str})\nCode output: {logs}"

        # no code blocks are found, push last_n_messages back and return.
        code_execution_config["last_n_messages"] = last_n_messages

        return False, None

    def run_code(self: "ConversableAgentBase", code: str, **kwargs: Any) -> tuple[int, str, str | None]:
        """Run the code and return the result.

        Override this function to modify the way to run the code.

        Args:
            code (str): the code to be executed.
            **kwargs: other keyword arguments.

        Returns:
            A tuple of (exitcode, logs, image).
            exitcode (int): the exit code of the code execution.
            logs (str): the logs of the code execution.
            image (str or None): the docker image used for the code execution.
        """
        return execute_code(code, **kwargs)

    def execute_code_blocks(self: "ConversableAgentBase", code_blocks):
        """Execute the code blocks and return the result."""
        iostream = IOStream.get_default()

        logs_all = ""
        for i, code_block in enumerate(code_blocks):
            lang, code = code_block
            if not lang:
                lang = infer_lang(code)

            iostream.send(ExecuteCodeBlockEvent(code=code, language=lang, code_block_count=i, recipient=self))

            if lang in ["bash", "shell", "sh"]:
                exitcode, logs, image = self.run_code(code, lang=lang, **self._code_execution_config)
            elif lang in PYTHON_VARIANTS:
                filename = code[11 : code.find("\n")].strip() if code.startswith("# filename: ") else None
                exitcode, logs, image = self.run_code(
                    code,
                    lang="python",
                    filename=filename,
                    **self._code_execution_config,
                )
            else:
                # In case the language is not supported, we return an error message.
                exitcode, logs, image = (
                    1,
                    f"unknown language {lang}",
                    None,
                )
                # raise NotImplementedError
            if image is not None:
                self._code_execution_config["use_docker"] = image
            logs_all += "\n" + logs
            if exitcode != 0:
                return exitcode, logs_all
        return exitcode, logs_all
