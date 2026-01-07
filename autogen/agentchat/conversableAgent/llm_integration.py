# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import functools
import warnings
from typing import TYPE_CHECKING, Any

from ...cache.cache import AbstractCache
from ...io.base import IOStream
from ...llm_config.client import ModelClient
from ...oai.client import OpenAIWrapper

if TYPE_CHECKING:
    from ..agent import Agent
    from .base import ConversableAgentBase


class LLMIntegrationMixin:
    """Mixin class for LLM integration functionality."""

    def generate_oai_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: "Agent" | None = None,
        config: OpenAIWrapper | None = None,
        **kwargs: Any,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """Generate a reply using autogen.oai."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        # Process messages before sending to LLM, hook point for llm input monitoring
        processed_messages = self._process_llm_input(self._oai_system_message + messages)
        if processed_messages is None:
            return True, {"content": "LLM call blocked by safeguard", "role": "assistant"}

        extracted_response = self._generate_oai_reply_from_client(
            client, processed_messages, self.client_cache, **kwargs
        )

        # Process LLM response
        if extracted_response is not None:
            processed_extracted_response = self._process_llm_output(extracted_response)
            if processed_extracted_response is None:
                raise ValueError("safeguard_llm_outputs hook returned None")
            extracted_response = processed_extracted_response

        return (False, None) if extracted_response is None else (True, extracted_response)

    def _generate_oai_reply_from_client(
        self: "ConversableAgentBase", llm_client, messages, cache, **kwargs: Any
    ) -> str | dict[str, Any] | None:
        # unroll tool_responses
        all_messages = []
        for message in messages:
            tool_responses = message.get("tool_responses", [])
            if tool_responses:
                all_messages += tool_responses
                # tool role on the parent message means the content is just concatenation of all of the tool_responses
                if message.get("role") != "tool":
                    all_messages.append({key: message[key] for key in message if key != "tool_responses"})
            else:
                all_messages.append(message)

        # TODO: #1143 handle token limit exceeded error
        response = llm_client.create(
            context=messages[-1].pop("context", None),
            messages=all_messages,
            cache=cache,
            agent=self,
            **kwargs,
        )
        extracted_response = llm_client.extract_text_or_completion_object(response)[0]

        if extracted_response is None:
            warnings.warn(f"Extracted_response from {response} is None.", UserWarning)
            return None
        # ensure function and tool calls will be accepted when sent back to the LLM
        if not isinstance(extracted_response, str) and hasattr(extracted_response, "model_dump"):
            extracted_response = extracted_response.model_dump()
        if isinstance(extracted_response, dict):
            if extracted_response.get("function_call"):
                extracted_response["function_call"]["name"] = self._normalize_name(
                    extracted_response["function_call"]["name"]
                )
            for tool_call in extracted_response.get("tool_calls") or []:
                tool_call["function"]["name"] = self._normalize_name(tool_call["function"]["name"])
                # Remove id and type if they are not present.
                # This is to make the tool call object compatible with Mistral API.
                if tool_call.get("id") is None:
                    tool_call.pop("id")
                if tool_call.get("type") is None:
                    tool_call.pop("type")
        return extracted_response

    async def a_generate_oai_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: "Agent" | None = None,
        config: Any | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """Generate a reply using autogen.oai asynchronously."""
        iostream = IOStream.get_default()

        def _generate_oai_reply(
            self, iostream: IOStream, *args: Any, **kwargs: Any
        ) -> tuple[bool, str | dict[str, Any] | None]:
            with IOStream.set_default(iostream):
                return self.generate_oai_reply(*args, **kwargs)

        return await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(
                _generate_oai_reply, self=self, iostream=iostream, messages=messages, sender=sender, config=config
            ),
        )

    def _reflection_with_llm(
        self: "ConversableAgentBase",
        prompt,
        messages,
        llm_agent: "Agent" | None = None,
        cache: AbstractCache | None = None,
        role: str | None = None,
    ) -> str:
        """Get a chat summary using reflection with an llm client based on the conversation history.

        Args:
            prompt (str): The prompt (in this method it is used as system prompt) used to get the summary.
            messages (list): The messages generated as part of a chat conversation.
            llm_agent: the agent with an llm client.
            cache (AbstractCache or None): the cache client to be used for this conversation.
            role (str): the role of the message, usually "system" or "user". Default is "system".
        """
        if not role:
            role = "system"

        system_msg = [
            {
                "role": role,
                "content": prompt,
            }
        ]

        messages = messages + system_msg
        if llm_agent and llm_agent.client is not None:
            llm_client = llm_agent.client
        elif self.client is not None:
            llm_client = self.client
        else:
            raise ValueError("No OpenAIWrapper client is found.")
        response = self._generate_oai_reply_from_client(llm_client=llm_client, messages=messages, cache=cache)
        return response

    @staticmethod
    def _reflection_with_llm_as_summary(sender, recipient, summary_args):
        prompt = summary_args.get("summary_prompt")
        prompt = ConversableAgentBase.DEFAULT_SUMMARY_PROMPT if prompt is None else prompt
        if not isinstance(prompt, str):
            raise ValueError("The summary_prompt must be a string.")
        msg_list = recipient.chat_messages_for_summary(sender)
        agent = sender if recipient is None else recipient
        role = summary_args.get("summary_role", None)
        if role and not isinstance(role, str):
            raise ValueError("The summary_role in summary_arg must be a string.")
        try:
            summary = sender._reflection_with_llm(
                prompt, msg_list, llm_agent=agent, cache=summary_args.get("cache"), role=role
            )
        except Exception as e:
            warnings.warn(
                f"Cannot extract summary using reflection_with_llm: {e}. Using an empty str as summary.", UserWarning
            )
            summary = ""
        return summary

    @staticmethod
    def _last_msg_as_summary(sender, recipient, summary_args) -> str:
        """Get a chat summary from the last message of the recipient."""
        summary = ""
        try:
            content = recipient.last_message(sender)["content"]
            if isinstance(content, str):
                summary = content.replace("TERMINATE", "")
            elif isinstance(content, list):
                # Remove the `TERMINATE` word in the content list.
                summary = "\n".join(
                    x["text"].replace("TERMINATE", "") for x in content if isinstance(x, dict) and "text" in x
                )
        except (IndexError, AttributeError) as e:
            warnings.warn(f"Cannot extract summary using last_msg: {e}. Using an empty str as summary.", UserWarning)
        return summary

    def register_model_client(self: "ConversableAgentBase", model_client_cls: ModelClient, **kwargs: Any):
        """Register a model client.

        Args:
            model_client_cls: A custom client class that follows the Client interface
            **kwargs: The kwargs for the custom client class to be initialized with
        """
        self.client.register_model_client(model_client_cls, **kwargs)
