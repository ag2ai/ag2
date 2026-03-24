# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import logging
from pathlib import Path
from typing import Any

from .... import ConversableAgent
from ....agentchat.agent import Agent
from ....doc_utils import export_module
from .a2a_helpers import A2UI_DEFAULT_DELIMITER
from .actions import A2UIAction
from .response_parser import A2UIParseResult, A2UIResponseParser, A2UIValidationResult
from .schema_manager import A2UISchemaManager

__all__ = ["A2UIAgent"]

logger = logging.getLogger(__name__)


@export_module("autogen.agents.experimental")
class A2UIAgent(ConversableAgent):
    """An AG2 agent that produces A2UI rich UI output.

    Supports v0.9 of the A2UI protocol with the basic catalog and support for
    custom catalogs.

    Has an internal creation and validation against the schema with automatic
    retry.

    Example:
        ```python
        from autogen.agents.experimental import A2UIAgent

        agent = A2UIAgent(
            name="ui_agent",
            llm_config={"api_type": "google", "model": "gemini-3.1-flash-lite-preview"},
            validate_responses=True,
            validation_retries=2,
        )
        ```
    """

    DEFAULT_SYSTEM_MESSAGE = (
        "You are a helpful AI assistant that can generate rich user interfaces "
        "using the A2UI protocol. When the user's request would benefit from a "
        "visual UI (cards, forms, lists, etc.), generate A2UI output. "
        "For simple text responses, respond normally without A2UI."
    )

    def __init__(
        self,
        name: str = "a2ui_agent",
        system_message: str | None = None,
        protocol_version: str = "v0.9",
        custom_catalog: str | Path | dict[str, Any] | None = None,
        custom_catalog_rules: str | None = None,
        include_schema_in_prompt: bool = True,
        include_rules_in_prompt: bool = True,
        response_delimiter: str = A2UI_DEFAULT_DELIMITER,
        validate_responses: bool = True,
        validation_retries: int = 1,
        actions: list[A2UIAction] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the A2UIAgent.

        Args:
            name: The agent name.
            system_message: Custom system message. If None, uses DEFAULT_SYSTEM_MESSAGE
                plus the A2UI prompt section.
            protocol_version: The A2UI protocol version to target. Currently only "v0.9"
                is supported.
            custom_catalog: A custom catalog that extends the basic catalog. Can be a
                file path (str/Path) to a JSON file, or a dict with the schema directly.
                Must include a ``$id`` field (used as the catalogId in A2UI messages).
                The custom catalog's components are available alongside all basic catalog
                components.
            custom_catalog_rules: Plain-text rules for the custom catalog components,
                appended to the basic catalog rules in the system prompt.
            include_schema_in_prompt: Whether to include the full JSON schema in the
                system prompt (helps with validation but uses more tokens).
            include_rules_in_prompt: Whether to include the catalog rules in the prompt.
            response_delimiter: The delimiter separating text from A2UI JSON in responses.
            validate_responses: Whether to validate A2UI output against the schema
                and retry on failure. Requires the ``a2ui`` extra (jsonschema).
            validation_retries: Maximum number of retry attempts when validation fails.
                Only used when ``validate_responses=True``. Default is 1.
            actions: List of ``A2UIAction`` definitions for button handling.
                Server actions (``action_type="event"``) dispatch events to the
                server — routed to a tool via ``tool_name`` or handled by the LLM.
                Client actions (``action_type="functionCall"``) execute client-side
                functions like ``openUrl`` without server communication.
                Available actions are auto-injected into the system prompt.
            **kwargs: Additional arguments passed to ConversableAgent.
        """
        self._schema_manager = A2UISchemaManager(
            protocol_version=protocol_version,
            custom_catalog=custom_catalog,
            custom_catalog_rules=custom_catalog_rules,
        )

        self._response_parser = A2UIResponseParser(
            delimiter=response_delimiter,
            server_to_client_schema=(self._schema_manager.server_to_client_schema if validate_responses else None),
            schema_registry=(self._schema_manager.build_schema_registry() if validate_responses else None),
            version_string=self._schema_manager.version_string,
            component_schemas=(self._schema_manager.get_component_schemas() if validate_responses else None),
            catalog_id=(self._schema_manager.catalog_id if validate_responses else None),
        )

        self._response_delimiter = response_delimiter
        self._validate_responses = validate_responses
        self._validation_retries = validation_retries
        self._actions = actions or []

        # Store the A2UI prompt section separately so it survives UpdateSystemMessage
        self._a2ui_prompt_section = self._schema_manager.generate_prompt_section(
            include_schema=include_schema_in_prompt,
            include_rules=include_rules_in_prompt,
            response_delimiter=response_delimiter,
            actions=self._actions,
        )

        # Build the full system message
        base_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        full_system_message = f"{base_message}\n\n{self._a2ui_prompt_section}"

        super().__init__(name=name, system_message=full_system_message, **kwargs)

        # Register validation reply when validation is enabled
        if validate_responses:
            self.register_reply(
                [Agent, None],
                A2UIAgent._a2ui_validating_reply,
            )

    def _validate_a2ui_response(self, response_text: str) -> tuple[A2UIParseResult, list[str] | None]:
        """Parse and validate an A2UI response.

        Returns:
            A tuple of (parse_result, validation_errors). If validation_errors
            is None, the response is valid (or has no A2UI content).
        """
        parse_result = self._response_parser.parse(response_text)

        if not parse_result.has_a2ui:
            return parse_result, None

        if parse_result.parse_error:
            return parse_result, [parse_result.parse_error]

        validation_result = self._response_parser.validate(parse_result.operations)
        if validation_result.is_valid:
            return parse_result, None

        return parse_result, validation_result.errors

    def _build_retry_messages(
        self,
        working_messages: list[dict[str, Any]],
        response_text: str,
        parse_result: A2UIParseResult,
        validation_errors: list[str],
    ) -> list[dict[str, Any]]:
        """Append the invalid response and error feedback to the message list for retry."""
        error_result = A2UIValidationResult(is_valid=False, errors=validation_errors)
        feedback = self._response_parser.format_validation_error(parse_result, error_result)

        retry_messages = list(working_messages)
        retry_messages.append({"role": "assistant", "content": response_text})
        retry_messages.append({"role": "user", "content": feedback})
        return retry_messages

    def _a2ui_validating_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """Reply function that validates A2UI output and retries on failure.

        Implements the v0.9 prompt-generate-validate feedback loop:
        1. Generate LLM response
        2. Parse for A2UI content
        3. If valid (or no A2UI), return as-is
        4. If invalid, feed errors back and retry
        5. On exhaustion, return text-only (graceful degradation)
        """
        working_messages = list(messages) if messages else []

        for attempt in range(1 + self._validation_retries):
            final, response = self.generate_oai_reply(working_messages, sender)
            if not final or response is None:
                return False, None

            response_text = response if isinstance(response, str) else response.get("content", "")
            if not isinstance(response_text, str):
                return True, response

            parse_result, validation_errors = self._validate_a2ui_response(response_text)

            if validation_errors is None:
                return True, response

            # Last attempt — return text-only (graceful degradation)
            if attempt >= self._validation_retries:
                logger.warning(
                    "A2UI validation failed after %d attempt(s). Returning text-only response.",
                    attempt + 1,
                )
                return True, parse_result.text

            logger.info("A2UI validation failed (attempt %d/%d). Retrying.", attempt + 1, self._validation_retries)
            logger.debug("Validation errors: %s", validation_errors)

            working_messages = self._build_retry_messages(
                working_messages, response_text, parse_result, validation_errors
            )

        return True, response if response else "Sorry, I was not able to generate a valid UI."

    def update_system_message(self, system_message: str) -> None:
        """Update the system message, preserving the A2UI prompt section.

        If the new system message doesn't already contain the A2UI prompt section,
        it will be appended automatically. This ensures that A2UI instructions
        survive ``UpdateSystemMessage`` hooks and dynamic system message updates.

        Args:
            system_message: The new system message content.
        """
        if self._a2ui_prompt_section not in system_message:
            system_message = f"{system_message}\n\n{self._a2ui_prompt_section}"
        super().update_system_message(system_message)

    @property
    def a2ui_prompt_section(self) -> str:
        """The A2UI prompt section that is appended to the system message."""
        return self._a2ui_prompt_section

    @property
    def schema_manager(self) -> A2UISchemaManager:
        """The schema manager for this agent."""
        return self._schema_manager

    @property
    def response_parser(self) -> A2UIResponseParser:
        """The response parser for this agent."""
        return self._response_parser

    @property
    def protocol_version(self) -> str:
        """The A2UI protocol version this agent targets."""
        return self._schema_manager.protocol_version

    @property
    def catalog_id(self) -> str:
        """The A2UI catalog ID this agent uses."""
        return self._schema_manager.catalog_id

    @property
    def actions(self) -> list[A2UIAction]:
        """The registered A2UI actions."""
        return self._actions

    def get_action(self, name: str) -> A2UIAction | None:
        """Look up a registered action by name."""
        for action in self._actions:
            if action.name == name:
                return action
        return None

    @property
    def validation_retries(self) -> int:
        """Maximum number of validation retry attempts."""
        return self._validation_retries
