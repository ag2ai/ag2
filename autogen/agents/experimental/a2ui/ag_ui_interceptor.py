# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from uuid import uuid4

from ....import_utils import optional_import_block
from .a2a_helpers import A2UI_DEFAULT_ACTIVITY_TYPE, A2UI_DEFAULT_DELIMITER, A2UI_DEFAULT_VERSION
from .response_parser import A2UIResponseParser

with optional_import_block():
    from ag_ui.core import ActivitySnapshotEvent, BaseEvent

    from ....agentchat.remote import ServiceResponse

__all__ = ["a2ui_event_interceptor", "create_a2ui_event_interceptor"]

logger = logging.getLogger(__name__)


def create_a2ui_event_interceptor(
    delimiter: str = A2UI_DEFAULT_DELIMITER,
    version_string: str = A2UI_DEFAULT_VERSION,
    activity_type: str = A2UI_DEFAULT_ACTIVITY_TYPE,
) -> Callable[[ServiceResponse], AsyncIterator[BaseEvent]]:
    """Create an AG-UI event interceptor that extracts A2UI JSON from agent responses.

    The interceptor parses agent response text for A2UI content (separated by a
    delimiter), emits ``ActivitySnapshotEvent`` events with the A2UI operations,
    and strips the A2UI JSON from the response text to prevent duplication.

    The emitted ``ActivitySnapshotEvent`` is compatible with CopilotKit's
    ``@copilotkit/a2ui-renderer`` which renders A2UI surfaces out of the box.

    Args:
        delimiter: The delimiter separating text from A2UI JSON in agent responses.
        version_string: The A2UI protocol version string.
        activity_type: The activity type for emitted events. CopilotKit's A2UI
            renderer matches on this value.

    Returns:
        An async generator function with the signature
        ``(ServiceResponse) -> AsyncIterator[BaseEvent]`` suitable for passing
        to ``AGUIStream(event_interceptors=[...])``.

    Example::

        from autogen.ag_ui import AGUIStream
        from autogen.agents.experimental.a2ui import A2UIAgent, a2ui_event_interceptor

        agent = A2UIAgent(name="ui_agent", llm_config=llm_config)
        stream = AGUIStream(agent, event_interceptors=[a2ui_event_interceptor])
    """
    parser = A2UIResponseParser(version_string=version_string, delimiter=delimiter)

    async def a2ui_event_interceptor(response: ServiceResponse) -> AsyncIterator[BaseEvent]:
        if not response.message:
            return

        content = response.message.get("content")
        if not content or not isinstance(content, str):
            return

        parse_result = parser.parse(content)

        if not parse_result.has_a2ui:
            return

        if parse_result.parse_error:
            logger.warning("A2UI parse error in AG-UI interceptor: %s", parse_result.parse_error)
            return

        if parse_result.operations:
            yield ActivitySnapshotEvent(
                message_id=str(uuid4()),
                activity_type=activity_type,
                content={"operations": parse_result.operations},
            )

        # Strip A2UI JSON from the response text to prevent duplication
        if parse_result.text:
            response.message["content"] = parse_result.text
        else:
            response.message = None  # type: ignore[assignment]

    return a2ui_event_interceptor


#: Default AG-UI event interceptor for A2UI with standard configuration.
#: Use directly with ``AGUIStream(event_interceptors=[a2ui_event_interceptor])``.
a2ui_event_interceptor = create_a2ui_event_interceptor()
