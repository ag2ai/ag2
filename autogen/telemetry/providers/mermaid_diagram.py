# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
"""Mermaid Diagram Provider

This module provides a telemetry provider that tracks interactions for generating Mermaid diagrams.

Uses the following spans:
<N/A>

Uses the following events:
EventKind.AGENT_SEND_MSG

Builds up a list of interactions between agents and uses that to generate a Mermaid diagram.

Data gathered can build a mermaid sequence diagram text can be retrieved via the generate_sequence_diagram method.

Code example:
```python
from autogen.telemetry.telemetry_core import InstrumentationManager, telemetry_context
from autogen.telemetry.providers.mermaid_diagram import MermaidDiagramProvider

mermaid_provider = MermaidDiagramProvider()

with telemetry_context(instrumentation_manager, "My Program Name", {"my_program_id": "456"}) as telemetry:
    ... AG2 workflow code here ...

    # Print out the sequence diagram text, which can be pasted into: https://mermaid.live/
    print(mermaid_provider.generate_sequence_diagram())

```
"""


from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from ..base_telemetry import EventKind, SpanContext, SpanKind, TelemetryProvider


class Interaction(BaseModel):
    """Interactions for sequence style mermaid diagrams"""

    time: datetime
    from_agent: str
    to_agent: str
    message: str


class MermaidDiagramProvider(TelemetryProvider):
    def __init__(self):
        self.interactions: List[Interaction] = []
        self.participants: Dict = {}
        self.first_agent_name = None

        # Store active spans
        self._active_contexts = {}

        # Tool ids + name pairs, needed for reconciling tool responses and calls
        self.tool_id_names = {}

    def start_trace(self, name: str, core_span_id: str, attributes: Dict[str, Any] = None) -> SpanContext:
        pass

    def start_span(
        self,
        kind: SpanKind,
        core_span_id: str,
        parent_context: Optional[SpanContext] = None,
        attributes: Dict[str, Any] = None,
    ) -> SpanContext:
        pass

    def set_span_attribute(self, context: SpanContext, key: str, value: Any) -> None:
        pass

    def end_span(self, context: SpanContext) -> None:
        pass

    def record_event(self, span_context: SpanContext, event_kind: EventKind, attributes: Dict[str, Any] = None) -> None:
        """Record agent send events to build up interactions for the diagram

        See telemetry_core.py for details on this event.
        """
        if event_kind == EventKind.AGENT_SEND_MSG:
            # Don't include silent messages
            if not attributes.get("ag2.silent", True):
                sender = attributes.get("ag2.agent.sender")
                recipient = attributes.get("ag2.agent.recipient")

                if self.first_agent_name is None:
                    self.first_agent_name = sender

                if sender:
                    self.participants[sender] = sender
                if recipient:
                    self.participants[recipient] = recipient

                message = attributes.get("ag2.message", "")

                if message:
                    # Create and add interaction
                    # For mermaid diagrams we will show in time order not span creation order
                    # so we'll set the interaction time at end of span (now)
                    interaction = Interaction(
                        time=datetime.now(),
                        from_agent=sender,
                        to_agent=recipient,
                        message=self._get_display_text(message),
                    )

                    self.interactions.append(interaction)

    def generate_sequence_diagram(self) -> str:
        """Generate Mermaid sequence diagram markup from recorded interactions

        Mermaid documentation:
        https://mermaid.js.org/syntax/sequenceDiagram.html

        Args:
            None

        Returns:
            str: Mermaid sequence diagram markup
        """
        diagram = ["sequenceDiagram"]

        # Add starting agent first
        diagram.append(f"    participant {self.first_agent_name}")

        # If it's a group chat auto selection, put the the internal agents at the end
        is_group_chat_auto = "checking_agent" in self.participants and "speaker_selection_agent" in self.participants

        # Add participants except the first agent name and group chat internal agents
        for _, participant in self.participants.items():
            if participant != self.first_agent_name and (
                not is_group_chat_auto or participant not in ["checking_agent", "speaker_selection_agent"]
            ):
                diagram.append(f"    participant {participant}")

        if is_group_chat_auto:
            diagram.append("    participant checking_agent")
            diagram.append("    participant speaker_selection_agent")

        # Sort interactions by time
        sorted_interactions = sorted(self.interactions, key=lambda x: x.time)

        # Add interactions
        for interaction in sorted_interactions:
            # Clean and truncate message for diagram
            message = self._replace_invalid_chars(interaction.message, [":", ";", "{", "}", "\n"])
            diagram.append(f"    {interaction.from_agent}->>{interaction.to_agent}: {message}")

        return "\n".join(diagram)

    def _get_display_text(self, message: Union[dict, str]) -> str:
        """Processes a message string, formatting for display on the Mermaid transition with limited text.

        Args:
            message (Union[dict, str]): The message to process

        Returns:
            str: The processed/display-friendly message
        """
        if isinstance(message, str):
            return message[:50] + ("..." if len(message) > 50 else "")

        if isinstance(message, dict):

            # Handle "tool_calls" if present
            if "tool_calls" in message and isinstance(message["tool_calls"], list):
                tool_names = [
                    call["function"]["name"]
                    for call in message["tool_calls"]
                    if "function" in call and "name" in call["function"]
                ]
                self.tool_id_names.update({call["id"]: call["function"]["name"] for call in message["tool_calls"]})
                return f"Tool Call(s) -> {', '.join(tool_names)}"

            # Handle "tool_responses" if present
            if "tool_responses" in message and isinstance(message["tool_responses"], list):
                responses = []
                for response in message["tool_responses"]:
                    if "content" in response and "tool_call_id" in response:
                        tool_name = self.tool_id_names.get(response["tool_call_id"], "...")
                        content_preview = response["content"][:50] + ("..." if len(response["content"]) > 50 else "")
                        responses.append(f"{tool_name} -> {content_preview}")
                return f"Tool Response(s): {', '.join(responses)}"

            # Otherwise, extract the "content" field if present
            if "content" in message:
                return message["content"][:50] if message["content"] is not None else ""

        # If no relevant fields, return converted value to string
        return str(message)[:50] + ("..." if str(message) > 50 else "")

    def convert_attribute_value(self, value: Any) -> str:
        """Convert attribute values to strings for Mermaid compatibility

        Args:
            value (Any): The value to convert

        Returns:
            str: The converted value
        """
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return str(value)

    def _replace_invalid_chars(self, input: str, invalid_chars: List[str]) -> str:
        """Replace invalid characters in the input string as Mermaid may interpret them as special characters

        Args:
            input (str): The input string to process
            invalid_chars (List[str]): The list of invalid characters to replace

        Returns:
            str: The processed string with replaced invalid characters
        """
        for char in invalid_chars:
            input = input.replace(char, " ")
        return input
