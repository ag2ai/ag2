# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.agents.experimental.a2ui.a2a_executor import A2UIAgentExecutor


@pytest.fixture()
def executor() -> A2UIAgentExecutor:
    """Create an executor with a mock agent."""
    mock_agent = MagicMock()
    mock_agent.function_map = {}
    return A2UIAgentExecutor(agent=mock_agent)


class TestBuildFinalParts:
    """Tests for _build_final_parts — the pure parsing/part-building logic."""

    def test_a2ui_response_returns_text_and_data_parts(self, executor: A2UIAgentExecutor) -> None:
        response = 'Here is your UI.\n---a2ui_JSON---\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]'
        parts = executor._build_final_parts(response, use_a2ui=True)
        assert len(parts) == 2
        # First part is text
        assert parts[0].root.text == "Here is your UI."  # type: ignore[union-attr]
        # Second part is DataPart with a2ui content
        assert parts[1].root.data is not None  # type: ignore[union-attr]

    def test_a2ui_response_no_text_prefix(self, executor: A2UIAgentExecutor) -> None:
        response = '---a2ui_JSON---\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]'
        parts = executor._build_final_parts(response, use_a2ui=True)
        # Only DataPart, no TextPart since text is empty
        assert len(parts) == 1
        assert parts[0].root.data is not None  # type: ignore[union-attr]

    def test_plain_text_when_a2ui_enabled(self, executor: A2UIAgentExecutor) -> None:
        parts = executor._build_final_parts("Just plain text.", use_a2ui=True)
        assert len(parts) == 1
        assert parts[0].root.text == "Just plain text."  # type: ignore[union-attr]

    def test_plain_text_when_a2ui_disabled(self, executor: A2UIAgentExecutor) -> None:
        response = 'Text\n---a2ui_JSON---\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]'
        parts = executor._build_final_parts(response, use_a2ui=False)
        # A2UI not active — returns full text as-is
        assert len(parts) == 1
        assert "---a2ui_JSON---" in parts[0].root.text  # type: ignore[union-attr]

    def test_empty_response(self, executor: A2UIAgentExecutor) -> None:
        parts = executor._build_final_parts("", use_a2ui=True)
        assert parts == []

    def test_invalid_json_falls_back_to_text(self, executor: A2UIAgentExecutor) -> None:
        response = "Text\n---a2ui_JSON---\n{not valid json}"
        parts = executor._build_final_parts(response, use_a2ui=True)
        # Parse error — falls back to full text
        assert len(parts) == 1
        assert "---a2ui_JSON---" in parts[0].root.text  # type: ignore[union-attr]

    def test_markdown_fences_stripped(self, executor: A2UIAgentExecutor) -> None:
        response = (
            'UI below.\n---a2ui_JSON---\n```json\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]\n```'
        )
        parts = executor._build_final_parts(response, use_a2ui=True)
        assert len(parts) == 2
        assert parts[1].root.data is not None  # type: ignore[union-attr]
