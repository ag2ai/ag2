# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.a2a.constants import A2UI_MIME_TYPE
from autogen.agents.experimental.a2ui import A2UIAgent
from autogen.agents.experimental.a2ui.a2a_executor import A2UIAgentExecutor
from autogen.agents.experimental.a2ui.a2a_helpers import A2UI_EXTENSION_URI


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


class TestExtensionNegotiation:
    """Tests for A2UI extension version negotiation in the executor.

    Verifies that the executor correctly splits or preserves responses
    based on whether the client requested the A2UI extension.
    """

    A2UI_RESPONSE = 'Here is your UI.\n---a2ui_JSON---\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]'

    def test_client_requests_v09_gets_datapart(self, executor: A2UIAgentExecutor) -> None:
        """Client requests A2UI v0.9 — response should be split into TextPart + DataPart."""
        parts = executor._build_final_parts(self.A2UI_RESPONSE, use_a2ui=True)
        assert len(parts) == 2
        # TextPart
        assert parts[0].root.text == "Here is your UI."  # type: ignore[union-attr]
        # DataPart with A2UI MIME type — data is the operation directly
        assert parts[1].root.metadata["mimeType"] == A2UI_MIME_TYPE  # type: ignore[union-attr,index]
        assert parts[1].root.data["version"] == "v0.9"  # type: ignore[union-attr,index]
        assert "deleteSurface" in parts[1].root.data  # type: ignore[union-attr]

    def test_client_does_not_request_a2ui_gets_text_only(self, executor: A2UIAgentExecutor) -> None:
        """Client doesn't request A2UI — response returned as raw text, no splitting."""
        parts = executor._build_final_parts(self.A2UI_RESPONSE, use_a2ui=False)
        assert len(parts) == 1
        # Full response including delimiter and JSON returned as text
        assert "---a2ui_JSON---" in parts[0].root.text  # type: ignore[union-attr]
        assert "deleteSurface" in parts[0].root.text  # type: ignore[union-attr]

    def test_client_requests_wrong_version_gets_text_only(self, executor: A2UIAgentExecutor) -> None:
        """Client requests a different A2UI version — should behave like no A2UI.

        The version mismatch is handled at the try_activate_a2ui_extension level,
        which would set use_a2ui=False. This test verifies the executor correctly
        returns unsplit text when use_a2ui is False.
        """
        parts = executor._build_final_parts(self.A2UI_RESPONSE, use_a2ui=False)
        assert len(parts) == 1
        # No DataPart — full text including A2UI JSON
        raw_text = parts[0].root.text  # type: ignore[union-attr]
        assert "---a2ui_JSON---" in raw_text

    def test_version_negotiation_e2e_supported(self) -> None:
        """End-to-end: client requests v0.9, server supports v0.9 → activated."""
        from autogen.agents.experimental.a2ui.a2a_helpers import try_activate_a2ui_extension

        class MockContext:
            def __init__(self) -> None:
                self.requested_extensions = {A2UI_EXTENSION_URI}
                self._activated: set[str] = set()

            def add_activated_extension(self, uri: str) -> None:
                self._activated.add(uri)

        ctx = MockContext()
        use_a2ui = try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert use_a2ui is True

        # Now verify executor splits the response
        mock_agent = MagicMock()
        mock_agent.function_map = {}
        exec = A2UIAgentExecutor(agent=mock_agent)
        parts = exec._build_final_parts(self.A2UI_RESPONSE, use_a2ui=use_a2ui)
        assert len(parts) == 2
        assert parts[1].root.metadata["mimeType"] == A2UI_MIME_TYPE  # type: ignore[union-attr,index]

    def test_version_negotiation_e2e_unsupported(self) -> None:
        """End-to-end: client requests v1.0, server supports v0.9 → not activated."""
        from autogen.agents.experimental.a2ui.a2a_helpers import try_activate_a2ui_extension

        class MockContext:
            def __init__(self) -> None:
                self.requested_extensions = {"https://a2ui.org/a2a-extension/a2ui/v1.0"}
                self._activated: set[str] = set()

            def add_activated_extension(self, uri: str) -> None:
                self._activated.add(uri)

        ctx = MockContext()
        use_a2ui = try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert use_a2ui is False

        # Executor should return unsplit text
        mock_agent = MagicMock()
        mock_agent.function_map = {}
        exec = A2UIAgentExecutor(agent=mock_agent)
        parts = exec._build_final_parts(self.A2UI_RESPONSE, use_a2ui=use_a2ui)
        assert len(parts) == 1
        assert "---a2ui_JSON---" in parts[0].root.text  # type: ignore[union-attr]

    def test_version_negotiation_e2e_no_extensions(self) -> None:
        """End-to-end: client sends no extensions → not activated, text-only."""
        from autogen.agents.experimental.a2ui.a2a_helpers import try_activate_a2ui_extension

        class MockContext:
            def __init__(self) -> None:
                self.requested_extensions: set[str] = set()
                self._activated: set[str] = set()

            def add_activated_extension(self, uri: str) -> None:
                self._activated.add(uri)

        ctx = MockContext()
        use_a2ui = try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert use_a2ui is False

        mock_agent = MagicMock()
        mock_agent.function_map = {}
        executor = A2UIAgentExecutor(agent=mock_agent)
        parts = executor._build_final_parts(self.A2UI_RESPONSE, use_a2ui=use_a2ui)
        assert len(parts) == 1


class TestAgentCardAutoDetection:
    """Tests for A2aAgentServer auto-detecting A2UIAgent and configuring the agent card."""

    def test_a2ui_agent_adds_extension_to_card(self) -> None:
        """Wrapping an A2UIAgent in A2aAgentServer should auto-add the A2UI extension to the card."""
        from autogen.a2a import A2aAgentServer, CardSettings

        agent = A2UIAgent(name="test_agent", llm_config=False)
        server = A2aAgentServer(
            agent=agent,
            url="http://localhost:9000",
            agent_card=CardSettings(name="Test", description="Test agent"),
        )

        extensions = server.card.capabilities.extensions or []
        a2ui_exts = [e for e in extensions if e.uri == A2UI_EXTENSION_URI]
        assert len(a2ui_exts) == 1
        assert "A2UI" in (a2ui_exts[0].description or "")

    def test_a2ui_extension_includes_supported_catalog_ids(self) -> None:
        """The A2UI extension in the card should list supportedCatalogIds."""
        from autogen.a2a import A2aAgentServer, CardSettings

        agent = A2UIAgent(name="test_agent", llm_config=False)
        server = A2aAgentServer(
            agent=agent,
            url="http://localhost:9000",
            agent_card=CardSettings(name="Test", description="Test"),
        )

        extensions = server.card.capabilities.extensions or []
        a2ui_ext = next(e for e in extensions if e.uri == A2UI_EXTENSION_URI)
        assert a2ui_ext.params is not None
        assert "supportedCatalogIds" in a2ui_ext.params
        catalog_ids = a2ui_ext.params["supportedCatalogIds"]
        assert isinstance(catalog_ids, list)
        assert len(catalog_ids) >= 1
        assert agent.catalog_id in catalog_ids

    def test_custom_catalog_id_in_card(self) -> None:
        """Custom catalog ID should appear in supportedCatalogIds."""
        from autogen.a2a import A2aAgentServer, CardSettings

        agent = A2UIAgent(
            name="test_agent",
            llm_config=False,
            custom_catalog={"$id": "https://mycompany.com/custom.json", "components": {}},
        )
        server = A2aAgentServer(
            agent=agent,
            url="http://localhost:9000",
            agent_card=CardSettings(name="Test", description="Test"),
        )

        extensions = server.card.capabilities.extensions or []
        a2ui_ext = next(e for e in extensions if e.uri == A2UI_EXTENSION_URI)
        catalog_ids = a2ui_ext.params["supportedCatalogIds"]  # type: ignore[index]
        assert "https://mycompany.com/custom.json" in catalog_ids

    def test_non_a2ui_agent_no_extension(self) -> None:
        """A regular ConversableAgent should NOT get the A2UI extension."""
        from autogen import ConversableAgent
        from autogen.a2a import A2aAgentServer, CardSettings

        agent = ConversableAgent(name="regular_agent", llm_config=False)
        server = A2aAgentServer(
            agent=agent,
            url="http://localhost:9000",
            agent_card=CardSettings(name="Test", description="Test"),
        )

        extensions = server.card.capabilities.extensions or []
        a2ui_exts = [e for e in extensions if "a2ui" in e.uri.lower()]
        assert len(a2ui_exts) == 0

    def test_executor_type_for_a2ui_agent(self) -> None:
        """A2aAgentServer.executor should return A2UIAgentExecutor for A2UIAgent."""
        from autogen.a2a import A2aAgentServer, CardSettings

        agent = A2UIAgent(name="test_agent", llm_config=False)
        server = A2aAgentServer(
            agent=agent,
            url="http://localhost:9000",
            agent_card=CardSettings(name="Test", description="Test"),
        )

        assert isinstance(server.executor, A2UIAgentExecutor)

    def test_executor_type_for_regular_agent(self) -> None:
        """A2aAgentServer.executor should return standard executor for non-A2UIAgent."""
        from autogen import ConversableAgent
        from autogen.a2a import A2aAgentServer, CardSettings

        agent = ConversableAgent(name="regular_agent", llm_config=False)
        server = A2aAgentServer(
            agent=agent,
            url="http://localhost:9000",
            agent_card=CardSettings(name="Test", description="Test"),
        )

        assert not isinstance(server.executor, A2UIAgentExecutor)

    def test_no_duplicate_extension_if_manually_added(self) -> None:
        """If A2UI extension is already in the card, don't add it again."""
        from a2a.types import AgentExtension

        from autogen.a2a import A2aAgentServer, CardSettings

        agent = A2UIAgent(name="test_agent", llm_config=False)
        card_settings = CardSettings(
            name="Test",
            description="Test",
            extensions=[
                AgentExtension(
                    uri=A2UI_EXTENSION_URI,
                    description="Manually added",
                ),
            ],
        )
        server = A2aAgentServer(
            agent=agent,
            url="http://localhost:9000",
            agent_card=card_settings,
        )

        extensions = server.card.capabilities.extensions or []
        a2ui_exts = [e for e in extensions if e.uri == A2UI_EXTENSION_URI]
        assert len(a2ui_exts) == 1
