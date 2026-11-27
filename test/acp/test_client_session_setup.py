# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Session setup over the real SDK: instructions and model selection on ``session/new``.

Both ends are the genuine ACP connection classes (``duplex_acp_config``), so what
is asserted is what crossed the wire — serialization and routing included — and
the peer decides for itself which model its session starts on.
"""

import logging
from typing import Any

import acp
import pytest
from acp import schema

from ag2 import Agent
from ag2.acp.testing import FakeACPConfig, duplex_acp_config

OPTIONS = schema.SessionCapabilities(options=schema.SessionOptionsCapabilities())
SYSTEM_PROMPT = schema.SessionCapabilities(system_prompt=schema.SessionSystemPromptCapabilities())


def _picker(current: str, *values: str, option_id: str = "model") -> schema.SessionConfigOptionSelect:
    return schema.SessionConfigOptionSelect(
        id=option_id,
        name="Model",
        category="model",
        type="select",
        current_value=current,
        options=[schema.SessionConfigSelectOption(value=v, name=v) for v in values],
    )


class SetupPeer:
    """An ACP agent that records the setup calls it receives, in order.

    ``preview`` answers ``session/options``; ``created`` is what ``session/new``
    reports, whatever initial values the client sent — so a test decides whether
    the session really starts on the requested model.
    """

    def __init__(
        self,
        conn: acp.Client,
        log: list[tuple[str, Any]],
        *,
        capabilities: schema.SessionCapabilities,
        preview: list[schema.SessionConfigOptionSelect] | None = None,
        created: list[schema.SessionConfigOptionSelect] | None = None,
    ) -> None:
        self.conn = conn
        self.log = log
        self.capabilities = capabilities
        self.preview = preview
        self.created = created

    async def initialize(self, **kwargs: Any) -> schema.InitializeResponse:
        return schema.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=schema.AgentCapabilities(session_capabilities=self.capabilities),
            agent_info=schema.Implementation(name="setup-peer", version="test"),
        )

    async def session_options(self, **kwargs: Any) -> schema.SessionOptionsResponse:
        self.log.append(("session/options", None))
        return schema.SessionOptionsResponse(config_options=self.preview)

    async def new_session(
        self,
        *,
        system_prompt: schema.SystemPrompt | None = None,
        config_options: list[Any] | None = None,
        **kwargs: Any,
    ) -> schema.NewSessionResponse:
        self.log.append(("session/new", {"system_prompt": system_prompt, "config_options": config_options}))
        return schema.NewSessionResponse(session_id="setup-1", config_options=self.created)

    async def set_config_option(
        self, *, session_id: str, config_id: str, value: Any, **kwargs: Any
    ) -> schema.SetSessionConfigOptionResponse:
        self.log.append(("session/set_config_option", (config_id, value)))
        return schema.SetSessionConfigOptionResponse(config_options=self.created or [])

    async def prompt(self, *, session_id: str, **kwargs: Any) -> schema.PromptResponse:
        self.log.append(("session/prompt", None))
        await self.conn.session_update(
            session_id=session_id,
            update=schema.AgentMessageChunk(
                session_update="agent_message_chunk", content=schema.TextContentBlock(type="text", text="ok")
            ),
        )
        return schema.PromptResponse(stop_reason="end_turn")


def _config(log: list[tuple[str, Any]], *, model: str | None = None, **peer: Any) -> FakeACPConfig:
    overrides = {"model": model} if model is not None else {}
    return duplex_acp_config(
        lambda conn: SetupPeer(conn, log, **peer), use_unstable_protocol=True, permission_policy="auto", **overrides
    )


def _methods(log: list[tuple[str, Any]]) -> list[str]:
    return [method for method, _ in log]


@pytest.mark.asyncio
class TestInstructions:
    async def test_static_and_hook_instructions_go_with_session_new_in_append_mode(self) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(log, capabilities=SYSTEM_PROMPT)
        agent = Agent("acp", "Answer in French.", config=cfg)

        @agent.prompt
        def be_brief() -> str:
            return "Be brief."

        try:
            reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == "ok"
        assert log[0] == (
            "session/new",
            {
                "system_prompt": schema.SystemPrompt(
                    mode="append",
                    content=[
                        schema.TextContentBlock(type="text", text="Answer in French."),
                        schema.TextContentBlock(type="text", text="Be brief."),
                    ],
                ),
                "config_options": None,
            },
        )

    async def test_are_withheld_with_a_warning_from_an_agent_that_does_not_accept_them(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(log, capabilities=schema.SessionCapabilities())

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.session"):
                await Agent("acp", "Answer in French.", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log[0] == ("session/new", {"system_prompt": None, "config_options": None})
        assert "does not accept a system prompt" in caplog.text

    async def test_nothing_is_sent_and_nothing_logged_without_any(self, caplog: pytest.LogCaptureFixture) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(log, capabilities=SYSTEM_PROMPT)

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.session"):
                await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log[0] == ("session/new", {"system_prompt": None, "config_options": None})
        assert caplog.text == ""


@pytest.mark.asyncio
class TestModelSelection:
    async def test_a_session_that_starts_on_the_model_needs_no_set_config_option(self) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(
            log,
            model="smart",
            capabilities=OPTIONS,
            preview=[_picker("default", "default", "smart")],
            created=[_picker("smart", "default", "smart")],
        )

        try:
            reply = await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log == [
            ("session/options", None),
            (
                "session/new",
                {
                    "system_prompt": None,
                    "config_options": [schema.InitialSessionConfigOption2(config_id="model", value="smart")],
                },
            ),
            ("session/prompt", None),
        ]
        assert reply.response.model == "smart"

    async def test_nothing_is_sent_for_the_model_a_new_session_already_has(self) -> None:
        log: list[tuple[str, Any]] = []
        picker = _picker("default", "default", "smart")
        cfg = _config(log, model="default", capabilities=OPTIONS, preview=[picker], created=[picker])

        try:
            await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log == [
            ("session/options", None),
            ("session/new", {"system_prompt": None, "config_options": None}),
            ("session/prompt", None),
        ]

    async def test_set_config_option_runs_before_the_first_prompt_when_session_new_ignored_the_model(self) -> None:
        log: list[tuple[str, Any]] = []
        picker = _picker("default", "default", "smart")
        cfg = _config(log, model="smart", capabilities=OPTIONS, preview=[picker], created=[picker])

        try:
            await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log == [
            ("session/options", None),
            (
                "session/new",
                {
                    "system_prompt": None,
                    "config_options": [schema.InitialSessionConfigOption2(config_id="model", value="smart")],
                },
            ),
            ("session/set_config_option", ("model", "smart")),
            ("session/prompt", None),
        ]

    async def test_a_model_missing_from_the_preview_is_selected_from_the_created_session(self) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(
            log,
            model="smart",
            capabilities=OPTIONS,
            preview=[_picker("default", "default")],
            created=[_picker("default", "default", "smart")],
        )

        try:
            await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log == [
            ("session/options", None),
            ("session/new", {"system_prompt": None, "config_options": None}),
            ("session/set_config_option", ("model", "smart")),
            ("session/prompt", None),
        ]

    async def test_a_model_missing_from_both_pickers_fails_before_the_first_prompt(self) -> None:
        log: list[tuple[str, Any]] = []
        picker = _picker("default", "default")
        cfg = _config(log, model="smart", capabilities=OPTIONS, preview=[picker], created=[picker])

        try:
            with pytest.raises(ValueError, match="not offered by the ACP agent"):
                await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert "session/prompt" not in _methods(log)
        assert "session/set_config_option" not in _methods(log)

    async def test_a_picker_is_found_by_its_model_category_whatever_its_id(self) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(
            log,
            model="smart",
            capabilities=schema.SessionCapabilities(),
            created=[_picker("default", "default", "smart", option_id="engine")],
        )

        try:
            await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log[1:] == [
            ("session/set_config_option", ("engine", "smart")),
            ("session/prompt", None),
        ]
