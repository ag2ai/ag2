# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Session setup over the real SDK: instructions and model selection on ``session/new``.

Both ends are the genuine ACP connection classes (``duplex_acp_config``), so what
is asserted is what crossed the wire — serialization and routing included — and
the peer decides for itself which model its session starts on.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import acp
import pytest
from acp import schema

from ag2 import Agent, Context, MemoryStream, Plugin
from ag2.acp.testing import FakeACPConfig, duplex_acp_config
from ag2.events import BaseEvent, ModelResponse
from ag2.middleware import BaseMiddleware, LLMCall

OPTIONS = schema.SessionCapabilities(options=schema.SessionOptionsCapabilities())
SYSTEM_PROMPT = schema.SessionCapabilities(system_prompt=schema.SessionSystemPromptCapabilities())


@dataclass
class AppendPolicy:
    """An assembly policy that appends its own instruction."""

    name: str

    async def apply(
        self, prompts: list[str], events: list[BaseEvent], context: Context
    ) -> tuple[list[str], list[BaseEvent]]:
        return [*prompts, f"{self.name} instruction"], events


class SuffixMiddleware(BaseMiddleware):
    """Adds an instruction for the LLM calls it wraps, and takes it back after."""

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        original = context.prompt
        context.prompt = [*original, "Middleware instruction"]
        try:
            return await call_next(events, context)
        finally:
            context.prompt = original


def _picker(
    current: str, *values: str, option_id: str = "model", category: str | None = "model"
) -> schema.SessionConfigOptionSelect:
    return schema.SessionConfigOptionSelect(
        id=option_id,
        name="Model",
        category=category,
        type="select",
        current_value=current,
        options=[schema.SessionConfigSelectOption(value=v, name=v) for v in values],
    )


class SetupPeer:
    """An ACP agent that records the setup calls it receives, in order.

    ``preview`` answers ``session/options``; ``created`` is what ``session/new``
    reports, whatever initial values the client sent — so a test decides whether
    the session really starts on the requested model. ``session/set_config_option``
    applies the value unless ``applies`` is off, and answers with the options as
    they then stand. Each ``session/prompt`` logs the instructions the session is
    running with: the ``systemPrompt`` text it was created with.
    """

    def __init__(
        self,
        conn: acp.Client,
        log: list[tuple[str, Any]],
        *,
        capabilities: schema.SessionCapabilities,
        preview: list[schema.SessionConfigOptionSelect] | None = None,
        created: list[schema.SessionConfigOptionSelect] | None = None,
        applies: bool = True,
    ) -> None:
        self.conn = conn
        self.log = log
        self.capabilities = capabilities
        self.preview = preview
        self.created = created
        self.applies = applies
        self.instructions: list[str] | None = None

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
        if system_prompt is not None:
            self.instructions = [b.text for b in system_prompt.content if isinstance(b, schema.TextContentBlock)]
        return schema.NewSessionResponse(session_id="setup-1", config_options=self.created)

    async def set_config_option(
        self, *, session_id: str, config_id: str, value: Any, **kwargs: Any
    ) -> schema.SetSessionConfigOptionResponse:
        self.log.append(("session/set_config_option", (config_id, value)))
        if self.applies:
            self.created = [
                o.model_copy(update={"current_value": value}) if o.id == config_id else o for o in self.created or []
            ]
        return schema.SetSessionConfigOptionResponse(config_options=self.created or [])

    async def prompt(self, *, session_id: str, **kwargs: Any) -> schema.PromptResponse:
        self.log.append(("session/prompt", self.instructions))
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

    async def test_an_invocation_plugin_does_not_outlive_its_invocation_on_the_peer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(log, capabilities=SYSTEM_PROMPT)
        agent = Agent("acp", "Answer in French.", config=cfg)
        stream = MemoryStream()

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                await agent.ask("first", stream=stream, plugins=[Plugin(prompt="First turn only")])
            await agent.ask("second", stream=stream)
        finally:
            await cfg.aclose()

        assert [entry for entry in log if entry[0] == "session/prompt"] == [
            ("session/prompt", ["Answer in French."]),
            ("session/prompt", ["Answer in French."]),
        ]
        assert "Instructions supplied for this invocation only are not sent" in caplog.text

    async def test_invocation_middleware_does_not_outlive_its_invocation_on_the_peer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(log, capabilities=SYSTEM_PROMPT)
        agent = Agent("acp", "Answer in French.", config=cfg)
        stream = MemoryStream()

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                await agent.ask("first", stream=stream, middleware=[SuffixMiddleware])
            await agent.ask("second", stream=stream)
        finally:
            await cfg.aclose()

        assert [entry for entry in log if entry[0] == "session/prompt"] == [
            ("session/prompt", ["Answer in French."]),
            ("session/prompt", ["Answer in French."]),
        ]
        assert "Instructions supplied for this invocation only are not sent" in caplog.text

    async def test_an_invocation_plugin_policy_does_not_outlive_its_invocation_on_the_peer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(log, capabilities=SYSTEM_PROMPT)
        agent = Agent("acp", "Answer in French.", config=cfg)
        agent.add_policy(AppendPolicy("Agent"))
        stream = MemoryStream()

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                await agent.ask("first", stream=stream, plugins=[Plugin().add_policy(AppendPolicy("Plugin"))])
            await agent.ask("second", stream=stream)
        finally:
            await cfg.aclose()

        assert [entry for entry in log if entry[0] == "session/prompt"] == [
            ("session/prompt", ["Answer in French.", "Agent instruction"]),
            ("session/prompt", ["Answer in French.", "Agent instruction"]),
        ]
        assert "Instructions supplied for this invocation only are not sent" in caplog.text

    async def test_the_agents_own_middleware_reaches_the_peer(self, caplog: pytest.LogCaptureFixture) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(log, capabilities=SYSTEM_PROMPT)

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                await Agent("acp", "Answer in French.", config=cfg, middleware=[SuffixMiddleware]).ask("hello")
        finally:
            await cfg.aclose()

        assert [entry for entry in log if entry[0] == "session/prompt"] == [
            ("session/prompt", ["Answer in French.", "Middleware instruction"]),
        ]
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

    async def test_an_uncategorised_picker_with_the_model_id_is_still_used(self) -> None:
        log: list[tuple[str, Any]] = []
        cfg = _config(
            log,
            model="smart",
            capabilities=schema.SessionCapabilities(),
            created=[_picker("default", "default", "smart", category=None)],
        )

        try:
            reply = await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log[1:] == [
            ("session/set_config_option", ("model", "smart")),
            ("session/prompt", None),
        ]
        assert reply.response.model == "smart"

    async def test_a_picker_with_the_model_id_in_another_category_is_left_alone(self) -> None:
        log: list[tuple[str, Any]] = []
        mode = _picker("ask", "ask", "code", category="mode")
        cfg = _config(log, model="code", capabilities=OPTIONS, preview=[mode], created=[mode])

        try:
            await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert log == [
            ("session/options", None),
            ("session/new", {"system_prompt": None, "config_options": None}),
            ("session/prompt", None),
        ]

    async def test_setup_fails_when_set_config_option_leaves_the_session_on_another_model(self) -> None:
        log: list[tuple[str, Any]] = []
        picker = _picker("default", "default", "smart")
        cfg = _config(log, model="smart", capabilities=OPTIONS, preview=[picker], created=[picker], applies=False)

        try:
            with pytest.raises(ValueError, match="did not switch to model 'smart'.*left it on 'default'"):
                await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert _methods(log)[-1] == "session/set_config_option"
