# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase 5a.1 transforms package.

Covers the low-level machinery in isolation — registry, when matcher,
and each of the three adapter shapes. End-to-end wiring through
:class:`ActorClient` + :class:`Hub` is covered in
``test_transforms_integration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from autogen.beta.network.client.transforms import (
    HttpTransform,
    NamedTransform,
    PythonTransform,
    Transform,
    TransformContext,
    TransformLookupError,
    TransformPipeline,
    TransformRegistry,
    when_matches,
)
from autogen.beta.network.client.transforms.protocol import TransformRejected
from autogen.beta.network.envelope import EV_TEXT, Envelope
from autogen.beta.network.rule import (
    Rule,
    TransformSpec,
    TransformStage,
)


# ---------------------------------------------------------------------------
# Test doubles — minimal stand-ins for the ActorClient the pipeline uses
# ---------------------------------------------------------------------------


class _StubClient:
    """Just enough of :class:`ActorClient` for the unit tests.

    The real ``TransformContext`` captures a live ``ActorClient``;
    pipeline and matcher tests only need ``_session_type_for`` plus
    a readable ``actor_id`` / ``identity`` handle for audit stamping.
    """

    def __init__(
        self,
        actor_id: str = "stub-actor",
        session_types: dict[str, str] | None = None,
    ) -> None:
        self.actor_id = actor_id
        self.identity = _StubIdentity(name=actor_id)
        self._session_types: dict[str, str] = session_types or {}

    def _session_type_for(self, session_id: str | None) -> str | None:
        if session_id is None:
            return None
        return self._session_types.get(session_id)


@dataclass
class _StubIdentity:
    name: str


def _make_envelope(
    *,
    content: str = "hello",
    session_id: str = "sess-1",
    sender_id: str = "alice",
    event_type: str = EV_TEXT,
) -> Envelope:
    if event_type == EV_TEXT:
        return Envelope.text(
            session_id=session_id,
            sender_id=sender_id,
            content=content,
        )
    return Envelope(
        session_id=session_id,
        sender_id=sender_id,
        event_type=event_type,
        event_data={"content": content},
    )


def _make_ctx(
    *,
    stage: TransformStage = TransformStage.PRE_SEND,
    client: _StubClient | None = None,
    session_id: str | None = "sess-1",
) -> TransformContext:
    direction = (
        "outbound"
        if stage in {TransformStage.PRE_SEND, TransformStage.POST_SEND}
        else "inbound"
    )
    return TransformContext(
        stage=stage,
        client=client or _StubClient(),  # type: ignore[arg-type]
        session_id=session_id,
        rule_version=1,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# TransformRegistry
# ---------------------------------------------------------------------------


class TestTransformRegistry:
    def test_register_and_create(self) -> None:
        reg = TransformRegistry()

        async def _pass(envelope: Envelope, ctx: TransformContext) -> Envelope:
            return envelope

        reg.register("passthrough", lambda: _pass)
        assert reg.has("passthrough")
        assert reg.names() == ["passthrough"]
        instance = reg.create("passthrough")
        assert callable(instance)

    def test_register_replaces_existing(self) -> None:
        reg = TransformRegistry()
        reg.register("name", lambda: (lambda e, c: e))
        first = reg.create("name")
        reg.register("name", lambda: (lambda e, c: None))
        second = reg.create("name")
        assert first is not second

    def test_unregister_removes(self) -> None:
        reg = TransformRegistry()
        reg.register("tmp", lambda: (lambda e, c: e))
        reg.unregister("tmp")
        assert not reg.has("tmp")

    def test_unregister_missing_is_no_op(self) -> None:
        reg = TransformRegistry()
        reg.unregister("never-added")  # no raise

    def test_unknown_name_raises_lookup_error(self) -> None:
        reg = TransformRegistry()
        reg.register("one", lambda: (lambda e, c: e))
        with pytest.raises(TransformLookupError) as exc_info:
            reg.create("two")
        assert "two" in str(exc_info.value)
        assert "one" in str(exc_info.value)  # lists known names

    def test_create_returns_fresh_instance(self) -> None:
        """Each ``create`` call instantiates a new :class:`Transform`.

        The pipeline caches the instance for the rule's lifetime, but
        the factory itself produces a new one on every call so tests
        can assert that a rule rebuild yields a fresh object.
        """

        reg = TransformRegistry()

        class _Counter:
            n = 0

            async def __call__(
                self, envelope: Envelope, ctx: TransformContext
            ) -> Envelope:
                self.n += 1
                return envelope

        reg.register("ctr", lambda: _Counter())
        a = reg.create("ctr")
        b = reg.create("ctr")
        assert a is not b


# ---------------------------------------------------------------------------
# when_matches
# ---------------------------------------------------------------------------


class TestWhenMatcher:
    def test_empty_when_always_matches(self) -> None:
        client = _StubClient(session_types={"sess-1": "consulting"})
        env = _make_envelope()
        assert when_matches({}, env, client) is True

    def test_event_key_matches(self) -> None:
        client = _StubClient()
        env = _make_envelope()
        assert when_matches({"event": "ag2.msg.text"}, env, client) is True
        assert when_matches({"event": "ag2.msg.other"}, env, client) is False

    def test_session_type_matches(self) -> None:
        client = _StubClient(session_types={"sess-1": "consulting"})
        env = _make_envelope(session_id="sess-1")
        assert when_matches({"session_type": "consulting"}, env, client) is True
        assert when_matches({"session_type": "conversation"}, env, client) is False

    def test_session_type_unknown_session_id_rejects(self) -> None:
        client = _StubClient(session_types={"sess-1": "consulting"})
        env = _make_envelope(session_id="sess-unknown")
        assert when_matches({"session_type": "consulting"}, env, client) is False

    def test_and_semantics(self) -> None:
        client = _StubClient(session_types={"sess-1": "consulting"})
        env = _make_envelope(session_id="sess-1")
        # Both keys satisfied → True.
        assert when_matches(
            {"event": "ag2.msg.text", "session_type": "consulting"}, env, client
        ) is True
        # Event OK but session type mismatch → False.
        assert when_matches(
            {"event": "ag2.msg.text", "session_type": "broadcast"}, env, client
        ) is False

    def test_unknown_keys_ignored_for_forward_compat(self) -> None:
        client = _StubClient()
        env = _make_envelope()
        # The matcher must not raise on unknown keys; 5b may add more
        # without breaking 5a.1-era rules.
        assert (
            when_matches({"future_key": "future_value"}, env, client) is True
        )


# ---------------------------------------------------------------------------
# NamedTransform
# ---------------------------------------------------------------------------


class TestNamedTransform:
    @pytest.mark.asyncio
    async def test_delegates_to_registry_entry(self) -> None:
        reg = TransformRegistry()

        async def _mutate(envelope: Envelope, ctx: TransformContext) -> Envelope:
            envelope.event_data = {**envelope.event_data, "content": "MUTATED"}
            return envelope

        reg.register("mutate", lambda: _mutate)
        transform = NamedTransform("mutate", reg)
        env = _make_envelope(content="original")
        result = await transform(env, _make_ctx())
        assert result is not None
        assert result.event_data["content"] == "MUTATED"

    def test_unknown_name_raises_at_construction(self) -> None:
        reg = TransformRegistry()
        with pytest.raises(TransformLookupError):
            NamedTransform("missing", reg)

    @pytest.mark.asyncio
    async def test_rejection_propagates(self) -> None:
        reg = TransformRegistry()
        reg.register(
            "reject",
            lambda: (lambda e, c: _async_none()),
        )
        transform = NamedTransform("reject", reg)
        result = await transform(_make_envelope(), _make_ctx())
        assert result is None


async def _async_none() -> None:
    return None


# ---------------------------------------------------------------------------
# PythonTransform
# ---------------------------------------------------------------------------


# Module-level classes so importlib can resolve them via dotted path.


class _PyPass:
    def __init__(self) -> None:
        self.seen = 0

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope:
        self.seen += 1
        return envelope


class _PyReject:
    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> None:
        return None


class _PyConfigurable:
    def __init__(self, suffix: str = "!") -> None:
        self.suffix = suffix

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope:
        content = envelope.event_data.get("content", "")
        envelope.event_data = {
            **envelope.event_data,
            "content": content + self.suffix,
        }
        return envelope


class _PyConfigDict:
    """Class that wants the whole config dict as a single arg."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.prefix = config.get("prefix", ">")

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope:
        envelope.event_data = {
            **envelope.event_data,
            "content": self.prefix + envelope.event_data.get("content", ""),
        }
        return envelope


_HERE_MODULE = "test.beta.network.test_transforms_unit"


class TestPythonTransform:
    @pytest.mark.asyncio
    async def test_happy_path_mutates(self) -> None:
        adapter = PythonTransform(
            {"module": _HERE_MODULE, "class": "_PyConfigurable",
             "config": {"suffix": " WORLD"}},
        )
        result = await adapter(_make_envelope(content="hello"), _make_ctx())
        assert result is not None
        assert result.event_data["content"] == "hello WORLD"

    @pytest.mark.asyncio
    async def test_happy_path_zero_arg_constructor(self) -> None:
        adapter = PythonTransform(
            {"module": _HERE_MODULE, "class": "_PyPass"}
        )
        result = await adapter(_make_envelope(), _make_ctx())
        assert result is not None

    @pytest.mark.asyncio
    async def test_config_dict_fallback_constructor(self) -> None:
        adapter = PythonTransform(
            {
                "module": _HERE_MODULE,
                "class": "_PyConfigDict",
                "config": {"prefix": ">>> "},
            },
        )
        result = await adapter(_make_envelope(content="x"), _make_ctx())
        assert result is not None
        assert result.event_data["content"] == ">>> x"

    @pytest.mark.asyncio
    async def test_rejection_propagates(self) -> None:
        adapter = PythonTransform(
            {"module": _HERE_MODULE, "class": "_PyReject"}
        )
        result = await adapter(_make_envelope(), _make_ctx())
        assert result is None

    def test_unknown_module_raises(self) -> None:
        from autogen.beta.network.client.transforms.protocol import (
            TransformError,
        )

        with pytest.raises(TransformError, match="not importable"):
            PythonTransform(
                {"module": "autogen._not_a_real_module_xyz", "class": "X"}
            )

    def test_unknown_class_raises(self) -> None:
        from autogen.beta.network.client.transforms.protocol import (
            TransformError,
        )

        with pytest.raises(TransformError, match="not found"):
            PythonTransform(
                {"module": _HERE_MODULE, "class": "_NotAClassInThisModule"}
            )

    def test_missing_module_field_raises(self) -> None:
        from autogen.beta.network.client.transforms.protocol import (
            TransformError,
        )

        with pytest.raises(TransformError, match="'module' string"):
            PythonTransform({"class": "_PyPass"})

    def test_missing_class_field_raises(self) -> None:
        from autogen.beta.network.client.transforms.protocol import (
            TransformError,
        )

        with pytest.raises(TransformError, match="'class' string"):
            PythonTransform({"module": _HERE_MODULE})

    def test_config_not_dict_raises(self) -> None:
        from autogen.beta.network.client.transforms.protocol import (
            TransformError,
        )

        with pytest.raises(TransformError, match="dict"):
            PythonTransform(
                {"module": _HERE_MODULE, "class": "_PyPass", "config": "no"}
            )

    @pytest.mark.asyncio
    async def test_aclose_on_missing_hook_is_noop(self) -> None:
        adapter = PythonTransform(
            {"module": _HERE_MODULE, "class": "_PyPass"}
        )
        await adapter.aclose()  # _PyPass has no aclose hook


# ---------------------------------------------------------------------------
# HttpTransform
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        status: int,
        *,
        json_body: dict | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status
        self._json = json_body
        self.text = text

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class _FakeHttpxClient:
    """Captures POST calls and returns pre-scripted responses."""

    def __init__(self, *, script: list[_FakeResponse] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.script = list(script or [])
        self.closed = False

    async def post(self, url: str, *, json: dict) -> _FakeResponse:
        self.calls.append((url, json))
        if not self.script:
            raise RuntimeError("no response scripted")
        return self.script.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class TestHttpTransform:
    def test_rejects_empty_url(self) -> None:
        from autogen.beta.network.client.transforms.protocol import (
            TransformError,
        )

        with pytest.raises(TransformError):
            HttpTransform("")

    @pytest.mark.asyncio
    async def test_200_returns_decoded_envelope(self) -> None:
        transform = HttpTransform("http://localhost:9000/sidecar")
        # Force our fake client.
        mutated = _make_envelope(content="mutated").to_dict()
        transform._client = _FakeHttpxClient(  # type: ignore[attr-defined]
            script=[_FakeResponse(200, json_body=mutated)],
        )
        result = await transform(
            _make_envelope(content="before"), _make_ctx()
        )
        assert result is not None
        assert result.event_data["content"] == "mutated"

    @pytest.mark.asyncio
    async def test_204_passes_through(self) -> None:
        transform = HttpTransform("http://localhost:9000/sidecar")
        transform._client = _FakeHttpxClient(  # type: ignore[attr-defined]
            script=[_FakeResponse(204)],
        )
        env = _make_envelope(content="unchanged")
        result = await transform(env, _make_ctx())
        assert result is env

    @pytest.mark.asyncio
    async def test_409_rejects(self) -> None:
        transform = HttpTransform("http://localhost:9000/sidecar")
        transform._client = _FakeHttpxClient(  # type: ignore[attr-defined]
            script=[_FakeResponse(409, text="policy violation")],
        )
        result = await transform(_make_envelope(), _make_ctx())
        assert result is None

    @pytest.mark.asyncio
    async def test_500_fails_as_reject(self) -> None:
        transform = HttpTransform("http://localhost:9000/sidecar")
        transform._client = _FakeHttpxClient(  # type: ignore[attr-defined]
            script=[_FakeResponse(500, text="server error")],
        )
        result = await transform(_make_envelope(), _make_ctx())
        assert result is None

    @pytest.mark.asyncio
    async def test_connection_failure_fails_as_reject(self) -> None:
        transform = HttpTransform("http://localhost:9000/sidecar")

        class _BrokenClient:
            async def post(self, url: str, *, json: dict) -> Any:
                raise ConnectionRefusedError("no sidecar")

            async def aclose(self) -> None:
                return None

        transform._client = _BrokenClient()  # type: ignore[attr-defined]
        result = await transform(_make_envelope(), _make_ctx())
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_200_body_fails_as_reject(self) -> None:
        transform = HttpTransform("http://localhost:9000/sidecar")
        # JSON body that is not a valid Envelope dict.
        transform._client = _FakeHttpxClient(  # type: ignore[attr-defined]
            script=[_FakeResponse(200, json_body={"not": "an envelope"})],
        )
        result = await transform(_make_envelope(), _make_ctx())
        assert result is None

    @pytest.mark.asyncio
    async def test_pooled_client_reuse_across_envelopes(self) -> None:
        transform = HttpTransform("http://localhost:9000/sidecar")
        fake = _FakeHttpxClient(
            script=[
                _FakeResponse(204),
                _FakeResponse(204),
                _FakeResponse(204),
            ],
        )
        transform._client = fake  # type: ignore[attr-defined]
        for _ in range(3):
            await transform(_make_envelope(), _make_ctx())
        # One client, three calls.
        assert len(fake.calls) == 3
        assert not fake.closed

    @pytest.mark.asyncio
    async def test_aclose_shuts_pooled_client(self) -> None:
        transform = HttpTransform("http://localhost:9000/sidecar")
        fake = _FakeHttpxClient(script=[_FakeResponse(204)])
        transform._client = fake  # type: ignore[attr-defined]
        await transform(_make_envelope(), _make_ctx())
        await transform.aclose()
        assert fake.closed
        # Second aclose is a no-op.
        await transform.aclose()


# ---------------------------------------------------------------------------
# TransformPipeline — build and dispatch
# ---------------------------------------------------------------------------


def _make_registry() -> TransformRegistry:
    reg = TransformRegistry()

    async def _upper(envelope: Envelope, ctx: TransformContext) -> Envelope:
        content = envelope.event_data.get("content", "")
        envelope.event_data = {
            **envelope.event_data,
            "content": content.upper(),
        }
        return envelope

    async def _exclaim(envelope: Envelope, ctx: TransformContext) -> Envelope:
        content = envelope.event_data.get("content", "")
        envelope.event_data = {**envelope.event_data, "content": content + "!"}
        return envelope

    async def _reject(envelope: Envelope, ctx: TransformContext) -> None:
        return None

    reg.register("upper", lambda: _upper)
    reg.register("exclaim", lambda: _exclaim)
    reg.register("reject", lambda: _reject)
    return reg


class TestTransformPipeline:
    @pytest.mark.asyncio
    async def test_empty_pipeline_passes_through(self) -> None:
        pipeline = TransformPipeline.build(
            Rule(), registry=_make_registry()
        )
        env = _make_envelope()
        result = await pipeline.run_pre_send(env, _StubClient())  # type: ignore[arg-type]
        assert result is env

    @pytest.mark.asyncio
    async def test_pipeline_runs_in_declaration_order(self) -> None:
        rule = Rule(
            transforms=[
                TransformSpec(stage="pre_send", apply="upper"),
                TransformSpec(stage="pre_send", apply="exclaim"),
            ],
        )
        pipeline = TransformPipeline.build(rule, registry=_make_registry())
        env = _make_envelope(content="hello")
        result = await pipeline.run_pre_send(env, _StubClient())  # type: ignore[arg-type]
        assert result is not None
        # "hello" → upper → "HELLO" → exclaim → "HELLO!"
        assert result.event_data["content"] == "HELLO!"

    @pytest.mark.asyncio
    async def test_rejection_short_circuits_remainder(self) -> None:
        rule = Rule(
            transforms=[
                TransformSpec(stage="pre_send", apply="reject"),
                TransformSpec(stage="pre_send", apply="upper"),
            ],
        )
        pipeline = TransformPipeline.build(rule, registry=_make_registry())
        result = await pipeline.run_pre_send(
            _make_envelope(content="unchanged"),
            _StubClient(),  # type: ignore[arg-type]
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_send_and_pre_receive_are_independent(self) -> None:
        rule = Rule(
            transforms=[
                TransformSpec(stage="pre_send", apply="upper"),
                TransformSpec(stage="pre_receive", apply="exclaim"),
            ],
        )
        pipeline = TransformPipeline.build(rule, registry=_make_registry())
        client = _StubClient()
        pre_send = await pipeline.run_pre_send(
            _make_envelope(content="send"), client  # type: ignore[arg-type]
        )
        pre_rcv = await pipeline.run_pre_receive(
            _make_envelope(content="recv"), client  # type: ignore[arg-type]
        )
        assert pre_send is not None and pre_send.event_data["content"] == "SEND"
        assert pre_rcv is not None and pre_rcv.event_data["content"] == "recv!"

    @pytest.mark.asyncio
    async def test_post_send_side_effects_run(self) -> None:
        reg = TransformRegistry()
        seen: list[str] = []

        async def _observe(envelope: Envelope, ctx: TransformContext) -> None:
            seen.append(envelope.event_data.get("content", ""))
            return None

        reg.register("observe", lambda: _observe)
        rule = Rule(
            transforms=[TransformSpec(stage="post_send", apply="observe")]
        )
        pipeline = TransformPipeline.build(rule, registry=reg)
        env = _make_envelope(content="recorded")
        await pipeline.run_post_send(env, _StubClient())  # type: ignore[arg-type]
        assert seen == ["recorded"]

    @pytest.mark.asyncio
    async def test_post_stage_exception_is_logged_not_raised(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reg = TransformRegistry()

        async def _boom(envelope: Envelope, ctx: TransformContext) -> None:
            raise RuntimeError("post stage boom")

        reg.register("boom", lambda: _boom)
        rule = Rule(
            transforms=[TransformSpec(stage="post_receive", apply="boom")]
        )
        pipeline = TransformPipeline.build(rule, registry=reg)
        # Must not raise.
        await pipeline.run_post_receive(
            _make_envelope(), _StubClient()  # type: ignore[arg-type]
        )

    @pytest.mark.asyncio
    async def test_pre_stage_exception_is_reject(self) -> None:
        reg = TransformRegistry()

        async def _boom(envelope: Envelope, ctx: TransformContext) -> None:
            raise RuntimeError("pre stage boom")

        reg.register("boom", lambda: _boom)
        rule = Rule(
            transforms=[TransformSpec(stage="pre_send", apply="boom")]
        )
        pipeline = TransformPipeline.build(rule, registry=reg)
        result = await pipeline.run_pre_send(
            _make_envelope(), _StubClient()  # type: ignore[arg-type]
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_exec_form_logs_once_and_passes_through(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rule = Rule(
            transforms=[
                TransformSpec(
                    stage="pre_send",
                    apply={"exec": ["/bin/echo"]},
                ),
                # Known form on the same stage.
                TransformSpec(stage="pre_send", apply="upper"),
            ],
        )
        import logging

        with caplog.at_level(
            logging.WARNING,
            logger="autogen.beta.network.client.transforms.pipeline",
        ):
            pipeline = TransformPipeline.build(rule, registry=_make_registry())
            result = await pipeline.run_pre_send(
                _make_envelope(content="x"), _StubClient()  # type: ignore[arg-type]
            )
        # Exec was skipped; upper still ran.
        assert result is not None
        assert result.event_data["content"] == "X"
        # Exactly one warning emitted for the exec form.
        exec_warnings = [
            r for r in caplog.records if "form=exec" in r.getMessage()
        ]
        assert len(exec_warnings) == 1

    @pytest.mark.asyncio
    async def test_unknown_ws_form_logs_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rule = Rule(
            transforms=[
                TransformSpec(
                    stage="pre_send",
                    apply={"ws": "ws://localhost:9000"},
                ),
                TransformSpec(
                    stage="pre_send",
                    apply={"ws": "ws://localhost:9000"},
                ),
            ],
        )
        import logging

        with caplog.at_level(
            logging.WARNING,
            logger="autogen.beta.network.client.transforms.pipeline",
        ):
            TransformPipeline.build(rule, registry=_make_registry())
        ws_warnings = [
            r for r in caplog.records if "form=ws" in r.getMessage()
        ]
        # Two specs but one warning — de-dup by (stage, form).
        assert len(ws_warnings) == 1

    @pytest.mark.asyncio
    async def test_when_filter_gates_transform(self) -> None:
        rule = Rule(
            transforms=[
                TransformSpec(
                    stage="pre_send",
                    apply="upper",
                    when={"session_type": "consulting"},
                ),
            ],
        )
        pipeline = TransformPipeline.build(rule, registry=_make_registry())
        # Session type matches → transform runs.
        matched_client = _StubClient(
            session_types={"sess-1": "consulting"},
        )
        out = await pipeline.run_pre_send(
            _make_envelope(content="hi", session_id="sess-1"),
            matched_client,  # type: ignore[arg-type]
        )
        assert out is not None and out.event_data["content"] == "HI"
        # Session type mismatch → transform skipped.
        mismatched = _StubClient(
            session_types={"sess-1": "conversation"},
        )
        out = await pipeline.run_pre_send(
            _make_envelope(content="hi", session_id="sess-1"),
            mismatched,  # type: ignore[arg-type]
        )
        assert out is not None and out.event_data["content"] == "hi"

    @pytest.mark.asyncio
    async def test_rule_version_propagates_to_context(self) -> None:
        reg = TransformRegistry()
        captured: list[int] = []

        async def _capture(
            envelope: Envelope, ctx: TransformContext
        ) -> Envelope:
            captured.append(ctx.rule_version)
            return envelope

        reg.register("capture", lambda: _capture)
        rule = Rule(
            version=42,
            transforms=[TransformSpec(stage="pre_send", apply="capture")],
        )
        pipeline = TransformPipeline.build(rule, registry=reg)
        await pipeline.run_pre_send(
            _make_envelope(), _StubClient()  # type: ignore[arg-type]
        )
        assert captured == [42]

    @pytest.mark.asyncio
    async def test_direction_is_outbound_for_pre_send(self) -> None:
        reg = TransformRegistry()
        captured: list[str] = []

        async def _capture(
            envelope: Envelope, ctx: TransformContext
        ) -> Envelope:
            captured.append(ctx.direction)
            return envelope

        reg.register("capture", lambda: _capture)
        rule = Rule(
            transforms=[
                TransformSpec(stage="pre_send", apply="capture"),
                TransformSpec(stage="pre_receive", apply="capture"),
            ],
        )
        pipeline = TransformPipeline.build(rule, registry=reg)
        await pipeline.run_pre_send(
            _make_envelope(), _StubClient()  # type: ignore[arg-type]
        )
        await pipeline.run_pre_receive(
            _make_envelope(), _StubClient()  # type: ignore[arg-type]
        )
        assert captured == ["outbound", "inbound"]

    @pytest.mark.asyncio
    async def test_aclose_closes_adapter_instances(self) -> None:
        closed = []

        class _Closable:
            async def __call__(
                self, envelope: Envelope, ctx: TransformContext
            ) -> Envelope:
                return envelope

            async def aclose(self) -> None:
                closed.append(True)

        reg = TransformRegistry()
        reg.register("closable", lambda: _Closable())
        rule = Rule(
            transforms=[TransformSpec(stage="pre_send", apply="closable")]
        )
        pipeline = TransformPipeline.build(rule, registry=reg)
        await pipeline.aclose()
        assert closed == [True]

    def test_unknown_name_fails_at_build(self) -> None:
        rule = Rule(
            transforms=[
                TransformSpec(stage="pre_send", apply="never_registered"),
            ],
        )
        with pytest.raises(TransformLookupError):
            TransformPipeline.build(rule, registry=_make_registry())
