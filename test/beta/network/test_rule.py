# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rule dataclass and its access-block matchers."""

from __future__ import annotations

import pytest

from autogen.beta.network.rule import (
    AccessBlock,
    LimitsBlock,
    Rule,
    SessionTypeAccess,
    TransformSpec,
    parse_duration,
)
from autogen.beta.network.session_types import SessionType


# ---------------------------------------------------------------------------
# Access matching
# ---------------------------------------------------------------------------


def test_access_default_allows_all_inbound_and_outbound() -> None:
    access = AccessBlock()
    assert access.allows_inbound("ag2:alice:1")
    assert access.allows_inbound("any:thing:42")
    assert access.allows_outbound("ag2:bob:1")


def test_access_restricted_inbound_list_rejects_non_matches() -> None:
    access = AccessBlock(inbound_from=["ag2:*:*"])
    assert access.allows_inbound("ag2:writer:1")
    assert not access.allows_inbound("acme:leaker:1")


def test_access_restricted_outbound_list_rejects_non_matches() -> None:
    access = AccessBlock(outbound_to=["owner:*:*", "ag2:writer:1"])
    assert access.allows_outbound("ag2:writer:1")
    assert access.allows_outbound("owner:anyone:42")
    assert not access.allows_outbound("ag2:researcher:1")


def test_session_type_access_defaults_allow_every_type() -> None:
    stypes = SessionTypeAccess()
    for t in SessionType:
        assert stypes.may_initiate(t)
        assert stypes.may_accept(t)


def test_session_type_access_respects_restriction() -> None:
    stypes = SessionTypeAccess(
        initiate=[SessionType.CONSULTING.value, SessionType.CONVERSATION.value],
        accept=[SessionType.CONSULTING.value],
    )
    assert stypes.may_initiate(SessionType.CONSULTING)
    assert not stypes.may_initiate(SessionType.BROADCAST)
    assert stypes.may_accept(SessionType.CONSULTING)
    assert not stypes.may_accept(SessionType.CONVERSATION)


def test_session_type_access_accepts_string_type() -> None:
    stypes = SessionTypeAccess(initiate=["consulting"])
    assert stypes.may_initiate("consulting")
    assert not stypes.may_initiate("broadcast")


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_limits_defaults_are_permissive() -> None:
    limits = LimitsBlock()
    assert limits.max_concurrent_sessions == 32
    assert limits.session_ttl_default == "2h"
    assert limits.delegation_depth == 5


def test_limits_roundtrip_preserves_integer_coercion() -> None:
    data = LimitsBlock(max_concurrent_sessions=5, delegation_depth=3).to_dict()
    restored = LimitsBlock.from_dict(data)
    assert restored.max_concurrent_sessions == 5
    assert restored.delegation_depth == 3


def test_limits_session_ttl_seconds_parses_default() -> None:
    assert LimitsBlock().session_ttl_seconds() == 7200  # "2h"
    assert LimitsBlock(session_ttl_default="15m").session_ttl_seconds() == 900
    assert LimitsBlock(session_ttl_default="30s").session_ttl_seconds() == 30
    assert LimitsBlock(session_ttl_default="1d").session_ttl_seconds() == 86400


@pytest.mark.parametrize(
    "text,expected",
    [
        ("30s", 30),
        ("15m", 900),
        ("2h", 7200),
        ("1d", 86400),
        ("42", 42),
    ],
)
def test_parse_duration_accepts_units(text: str, expected: int) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("bad", ["", "abc", "10x", "m10"])
def test_parse_duration_rejects_garbage(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(bad)


# ---------------------------------------------------------------------------
# Rule roundtrip
# ---------------------------------------------------------------------------


def test_rule_defaults_round_trip() -> None:
    rule = Rule()
    restored = Rule.from_dict(rule.to_dict())
    assert restored.access.allows_inbound("anybody")
    assert restored.limits.max_concurrent_sessions == 32
    assert restored.transforms == []


def test_rule_preserves_transforms_verbatim() -> None:
    rule = Rule(
        transforms=[
            TransformSpec(stage="pre_receive", apply="redact_pii"),
            TransformSpec(
                stage="pre_send",
                apply={"python": {"module": "m", "class": "C", "config": {}}},
                when={"session_type": "consulting"},
            ),
        ],
    )
    restored = Rule.from_json(rule.to_json())
    assert len(restored.transforms) == 2
    assert restored.transforms[0].stage == "pre_receive"
    assert restored.transforms[0].apply == "redact_pii"
    assert restored.transforms[1].apply == {
        "python": {"module": "m", "class": "C", "config": {}}
    }
    assert restored.transforms[1].when == {"session_type": "consulting"}


def test_rule_restricting_consulting_only() -> None:
    rule = Rule(
        access=AccessBlock(
            inbound_from=["ag2:*:*"],
            outbound_to=["ag2:*:*"],
            session_types=SessionTypeAccess(
                initiate=[SessionType.CONSULTING.value],
                accept=[SessionType.CONSULTING.value],
            ),
        ),
        limits=LimitsBlock(max_concurrent_sessions=2),
    )
    assert rule.access.allows_inbound("ag2:researcher:1")
    assert not rule.access.allows_inbound("acme:other:9")
    assert rule.access.session_types.may_initiate(SessionType.CONSULTING)
    assert not rule.access.session_types.may_initiate(SessionType.BROADCAST)
    assert rule.limits.max_concurrent_sessions == 2
