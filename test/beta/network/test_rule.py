# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Rule dataclass and its access-block matchers."""

from __future__ import annotations

import pytest

from autogen.beta.network.rule import (
    AccessBlock,
    KnowledgeAccess,
    LimitsBlock,
    Rule,
    SessionTypeAccess,
    TransformSpec,
    TransformStage,
    _match_path,
    _normalize_store_path,
    _path_glob_match,
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


class TestTransformStageValidation:
    """Phase 5a.1 — ``TransformSpec`` validates the ``stage`` field."""

    @pytest.mark.parametrize(
        "stage",
        ["pre_send", "post_send", "pre_receive", "post_receive"],
    )
    def test_all_four_valid_stages_accepted(self, stage: str) -> None:
        spec = TransformSpec(stage=stage, apply="redact_pii")
        assert spec.stage == stage

    def test_enum_members_also_accepted(self) -> None:
        spec = TransformSpec(
            stage=TransformStage.PRE_SEND.value,
            apply="redact_pii",
        )
        assert spec.stage == "pre_send"

    @pytest.mark.parametrize(
        "bad",
        ["", "presend", "pre-send", "Pre_Send", "before", "PRE_SEND"],
    )
    def test_invalid_stage_raises(self, bad: str) -> None:
        with pytest.raises(ValueError, match="TransformSpec.stage"):
            TransformSpec(stage=bad, apply="redact_pii")

    def test_from_dict_rejects_unknown_stage(self) -> None:
        with pytest.raises(ValueError, match="TransformSpec.stage"):
            TransformSpec.from_dict({"stage": "bogus", "apply": "x"})

    def test_from_dict_missing_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="'stage' field"):
            TransformSpec.from_dict({"apply": "x"})

    def test_rule_from_dict_rejects_bad_stage(self) -> None:
        data = {"transforms": [{"stage": "typo_stage", "apply": "x"}]}
        with pytest.raises(ValueError, match="TransformSpec.stage"):
            Rule.from_dict(data)

    def test_transform_stage_enum_values(self) -> None:
        # Freeze the wire values so renames can't silently break on-disk
        # rules that were written against an earlier hub.
        assert TransformStage.PRE_SEND.value == "pre_send"
        assert TransformStage.POST_SEND.value == "post_send"
        assert TransformStage.PRE_RECEIVE.value == "pre_receive"
        assert TransformStage.POST_RECEIVE.value == "post_receive"
        assert TransformStage("pre_send") is TransformStage.PRE_SEND


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


# ---------------------------------------------------------------------------
# KnowledgeAccess — cross-actor KnowledgeStore exposure (Phase 3b)
# ---------------------------------------------------------------------------


class TestKnowledgeAccessDefaults:
    def test_default_denies_everything(self) -> None:
        ka = KnowledgeAccess()
        assert not ka.allows(reader_name="ag2:writer:1", path="/any/path")

    def test_empty_readers_denies_even_if_expose_matches(self) -> None:
        ka = KnowledgeAccess(expose=["/**"])
        assert not ka.allows(reader_name="ag2:writer:1", path="/foo")

    def test_empty_expose_denies_even_if_reader_matches(self) -> None:
        ka = KnowledgeAccess(readers=["*"])
        assert not ka.allows(reader_name="ag2:writer:1", path="/foo")


class TestKnowledgeAccessExposeMatching:
    def test_exact_path(self) -> None:
        ka = KnowledgeAccess(expose=["/foo/bar.txt"], readers=["*"])
        assert ka.allows(reader_name="x", path="/foo/bar.txt")
        assert not ka.allows(reader_name="x", path="/foo/bar")
        assert not ka.allows(reader_name="x", path="/foo/bar.txtx")

    def test_directory_prefix_matches_self_and_children(self) -> None:
        ka = KnowledgeAccess(expose=["/memory/biomed/"], readers=["*"])
        assert ka.allows(reader_name="x", path="/memory/biomed/")
        assert ka.allows(reader_name="x", path="/memory/biomed")
        assert ka.allows(reader_name="x", path="/memory/biomed/abstract1.txt")
        assert ka.allows(reader_name="x", path="/memory/biomed/2024/q1/summary.md")
        assert not ka.allows(reader_name="x", path="/memory/clinical/x.txt")

    def test_single_star_matches_one_segment(self) -> None:
        ka = KnowledgeAccess(expose=["/artifacts/*.md"], readers=["*"])
        assert ka.allows(reader_name="x", path="/artifacts/README.md")
        # Single-star must NOT cross `/`
        assert not ka.allows(reader_name="x", path="/artifacts/docs/intro.md")

    def test_double_star_matches_any_depth(self) -> None:
        ka = KnowledgeAccess(expose=["/artifacts/public/**"], readers=["*"])
        assert ka.allows(reader_name="x", path="/artifacts/public/a.txt")
        assert ka.allows(reader_name="x", path="/artifacts/public/deep/nest/file")
        assert not ka.allows(reader_name="x", path="/artifacts/private/secret")

    def test_question_mark_matches_single_char(self) -> None:
        ka = KnowledgeAccess(expose=["/v?/index"], readers=["*"])
        assert ka.allows(reader_name="x", path="/v1/index")
        assert ka.allows(reader_name="x", path="/v9/index")
        assert ka.allows(reader_name="x", path="/vv/index")
        # `?` matches exactly one non-slash char, not two
        assert not ka.allows(reader_name="x", path="/v10/index")
        # `?` does not cross `/`
        assert not ka.allows(reader_name="x", path="/v/index")

    def test_any_pattern_hit_wins(self) -> None:
        ka = KnowledgeAccess(
            expose=["/a/**", "/b/**"],
            readers=["*"],
        )
        assert ka.allows(reader_name="x", path="/a/1")
        assert ka.allows(reader_name="x", path="/b/nested/2")
        assert not ka.allows(reader_name="x", path="/c/1")


class TestKnowledgeAccessReaderMatching:
    def test_exact_reader_name(self) -> None:
        ka = KnowledgeAccess(expose=["/**"], readers=["ag2:writer:1"])
        assert ka.allows(reader_name="ag2:writer:1", path="/foo")
        assert not ka.allows(reader_name="ag2:writer:2", path="/foo")

    def test_wildcard_reader_pattern(self) -> None:
        ka = KnowledgeAccess(expose=["/**"], readers=["ag2:*:*"])
        assert ka.allows(reader_name="ag2:writer:1", path="/foo")
        assert ka.allows(reader_name="ag2:researcher:42", path="/foo")
        assert not ka.allows(reader_name="acme:user:1", path="/foo")

    def test_star_allows_any_reader(self) -> None:
        ka = KnowledgeAccess(expose=["/public/**"], readers=["*"])
        assert ka.allows(reader_name="anyone", path="/public/a")
        assert ka.allows(reader_name="ag2:x:1", path="/public/b")

    def test_multiple_reader_patterns(self) -> None:
        ka = KnowledgeAccess(
            expose=["/**"],
            readers=["ag2:*:*", "acme:senior_*"],
        )
        assert ka.allows(reader_name="ag2:writer:1", path="/x")
        assert ka.allows(reader_name="acme:senior_lead", path="/x")
        assert not ka.allows(reader_name="acme:junior", path="/x")


class TestKnowledgeAccessCombined:
    def test_reader_allowed_path_denied(self) -> None:
        ka = KnowledgeAccess(expose=["/public/**"], readers=["ag2:*:*"])
        assert not ka.allows(reader_name="ag2:r:1", path="/private/x")

    def test_path_allowed_reader_denied(self) -> None:
        ka = KnowledgeAccess(expose=["/public/**"], readers=["ag2:*:*"])
        assert not ka.allows(reader_name="acme:other:1", path="/public/x")

    def test_both_allowed(self) -> None:
        ka = KnowledgeAccess(
            expose=["/artifacts/public/**", "/memory/shared.md"],
            readers=["ag2:*:*"],
        )
        assert ka.allows(reader_name="ag2:writer:1", path="/artifacts/public/intro")
        assert ka.allows(reader_name="ag2:writer:1", path="/memory/shared.md")
        assert not ka.allows(reader_name="ag2:writer:1", path="/memory/private.md")


class TestKnowledgeAccessRoundTrip:
    def test_empty_default_round_trip(self) -> None:
        data = KnowledgeAccess().to_dict()
        assert data == {"expose": [], "readers": []}
        restored = KnowledgeAccess.from_dict(data)
        assert restored.expose == []
        assert restored.readers == []

    def test_populated_round_trip(self) -> None:
        original = KnowledgeAccess(
            expose=["/artifacts/public/**", "/memory/biomed/"],
            readers=["ag2:*:*", "acme:senior_research_lead"],
        )
        restored = KnowledgeAccess.from_dict(original.to_dict())
        assert restored.expose == original.expose
        assert restored.readers == original.readers

    def test_access_block_round_trip_includes_knowledge(self) -> None:
        access = AccessBlock(
            knowledge=KnowledgeAccess(
                expose=["/public/**"],
                readers=["ag2:*:*"],
            )
        )
        restored = AccessBlock.from_dict(access.to_dict())
        assert restored.knowledge.expose == ["/public/**"]
        assert restored.knowledge.readers == ["ag2:*:*"]
        assert restored.knowledge.allows(
            reader_name="ag2:writer:1", path="/public/foo"
        )

    def test_rule_json_round_trip_includes_knowledge(self) -> None:
        rule = Rule(
            access=AccessBlock(
                knowledge=KnowledgeAccess(
                    expose=["/docs/**"],
                    readers=["*"],
                )
            )
        )
        restored = Rule.from_json(rule.to_json())
        assert restored.access.knowledge.expose == ["/docs/**"]
        assert restored.access.knowledge.readers == ["*"]

    def test_from_dict_missing_knowledge_block_is_empty(self) -> None:
        raw = {
            "inbound_from": ["*"],
            "outbound_to": ["*"],
            "session_types": {},
            "subscribe": {},
        }  # no "knowledge" key
        access = AccessBlock.from_dict(raw)
        assert access.knowledge.expose == []
        assert access.knowledge.readers == []
        assert not access.knowledge.allows(reader_name="x", path="/any")


# ---------------------------------------------------------------------------
# Path helpers (low level)
# ---------------------------------------------------------------------------


class TestPathNormalization:
    def test_adds_leading_slash(self) -> None:
        assert _normalize_store_path("foo/bar") == "/foo/bar"

    def test_preserves_leading_slash(self) -> None:
        assert _normalize_store_path("/foo/bar") == "/foo/bar"

    def test_collapses_double_slashes(self) -> None:
        assert _normalize_store_path("//foo//bar//baz") == "/foo/bar/baz"

    def test_root_alone(self) -> None:
        assert _normalize_store_path("/") == "/"


class TestPathGlobMatch:
    def test_empty_patterns_never_matches(self) -> None:
        assert not _match_path("/foo", [])

    def test_literal_match(self) -> None:
        assert _path_glob_match("/foo/bar", "/foo/bar")

    def test_prefix_trailing_slash(self) -> None:
        assert _path_glob_match("/foo/bar/baz", "/foo/")
        assert _path_glob_match("/foo", "/foo/")

    def test_double_star_depth(self) -> None:
        assert _path_glob_match("/a/b/c/d", "/a/**")
        assert _path_glob_match("/a", "/**")
        assert _path_glob_match("/", "/**")

    def test_single_star_is_segment_scoped(self) -> None:
        assert _path_glob_match("/a/b.txt", "/a/*.txt")
        assert not _path_glob_match("/a/b/c.txt", "/a/*.txt")

    def test_no_match(self) -> None:
        assert not _path_glob_match("/foo/x", "/bar/**")
