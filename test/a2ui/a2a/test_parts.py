# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.compat.v0_3 import types as v0_3
from a2a.compat.v0_3.conversions import to_compat_part, to_core_part
from a2a.types import Part

from ag2.a2a.mappers import struct_to_dict
from ag2.a2ui.a2a import create_a2ui_parts, get_a2ui_data, is_a2ui_part
from ag2.a2ui.constants import A2UI_MIME_TYPE

SAMPLE_OPS = [
    {
        "version": "v0.9",
        "createSurface": {
            "surfaceId": "s1",
            "catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
        },
    },
    {"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}},
]


class TestCreateA2UIParts:
    def test_canonical_single_part_carries_full_list(self) -> None:
        parts = create_a2ui_parts(SAMPLE_OPS)
        assert len(parts) == 1
        assert is_a2ui_part(parts[0])
        assert get_a2ui_data(parts[0]) == SAMPLE_OPS

    def test_datapart_uses_canonical_a2ui_mime(self) -> None:
        # The canonical A2A encoding for A2UI uses MIME ``application/a2ui+json``;
        # ``application/json+a2ui`` would break A2A interop.
        assert A2UI_MIME_TYPE == "application/a2ui+json"
        [part] = create_a2ui_parts(SAMPLE_OPS)
        assert part.media_type == A2UI_MIME_TYPE

    def test_datapart_carries_the_mime_in_its_metadata(self) -> None:
        # The A2UI A2A extension spec (v1.0) identifies an A2UI part by ``metadata["mimeType"]``.
        [part] = create_a2ui_parts(SAMPLE_OPS)
        assert struct_to_dict(part.metadata) == {"mimeType": A2UI_MIME_TYPE}

    def test_canonical_wraps_single_dict_in_list(self) -> None:
        parts = create_a2ui_parts(SAMPLE_OPS[0])
        assert len(parts) == 1
        assert get_a2ui_data(parts[0]) == [SAMPLE_OPS[0]]

    def test_legacy_split_one_part_per_op(self) -> None:
        parts = create_a2ui_parts(SAMPLE_OPS, legacy_split=True)
        assert len(parts) == 2
        assert get_a2ui_data(parts[0]) == SAMPLE_OPS[0]
        assert get_a2ui_data(parts[1]) == SAMPLE_OPS[1]


class TestGetA2UIData:
    def test_returns_none_for_text_part(self) -> None:
        assert get_a2ui_data(Part(text="hi")) is None

    def test_roundtrip_list_payload(self) -> None:
        [part] = create_a2ui_parts(SAMPLE_OPS)
        decoded = get_a2ui_data(part)
        assert isinstance(decoded, list)
        assert decoded == SAMPLE_OPS

    def test_roundtrip_dict_payload(self) -> None:
        [part] = create_a2ui_parts(SAMPLE_OPS[0], legacy_split=True)
        decoded = get_a2ui_data(part)
        assert isinstance(decoded, dict)
        assert decoded == SAMPLE_OPS[0]


class TestUpstreamPeers:
    """Peers on the upstream ``a2ui_a2a`` helpers speak A2A v0.3 and mark parts in ``metadata["mimeType"]``."""

    def test_an_upstream_peer_recognises_our_parts(self) -> None:
        # An A2A v0.3 DataPart holds an object, so a v0.3 peer gets one part per message.
        [part] = create_a2ui_parts(SAMPLE_OPS[0], legacy_split=True)

        compat = to_compat_part(part)

        assert compat == v0_3.Part(root=v0_3.DataPart(data=SAMPLE_OPS[0], metadata={"mimeType": A2UI_MIME_TYPE}))

    def test_parts_from_an_upstream_peer_are_recognised(self) -> None:
        upstream = v0_3.Part(root=v0_3.DataPart(data=SAMPLE_OPS[0], metadata={"mimeType": A2UI_MIME_TYPE}))

        part = to_core_part(upstream)

        assert is_a2ui_part(part)
        assert get_a2ui_data(part) == SAMPLE_OPS[0]

    @pytest.mark.parametrize("mime_type", ["application/json", "application/json+a2ui"])
    def test_a_metadata_mime_of_another_type_is_not_a2ui(self, mime_type: str) -> None:
        upstream = v0_3.Part(root=v0_3.DataPart(data=SAMPLE_OPS[0], metadata={"mimeType": mime_type}))

        part = to_core_part(upstream)

        assert not is_a2ui_part(part)
