# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

from ag2.a2ui.constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG
from ag2.a2ui.parser import A2UIResponseParser
from ag2.a2ui.serialize import to_jsonl

SAMPLE_OPS = [
    {
        "version": "v0.9",
        "createSurface": {
            "surfaceId": "s1",
            "catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
        },
    },
    {
        "version": "v0.9",
        "updateComponents": {"surfaceId": "s1", "components": [{"id": "root", "component": "Text", "text": "hi"}]},
    },
    {"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}},
]


class TestToJsonl:
    def test_one_message_per_line(self) -> None:
        jsonl = to_jsonl(SAMPLE_OPS)
        lines = jsonl.split("\n")
        assert len(lines) == len(SAMPLE_OPS)
        assert [json.loads(line) for line in lines] == SAMPLE_OPS

    def test_empty_sequence_yields_empty_string(self) -> None:
        assert to_jsonl([]) == ""

    def test_no_trailing_newline(self) -> None:
        assert not to_jsonl(SAMPLE_OPS).endswith("\n")

    def test_roundtrip_parse_array_to_jsonl(self) -> None:
        # parse(text + <a2ui-json> JSON array </a2ui-json>) -> operations ->
        # to_jsonl -> per-line json.loads recovers the original messages.
        parser = A2UIResponseParser(version_string="v0.9")
        response = f"Here is the UI.\n{A2UI_JSON_OPEN_TAG}\n{json.dumps(SAMPLE_OPS)}\n{A2UI_JSON_CLOSE_TAG}"
        parsed = parser.parse(response)
        assert parsed.has_a2ui

        jsonl = to_jsonl(parsed.operations)
        recovered = [json.loads(line) for line in jsonl.split("\n")]
        assert recovered == list(parsed.operations) == SAMPLE_OPS
