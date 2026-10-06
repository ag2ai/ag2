# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.a2ui._types import server_to_client_payload_keys


def test_payload_keys_are_read_off_the_message_types() -> None:
    assert server_to_client_payload_keys() == {
        "createSurface",
        "updateComponents",
        "updateDataModel",
        "deleteSurface",
        "callFunction",
        "actionResponse",
    }
