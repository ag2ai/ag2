# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A hub client is built over either shipped link.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. The fixture
carries no expectations: a clean run is the assertion.
"""

from ag2.network import Hub, HubClient, LocalLink, WsLink


def in_process(hub: Hub) -> HubClient:
    return HubClient(LocalLink(hub), hub=hub)


def over_the_wire() -> HubClient:
    # `WsLinkClient.endpoint_id` is read-only: the hub's welcome frame assigns it.
    return HubClient(WsLink("ws://hub.example:8765"))
