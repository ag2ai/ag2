# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ag2.a2ui._types import A2UIVersion
from ag2.a2ui.parser import A2UIResponseParser
from ag2.a2ui.schema_manager import A2UISchemaManager

# The ``version`` each protocol version's messages carry; v1.0 drops the ``v`` (a2ui-project/a2ui#3726).
WIRE_VERSION: dict[A2UIVersion, str] = {"v0.9": "v0.9", "v0.9.1": "v0.9.1", "v1.0": "1.0"}


def validating_parser(
    protocol_version: A2UIVersion, custom_catalog: dict[str, Any] | None = None
) -> A2UIResponseParser:
    """A parser that validates against the protocol version's schemas and the given custom catalog."""
    manager = A2UISchemaManager(protocol_version=protocol_version, custom_catalog=custom_catalog)
    return A2UIResponseParser(
        version_string=manager.version_string,
        server_to_client_schema=manager.server_to_client_schema,
        schema_registry=manager.build_schema_registry(),
    )


def update_components(protocol_version: A2UIVersion, *components: dict[str, Any]) -> dict[str, Any]:
    """An ``updateComponents`` message for surface ``s``."""
    return {
        "version": WIRE_VERSION[protocol_version],
        "updateComponents": {"surfaceId": "s", "components": list(components)},
    }
