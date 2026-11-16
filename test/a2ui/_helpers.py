# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ag2.a2ui._types import A2UIVersion
from ag2.a2ui.parser import A2UIResponseParser
from ag2.a2ui.schema_manager import A2UISchemaManager


def validating_parser(
    protocol_version: A2UIVersion, custom_catalog: dict[str, Any] | None = None
) -> A2UIResponseParser:
    """A parser that validates against the protocol version's schemas and the given custom catalog."""
    manager = A2UISchemaManager(protocol_version=protocol_version, custom_catalog=custom_catalog)
    return A2UIResponseParser(
        protocol=manager.protocol,
        server_to_client_schema=manager.server_to_client_schema,
        schema_registry=manager.build_schema_registry(),
        component_references=manager.component_references(),
    )


def extracting_parser(protocol_version: A2UIVersion = "v0.9") -> A2UIResponseParser:
    """A parser that extracts the A2UI block without validating it, as ``validate_responses=False`` builds it."""
    return A2UIResponseParser(protocol=A2UISchemaManager(protocol_version=protocol_version).protocol)
