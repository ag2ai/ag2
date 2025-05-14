# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from autogen.import_utils import optional_import_block
from autogen.mcp.mcp_proxy.security import BaseSecurity

with optional_import_block() as result:
    from fastapi_code_generator.parser import OpenAPIParser
    from fastapi_code_generator.visitor import Visitor


def custom_visitor(parser: "OpenAPIParser", model_path: Path) -> dict[str, object]:
    if "components" not in parser.raw_obj or "securitySchemes" not in parser.raw_obj["components"]:
        return {}
    security_schemes = parser.raw_obj["components"]["securitySchemes"]
    server_url = parser.raw_obj["servers"][0]["url"]

    security_classes = []
    security_parameters = {}
    for k, v in security_schemes.items():
        v["server_url"] = server_url
        security_class = BaseSecurity.get_security_class(type=v["type"], schema_parameters=v)

        security_classes.append(security_class.__name__)

        security_parameters[k] = security_class.get_security_parameters(schema_parameters=v)

    return {
        "security_schemes": security_schemes,
        "security_classes": security_classes,
        "security_parameters": security_parameters,
    }


visit: "Visitor" = custom_visitor
