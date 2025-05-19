# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.import_utils import optional_import_block

from .patch_fastapi_code_generator import (  # noqa: E402
    SUCCESFUL_IMPORT,
    patch_function_name_parsing,
    patch_generate_code,
)

if SUCCESFUL_IMPORT:
    patch_function_name_parsing()
    patch_generate_code()

from .mcp_proxy import MCPProxy  # noqa: E402

__all__ = ["MCPProxy", "optional_import_block"]
