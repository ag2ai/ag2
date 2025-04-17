# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Dict, List

from fastapi_code_generator.parser import OpenAPIParser, Operation
from fastapi_code_generator.visitor import Visitor


def custom_visitor(parser: OpenAPIParser, model_path: Path) -> Dict[str, object]:
    sorted_operations: List[Operation] = sorted(parser.operations.values(), key=lambda m: m.path)
    for operation in sorted_operations:
        # Rename the operation ID to include the path and method
        operation.function_name = "some_func_name"
    return {"operations": sorted_operations}


visit: Visitor = custom_visitor
