# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Dict, List

from autogen.import_utils import optional_import_block

with optional_import_block() as result:
    from fastapi_code_generator.parser import OpenAPIParser, Operation
    from fastapi_code_generator.visitor import Visitor

from autogen.agentchat.conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)

# System prompt to guide the AI agent in naming functions
SYSTEM_MESSAGE = (
    "You are a helpful expert Python programmer. Your task is to generate a clear, concise, "
    "and descriptive name for a Python function based on a user-provided summary.\n"
    "- Only return the new function name.\n"
    "- The name must be fewer than 64 characters.\n"
    "- It should reflect the purpose of the function as described.\n"
    "- You will be provided with a list of already-taken names, which must not be reused.\n"
    "- The function name should be in snake_case.\n"
)


def validate_function_name(name: str, taken_names: List[str]) -> str:
    """
    Validate the generated function name against length, format, and uniqueness constraints.

    Returns:
        'exit' if the name is valid, or an error message string otherwise.
    """
    if len(name) > 64:
        return "Function name is too long. Please provide a shorter name."
    if not name.islower() or " " in name:
        return "Function name must be in snake_case."
    if name in taken_names:
        return f"Function name is already taken. Please provide a different name. Taken names: {', '.join(taken_names)}"
    return "exit"


def get_new_function_name(operation: "Operation", taken_names: List[str]) -> str:
    """
    Ask an AI agent to generate a new function name for a given OpenAPI operation.

    Args:
        operation: The OpenAPI operation that needs renaming.
        taken_names: A list of names already used, to avoid collisions.

    Returns:
        A new, validated function name.
    """
    agent = ConversableAgent(
        name="helpful_agent",
        system_message=SYSTEM_MESSAGE,
    )

    response = agent.run(
        message=(
            "How would you name this function? \n"
            f"Info:\n"
            f"- old function name: {operation.function_name}\n"
            f"- function summary: {operation.summary}\n"
            f"- function arguments: {operation.arguments}\n"
        ),
        user_input=True,
    )

    proposed_name = None

    for event in response.events:
        if event.type == "text" and event.content.sender == "helpful_agent":
            proposed_name = event.content.content.strip()
        elif event.type == "input_request":
            reply = (
                validate_function_name(proposed_name, taken_names)
                if proposed_name
                else "Please provide a function name."
            )
            event.content.respond(reply)

    logger.warning(f"Renamed operation '{operation.function_name}' to '{response.summary}'.")
    return response.summary


def custom_visitor(parser: "OpenAPIParser", model_path: Path) -> Dict[str, object]:
    """
    Visits and optionally renames operations in the OpenAPI parser.

    Args:
        parser: An OpenAPIParser instance containing API operations.
        model_path: Path to the model (not used in this implementation).

    Returns:
        A dictionary containing the updated list of operations.
    """
    operations = sorted(parser.operations.values(), key=lambda op: op.path)
    taken_names = []

    for op in operations:
        if len(op.function_name) > 64:
            new_name = get_new_function_name(op, taken_names)
            op.function_name = new_name
        taken_names.append(op.function_name)

    return {"operations": operations}


visit: "Visitor" = custom_visitor
