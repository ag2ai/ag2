# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from typing import Dict, List

from autogen.import_utils import optional_import_block

with optional_import_block() as result:
    from fastapi_code_generator.parser import OpenAPIParser, Operation
    from fastapi_code_generator.visitor import Visitor

from pydantic import BaseModel

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.llm_config import LLMConfig


class Group(BaseModel):
    name: str
    description: str


class GroupSuggestions(BaseModel):
    groups: list[Group]


class GroupNames(BaseModel):
    groups: list[str]


logger = logging.getLogger(__name__)

GROUP_DISCOVERY_MESSAGE = (
    "You are a senior Python engineer. You will be shown a batch of API functions. "
    "These functions are not guaranteed to be related — your task is to analyze them individually and find meaningful groups *within the batch*.\n\n"
    "You should propose a list of group names, and for each group, provide a short description of the kind of functions it contains.\n\n"
    "How to group:\n"
    "- Focus on what the function operates on — e.g., functions that manipulate a board go in 'board_operations', functions related to users in 'user_management', etc.\n"
    "- Do NOT assume the entire batch forms a single group.\n"
    "- Do NOT use generic categories like 'misc', 'utils', or 'general'.\n"
    "- Favor *granular but meaningful* groups. For instance, 'user_profile_handling' and 'user_authentication' could be separate if their purposes are distinct.\n\n"
    "Formatting:\n"
    "- Group names must be short, descriptive, and in snake_case.\n"
    "- Return a list of group names and a 1-2 sentence description of what each group includes.\n"
)


GROUP_ASSIGNMENT_MESSAGE = (
    "You are a senior Python engineer. You will be given a function description "
    "and a list of possible groups. Choose the most suitable groups for the function.\n"
    "- A function can belong to multiple groups.\n"
    "- The groups should be relevant to the function's purpose and functionality.\n"
    "- You MUST choose a group from the possible groups, you cannot respond with empty grouping any equivalent.\n"
)


def chunk_list(items: List, size: int) -> List[List]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def discover_groups(operations: List["Operation"], chunk_size: int = 30) -> Dict[str, str]:
    llm_config = LLMConfig.get_current_llm_config().copy()

    for config in llm_config.config_list:
        config.response_format = GroupSuggestions

    with llm_config:
        agent = ConversableAgent(name="group_discovery_agent", system_message=GROUP_DISCOVERY_MESSAGE)
        groups = {}

        for i, chunk in enumerate(chunk_list(operations, chunk_size)):
            func_descriptions = [f"- {op.function_name}: {op.summary} (args: {op.arguments})" for op in chunk]
            message = "Here are some functions:\n" + "\n".join(func_descriptions)

            response = agent.run(message=message, max_turns=1, user_input=False)

            for event in response.events:
                if event.type == "text" and event.content.sender == "group_discovery_agent":
                    # Naively parse "group_name: description" from text block
                    new_groups = GroupSuggestions.model_validate_json(event.content.content).groups
                    groups.update(new_groups)

    logger.warning("Discovered groups: %s", groups)

    # Remove duplicates
    with llm_config:
        agent = ConversableAgent(name="group_refining_agent", system_message=GROUP_DISCOVERY_MESSAGE)

        message = (
            "You need to refine the group names and descriptions to ensure they are unique.\n"
            "Here are the groups:\n" + "\n".join([f"- {name}: {desc}" for name, desc in groups.items()])
        )
        response = agent.run(message=message, max_turns=1, user_input=False)
        for event in response.events:
            if event.type == "text" and event.content.sender == "group_refining_agent":
                # Naively parse "group_name: description" from text block
                refined_groups = json.loads(event.content.content)

    return refined_groups


def assign_operation_to_group(operation: "Operation", groups: dict[str, str]) -> str:
    llm_config = LLMConfig.get_current_llm_config().copy()

    for config in llm_config.config_list:
        config.response_format = GroupNames

    with llm_config:
        agent = ConversableAgent(name="group_assignment_agent", system_message=GROUP_ASSIGNMENT_MESSAGE)

        message = (
            "Function summary:\n"
            f"{operation.summary}\n\n"
            f"Arguments: {operation.arguments}\n\n"
            f"Available groups: {json.dumps(groups)}\n\n"
            "What group should this function go in?"
        )

        response = agent.run(message=message, max_turns=1, user_input=True)

        groups = []
        for event in response.events:
            if event.type == "text" and event.content.sender == "group_assignment_agent":
                groups = GroupNames.model_validate_json(event.content.content).groups

        return groups


def refine_group_names(groups: Dict[str, str]) -> Dict[str, str]:
    # Optional: normalize names, merge similar ones (e.g., using embeddings or string similarity)
    # Placeholder for now:
    return groups


def custom_visitor(parser: "OpenAPIParser", model_path: Path) -> Dict[str, object]:
    operations = sorted(parser.operations.values(), key=lambda op: op.path)

    # ---- PASS 1: DISCOVER GROUPS ----
    logger.warning("Discovering groups...")
    discovered_groups = discover_groups(operations)
    logger.warning("Discovered groups: %s", discovered_groups)

    # ---- PASS 2: ASSIGN OPERATIONS TO GROUPS ----
    logger.warning("Assigning operations to groups...")
    for op in operations:
        logger.warning("Assigning operation %s to groups...", op.function_name)
        groups = assign_operation_to_group(op, discovered_groups)
        op.tags = groups
        logger.warning("Assigned groups: %s", groups)

    return {"operations": operations}


visit: "Visitor" = custom_visitor
