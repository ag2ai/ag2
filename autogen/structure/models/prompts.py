# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0

from pydantic import JsonValue

instruction_field_description = (
    "Define the core task or instruction to be executed. Your instruction should:\n\n"
    "1. Be specific and actionable\n"
    "2. Clearly state the expected outcome\n"
    "3. Include any critical constraints or requirements\n\n"
    "**Guidelines for writing effective instructions:**\n"
    "- Start with a clear action verb (e.g., analyze, create, evaluate)\n"
    "- Specify the scope and boundaries of the task\n"
    "- Include success criteria when applicable\n"
    "- Break complex tasks into distinct steps\n\n"
    "**Examples:**\n"
    "- 'Analyze the provided sales data and identify top 3 performing products'\n"
    "- 'Generate a Python function that validates email addresses'\n"
    "- 'Create a data visualization showing monthly revenue trends'"
)

guidance_field_description = (
    "Provide strategic direction and constraints for task execution.\n\n"
    "**Key components to include:**\n"
    "1. Methodological preferences\n"
    "2. Quality standards and requirements\n"
    "3. Specific limitations or boundaries\n"
    "4. Performance expectations\n\n"
    "**Best practices:**\n"
    "- Be explicit about any assumptions that should be made\n"
    "- Specify preferred approaches or techniques\n"
    "- Detail any constraints on resources or methods\n"
    "- Include relevant standards or compliance requirements\n\n"
    "Leave as None if no specific guidance is needed beyond the instruction."
)

context_field_description = (
    "Supply essential background information and current state data required for "
    "task execution.\n\n"
    "**Include relevant details about:**\n"
    "1. Environmental conditions\n"
    "2. Historical context\n"
    "3. Related systems or processes\n"
    "4. Previous outcomes or decisions\n\n"
    "**Context should:**\n"
    "- Be directly relevant to the task\n"
    "- Provide necessary background without excess detail\n"
    "- Include any dependencies or prerequisites\n"
    "- Specify the current state of the system\n\n"
    "Set to None if no additional context is required."
)


# Example structures for each field to demonstrate proper formatting
instruction_examples: list[JsonValue] = [
    "Analyze the dataset 'sales_2023.csv' and identify revenue trends",
    "Create a Python function to process customer feedback data",
    {
        "task": "data_analysis",
        "target": "sales_performance",
        "scope": ["revenue", "growth", "seasonality"],
    },
]

guidance_examples: list[JsonValue] = [
    "Use statistical methods for trend analysis",
    "Optimize for readability and maintainability",
    {
        "methods": ["regression", "time_series"],
        "constraints": {"memory": "2GB", "time": "5min"},
    },
]

context_examples: list[JsonValue] = [
    "Previous analysis showed seasonal patterns",
    {
        "prior_results": {"accuracy": 0.95},
        "system_state": "production",
        "dependencies": ["numpy", "pandas"],
    },
]
