# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A Z.AI option is checked against what the SDK's `Completions.create` takes.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite.
"""

from ag2.config import ZAIConfig

ZAIConfig(model="glm-5.2", tool_choice="auto", extra_body={"future_param": 1})

# The SDK types `tool_choice` as a string; an OpenAI-style object is refused before it reaches the API.
bad_choice = {"type": "function"}
# fmt: off
ZAIConfig(model="glm-5.2", tool_choice=bad_choice)  # E: Argument "tool_choice" to "ZAIConfig" has incompatible type "dict[str, str]"; expected "str | None"  [arg-type]
# fmt: on
