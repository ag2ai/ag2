# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An ACP config is a model config under the checker.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite.
"""

from typing_extensions import assert_type

from ag2 import Agent
from ag2.acp import ACPConfig, ACPRemoteConfig, ClaudeCodeConfig
from ag2.acp.testing import ACPTurn, fake_acp_config
from ag2.config import OpenAIConfig
from ag2.config.config import ModelConfig

# The documented constructions.
Agent("coder", config=ACPConfig(["claude-agent-acp"]))
Agent("coder", config=ClaudeCodeConfig())
Agent("coder", config=ACPRemoteConfig(url="ws://localhost:8000"))
Agent("coder", config=fake_acp_config(ACPTurn()))


def model_of(config: ModelConfig) -> None:
    # `None` is a config naming no model, as an ACP config left on its agent's default does.
    assert_type(config.model, str | None)


# A provider config still always names one.
assert_type(OpenAIConfig(model="gpt-5").model, str)
