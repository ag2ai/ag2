# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Every provider's response-schema mapper takes a schema of any result type.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. The fixture
carries no expectations: a clean run is the assertion.
"""

from pydantic import BaseModel

from ag2.config.anthropic import mappers as anthropic
from ag2.config.bedrock import mappers as bedrock
from ag2.config.dashscope import mappers as dashscope
from ag2.config.gemini import mappers as gemini
from ag2.config.mistral import mappers as mistral
from ag2.config.ollama import mappers as ollama
from ag2.config.openai import mappers as openai
from ag2.config.xai import mappers as xai
from ag2.config.zai import mappers as zai
from ag2.response import ResponseSchema


class User(BaseModel):
    name: str


# `ResponseProto`'s type parameter defaults to `str`, so a bare annotation took only a
# `str` schema; a mapper reads the JSON schema, never the result type.
schema = ResponseSchema(User)

anthropic.response_proto_to_output_config(schema)
bedrock.response_proto_to_output_config(schema)
dashscope.response_proto_to_format(schema)
gemini.response_proto_to_config(schema)
mistral.response_proto_to_format(schema)
ollama.response_proto_to_format(schema)
openai.response_proto_to_schema(schema)
openai.response_proto_to_text_config(schema)
xai.response_proto_to_format(schema)
zai.response_proto_to_format(schema)
zai.schema_instruction(schema)
