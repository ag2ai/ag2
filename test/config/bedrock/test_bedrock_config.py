# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2.config import BedrockConfig
from ag2.config.bedrock import BedrockClient
from ag2.events import ModelRequest, TextInput
from test.config.bedrock._helpers import FakeBedrockRuntime, StubSession, make_call_context


async def _ask(client: BedrockClient) -> None:
    await client(
        messages=[ModelRequest([TextInput("hello")])],
        context=make_call_context(),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )


def test_defaults() -> None:
    config = BedrockConfig(model="m1")

    assert config.streaming is False
    assert config.region_name is None
    assert config.profile_name is None
    assert config.max_tokens is None
    assert config.temperature is None
    assert config.timeout is None
    assert config.max_retries is None


def test_copy_without_overrides_returns_equal_new_instance() -> None:
    config = BedrockConfig(model="m1", region_name="us-east-1")

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = BedrockConfig(model="m1", temperature=0.1)

    copied = config.copy(temperature=0.9, streaming=True)

    assert copied.temperature == 0.9
    assert copied.streaming is True
    assert config.temperature == 0.1
    assert config.streaming is False


def test_create_returns_bedrock_client() -> None:
    config = BedrockConfig(model="m1", region_name="us-east-1")

    client = config.create()

    assert isinstance(client, BedrockClient)


@pytest.mark.asyncio
async def test_inference_params_reach_converse() -> None:
    fake = FakeBedrockRuntime()
    config = BedrockConfig(
        model="m1",
        max_tokens=512,
        temperature=0.5,
        top_p=0.9,
        stop_sequences=["STOP"],
        session=StubSession(fake),
    )

    await _ask(config.create())

    assert fake.converse_kwargs["inferenceConfig"] == {
        "maxTokens": 512,
        "temperature": 0.5,
        "topP": 0.9,
        "stopSequences": ["STOP"],
    }


@pytest.mark.asyncio
async def test_unset_inference_params_omit_inference_config() -> None:
    fake = FakeBedrockRuntime()
    config = BedrockConfig(model="m1", session=StubSession(fake))

    await _ask(config.create())

    assert "inferenceConfig" not in fake.converse_kwargs


@pytest.mark.asyncio
async def test_additional_request_fields_reach_converse() -> None:
    fake = FakeBedrockRuntime()
    config = BedrockConfig(
        model="m1",
        additional_model_request_fields={"top_k": 40},
        guardrail_config={"guardrailIdentifier": "g1", "guardrailVersion": "1"},
        performance_config={"latency": "optimized"},
        request_metadata={"team": "research"},
        session=StubSession(fake),
    )

    await _ask(config.create())

    assert fake.converse_kwargs["additionalModelRequestFields"] == {"top_k": 40}
    assert fake.converse_kwargs["guardrailConfig"] == {"guardrailIdentifier": "g1", "guardrailVersion": "1"}
    assert fake.converse_kwargs["performanceConfig"] == {"latency": "optimized"}
    assert fake.converse_kwargs["requestMetadata"] == {"team": "research"}


@pytest.mark.asyncio
async def test_connection_params_reach_client_creation() -> None:
    fake = FakeBedrockRuntime()
    session = StubSession(fake)
    config = BedrockConfig(
        model="m1",
        endpoint_url="http://localhost:9000",
        timeout=30,
        max_retries=5,
        session=session,
    )

    await _ask(config.create())

    service_name, client_kwargs = session.client_args
    assert service_name == "bedrock-runtime"
    assert client_kwargs["endpoint_url"] == "http://localhost:9000"
    botocore_config = client_kwargs["config"]
    assert botocore_config.connect_timeout == 30
    assert botocore_config.read_timeout == 30
    assert botocore_config.retries == {"max_attempts": 5, "mode": "standard"}
