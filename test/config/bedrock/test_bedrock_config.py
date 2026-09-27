# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError
from dirty_equals import IsPartialDict, IsStr

from ag2.config import BedrockConfig
from ag2.config.bedrock import BedrockClient
from test.config.bedrock._helpers import FakeBedrock, ask


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
class TestRequestFields:
    async def test_inference_params(self, bedrock: FakeBedrock) -> None:
        await ask(bedrock.config(max_tokens=512, temperature=0.5, top_p=0.9, stop_sequences=["STOP"]))

        assert bedrock.body == IsPartialDict({
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.5, "topP": 0.9, "stopSequences": ["STOP"]},
        })

    async def test_unset_options_send_only_messages(self, bedrock: FakeBedrock) -> None:
        await ask(bedrock.config())

        assert bedrock.body == {"messages": [{"role": "user", "content": [{"text": "hello"}]}]}

    async def test_additional_request_fields(self, bedrock: FakeBedrock) -> None:
        await ask(
            bedrock.config(
                additional_model_request_fields={"top_k": 40},
                guardrail_config={"guardrailIdentifier": "g1", "guardrailVersion": "1"},
                performance_config={"latency": "optimized"},
                request_metadata={"team": "research"},
            )
        )

        assert bedrock.body == IsPartialDict({
            "additionalModelRequestFields": {"top_k": 40},
            "guardrailConfig": {"guardrailIdentifier": "g1", "guardrailVersion": "1"},
            "performanceConfig": {"latency": "optimized"},
            "requestMetadata": {"team": "research"},
        })


@pytest.mark.asyncio
async def test_explicit_keys_and_region_sign_the_request(bedrock: FakeBedrock) -> None:
    await ask(bedrock.config(aws_session_token="session-token", region_name="eu-west-1"))

    [request] = bedrock.requests
    assert dict(request.headers) == IsPartialDict({
        "Authorization": IsStr(regex=r"AWS4-HMAC-SHA256 Credential=AKIDTEST/\d{8}/eu-west-1/bedrock/aws4_request, .+"),
        "X-Amz-Security-Token": "session-token",
    })


@pytest.mark.asyncio
class TestConnection:
    async def test_zero_max_retries_makes_a_single_attempt(self, bedrock: FakeBedrock) -> None:
        bedrock.status = 500

        with pytest.raises(ClientError):
            # Zero retries rather than one: a retry would add botocore's backoff sleep, and the
            # outer bound keeps a dropped setting from costing botocore's default retry budget
            await asyncio.wait_for(ask(bedrock.config(max_retries=0)), 2)

        assert len(bedrock.requests) == 1

    async def test_timeout_bounds_the_read(self, bedrock: FakeBedrock) -> None:
        bedrock.hang = True

        with pytest.raises(ReadTimeoutError):
            # The outer bound keeps a regressed timeout from costing botocore's 60s default
            await asyncio.wait_for(ask(bedrock.config(timeout=0.05, max_retries=0)), 2)
