# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any, TypedDict

from typing_extensions import Unpack

from ag2.config.config import ModelConfig

from .bedrock_client import BedrockClient, CreateOptions


class BedrockConfigOverrides(TypedDict, total=False):
    model: str
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_session_token: str | None
    profile_name: str | None
    region_name: str | None
    endpoint_url: str | None
    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    stop_sequences: list[str] | None
    streaming: bool
    additional_model_request_fields: dict[str, Any] | None
    additional_model_response_field_paths: list[str] | None
    guardrail_config: dict[str, Any] | None
    performance_config: dict[str, Any] | None
    request_metadata: dict[str, str] | None
    timeout: float | None
    max_retries: int | None
    botocore_config: Any | None
    session: Any | None


@dataclass(slots=True)
class BedrockConfig(ModelConfig):
    """Amazon Bedrock model configuration (Converse API).

    Credentials follow boto3's resolution chain: explicit keys, then
    ``profile_name``, then environment variables / shared config files /
    instance roles. ``model`` is a Bedrock model id or inference-profile ARN.
    """

    model: str
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    profile_name: str | None = None
    region_name: str | None = None
    endpoint_url: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    streaming: bool = False
    additional_model_request_fields: dict[str, Any] | None = None
    additional_model_response_field_paths: list[str] | None = None
    guardrail_config: dict[str, Any] | None = None
    performance_config: dict[str, Any] | None = None
    request_metadata: dict[str, str] | None = None
    timeout: float | None = None
    max_retries: int | None = None
    botocore_config: Any | None = None
    session: Any | None = None

    def copy(self, /, **overrides: Unpack[BedrockConfigOverrides]) -> "BedrockConfig":
        return replace(self, **overrides)

    def create(self) -> BedrockClient:
        options = CreateOptions(
            model=self.model,
            stream=self.streaming,
        )

        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.stop_sequences is not None:
            options["stop_sequences"] = self.stop_sequences
        if self.additional_model_request_fields is not None:
            options["additional_model_request_fields"] = self.additional_model_request_fields
        if self.additional_model_response_field_paths is not None:
            options["additional_model_response_field_paths"] = self.additional_model_response_field_paths
        if self.guardrail_config is not None:
            options["guardrail_config"] = self.guardrail_config
        if self.performance_config is not None:
            options["performance_config"] = self.performance_config
        if self.request_metadata is not None:
            options["request_metadata"] = self.request_metadata

        return BedrockClient(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            profile_name=self.profile_name,
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            botocore_config=self.botocore_config,
            session=self.session,
            create_options=options,
        )
