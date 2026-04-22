# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from google.oauth2 import service_account

from autogen.beta.config import GeminiConfig, VertexAIConfig
from autogen.beta.config.gemini import GeminiClient


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", api_key="key", temperature=0.2, streaming=False)

    copied = config.copy(model="gemini-2.5-flash", temperature=0.8, streaming=True, api_key=None)

    assert copied.model == "gemini-2.5-flash"
    assert copied.temperature == 0.8
    assert copied.streaming is True
    assert copied.api_key is None

    assert config.model == "gemini-2.0-flash"
    assert config.temperature == 0.2
    assert config.streaming is False
    assert config.api_key == "key"


def test_create_returns_gemini_client() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", api_key="test-key")
    client = config.create()

    assert isinstance(client, GeminiClient)


def test_vertex_config_create_returns_gemini_client() -> None:
    config = VertexAIConfig(model="gemini-2.5-pro", project="proj", location="us-central1")
    client = config.create()

    assert isinstance(client, GeminiClient)


def test_defaults() -> None:
    config = GeminiConfig(model="gemini-2.0-flash")
    assert config.streaming is False
    assert config.temperature is None
    assert config.max_output_tokens is None
    assert config.api_key is None


def test_vertex_config_defaults() -> None:
    config = VertexAIConfig(model="gemini-2.5-pro")
    assert config.streaming is False
    assert config.project is None
    assert config.location is None
    assert config.credentials is None


def test_max_output_tokens_can_be_set() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", max_output_tokens=8192)
    assert config.max_output_tokens == 8192


@patch("autogen.beta.config.gemini.gemini_client.genai.Client")
def test_gemini_config_forces_vertexai_false(mock_client) -> None:
    GeminiConfig(model="gemini-2.5-flash", api_key="key").create()

    _, kwargs = mock_client.call_args
    assert kwargs["vertexai"] is False
    assert kwargs["api_key"] == "key"
    assert kwargs["project"] is None
    assert kwargs["location"] is None
    assert kwargs["credentials"] is None


@patch("autogen.beta.config.gemini.gemini_client.genai.Client")
def test_vertex_config_forces_vertexai_true(mock_client) -> None:
    VertexAIConfig(
        model="gemini-2.5-pro",
        project="proj",
        location="us-central1",
    ).create()

    _, kwargs = mock_client.call_args
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "proj"
    assert kwargs["location"] == "us-central1"
    assert kwargs["api_key"] is None


@patch("autogen.beta.config.gemini.gemini_client.genai.Client")
@patch("autogen.beta.config.gemini.gemini_client.service_account.Credentials.from_service_account_file")
def test_credentials_string_loads_service_account_file(mock_from_file, mock_client) -> None:
    loaded = MagicMock(spec=service_account.Credentials)
    mock_from_file.return_value = loaded

    VertexAIConfig(
        model="gemini-2.5-flash",
        project="proj",
        location="us-central1",
        credentials="/fake/key.json",
    ).create()

    mock_from_file.assert_called_once_with(
        "/fake/key.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    _, kwargs = mock_client.call_args
    assert kwargs["credentials"] is loaded


@patch("autogen.beta.config.gemini.gemini_client.genai.Client")
@patch("autogen.beta.config.gemini.gemini_client.service_account.Credentials.from_service_account_file")
def test_credentials_object_passed_through_unchanged(mock_from_file, mock_client) -> None:
    creds = MagicMock(spec=service_account.Credentials)

    VertexAIConfig(
        model="gemini-2.5-flash",
        project="proj",
        location="us-central1",
        credentials=creds,
    ).create()

    mock_from_file.assert_not_called()
    _, kwargs = mock_client.call_args
    assert kwargs["credentials"] is creds


@patch("autogen.beta.config.gemini.gemini_client.genai.Client")
@patch("autogen.beta.config.gemini.gemini_client.service_account.Credentials.from_service_account_file")
def test_credentials_none_passes_through(mock_from_file, mock_client) -> None:
    VertexAIConfig(model="gemini-2.5-flash", project="proj", location="us-central1").create()

    mock_from_file.assert_not_called()
    _, kwargs = mock_client.call_args
    assert kwargs["credentials"] is None
