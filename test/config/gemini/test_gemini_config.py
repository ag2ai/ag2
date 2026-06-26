# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from google.genai import types
from google.oauth2 import service_account

from ag2.config import GeminiConfig, VertexAIConfig
from ag2.config.gemini import GeminiClient


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = GeminiConfig(model="gemini-3.5-flash", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = GeminiConfig(model="gemini-3.5-flash", api_key="key", temperature=0.2, streaming=False)

    copied = config.copy(model="gemini-2.5-flash", temperature=0.8, streaming=True, api_key=None)

    assert copied.model == "gemini-2.5-flash"
    assert copied.temperature == 0.8
    assert copied.streaming is True
    assert copied.api_key is None

    assert config.model == "gemini-3.5-flash"
    assert config.temperature == 0.2
    assert config.streaming is False
    assert config.api_key == "key"


def test_create_returns_gemini_client() -> None:
    config = GeminiConfig(model="gemini-3.5-flash", api_key="test-key")
    client = config.create()

    assert isinstance(client, GeminiClient)


def test_vertex_config_create_returns_gemini_client() -> None:
    config = VertexAIConfig(model="gemini-2.5-pro", project="proj", location="us-central1")
    client = config.create()

    assert isinstance(client, GeminiClient)


def test_defaults() -> None:
    config = GeminiConfig(model="gemini-3.5-flash")
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
    config = GeminiConfig(model="gemini-3.5-flash", max_output_tokens=8192)
    assert config.max_output_tokens == 8192


@patch("ag2.config.gemini.gemini_client.genai.Client")
def test_gemini_config_forces_vertexai_false(mock_client) -> None:
    GeminiConfig(model="gemini-2.5-flash", api_key="key").create()

    _, kwargs = mock_client.call_args
    assert kwargs["vertexai"] is False
    assert kwargs["api_key"] == "key"
    assert kwargs["project"] is None
    assert kwargs["location"] is None
    assert kwargs["credentials"] is None


@patch("ag2.config.gemini.gemini_client.genai.Client")
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


@patch("ag2.config.gemini.gemini_client.genai.Client")
@patch("ag2.config.gemini.gemini_client.service_account.Credentials.from_service_account_file")
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


@patch("ag2.config.gemini.gemini_client.genai.Client")
@patch("ag2.config.gemini.gemini_client.service_account.Credentials.from_service_account_file")
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


@patch("ag2.config.gemini.gemini_client.genai.Client")
@patch("ag2.config.gemini.gemini_client.service_account.Credentials.from_service_account_file")
def test_credentials_none_passes_through(mock_from_file, mock_client) -> None:
    VertexAIConfig(model="gemini-2.5-flash", project="proj", location="us-central1").create()

    mock_from_file.assert_not_called()
    _, kwargs = mock_client.call_args
    assert kwargs["credentials"] is None


class TestResponseModalities:
    def test_default_omits_response_modalities(self) -> None:
        config = GeminiConfig(model="gemini-3.1-flash-image")
        assert "response_modalities" not in config._build_create_config()

    def test_response_modalities_passes_through(self) -> None:
        config = GeminiConfig(model="gemini-3.1-flash-image", response_modalities=["TEXT", "IMAGE"])
        assert config._build_create_config()["response_modalities"] == ["TEXT", "IMAGE"]

    def test_copy_overrides_response_modalities(self) -> None:
        config = GeminiConfig(model="gemini-3.1-flash-image")
        copied = config.copy(response_modalities=["TEXT", "IMAGE"])

        assert copied.response_modalities == ["TEXT", "IMAGE"]
        assert config.response_modalities is None

    def test_vertex_response_modalities_passes_through(self) -> None:
        config = VertexAIConfig(
            model="gemini-3.1-flash-image",
            project="proj",
            location="us-central1",
            response_modalities=["TEXT", "IMAGE"],
        )
        assert config._build_create_config()["response_modalities"] == ["TEXT", "IMAGE"]


class TestImageConfig:
    def test_default_omits_image_config(self) -> None:
        config = GeminiConfig(model="gemini-3.1-flash-image")
        assert "image_config" not in config._build_create_config()

    def test_image_config_passes_through(self) -> None:
        image_config = types.ImageConfig(aspect_ratio="16:9", image_size="2K")
        config = GeminiConfig(model="gemini-3.1-flash-image", image_config=image_config)
        assert config._build_create_config()["image_config"] is image_config

    def test_copy_overrides_image_config(self) -> None:
        image_config = types.ImageConfig(aspect_ratio="1:1")
        config = GeminiConfig(model="gemini-3.1-flash-image")
        copied = config.copy(image_config=image_config)

        assert copied.image_config is image_config
        assert config.image_config is None

    def test_vertex_image_config_passes_through(self) -> None:
        image_config = types.ImageConfig(aspect_ratio="16:9", image_size="2K")
        config = VertexAIConfig(
            model="gemini-3.1-flash-image",
            project="proj",
            location="us-central1",
            image_config=image_config,
        )
        assert config._build_create_config()["image_config"] is image_config


class TestThinkingConfig:
    def test_default_omits_thinking_config(self) -> None:
        config = GeminiConfig(model="gemini-3.1-pro-preview")
        assert "thinking_config" not in config._build_create_config()

    def test_explicit_thinking_config_passes_through(self) -> None:
        thinking = types.ThinkingConfig(thinking_level="low")
        config = GeminiConfig(model="gemini-3.1-pro-preview", thinking_config=thinking)
        assert config._build_create_config()["thinking_config"] is thinking

    def test_thinking_level_shorthand_builds_config(self) -> None:
        config = GeminiConfig(model="gemini-3.1-pro-preview", thinking_level="low")
        built = config._build_create_config()["thinking_config"]
        assert isinstance(built, types.ThinkingConfig)
        assert built.thinking_level == types.ThinkingLevel.LOW
        assert built.thinking_budget is None

    def test_thinking_budget_shorthand_builds_config(self) -> None:
        config = GeminiConfig(model="gemini-2.5-pro", thinking_budget=1024)
        built = config._build_create_config()["thinking_config"]
        assert isinstance(built, types.ThinkingConfig)
        assert built.thinking_budget == 1024
        assert built.thinking_level is None

    def test_thinking_level_and_budget_combined(self) -> None:
        config = GeminiConfig(
            model="gemini-2.5-pro",
            thinking_level="medium",
            thinking_budget=2048,
        )
        built = config._build_create_config()["thinking_config"]
        assert isinstance(built, types.ThinkingConfig)
        assert built.thinking_level == types.ThinkingLevel.MEDIUM
        assert built.thinking_budget == 2048

    def test_explicit_thinking_config_wins_over_shorthand(self) -> None:
        explicit = types.ThinkingConfig(thinking_level="high")
        config = GeminiConfig(
            model="gemini-3.1-pro-preview",
            thinking_config=explicit,
            thinking_level="low",
        )
        assert config._build_create_config()["thinking_config"] is explicit

    def test_vertex_ai_thinking_level_shorthand_builds_config(self) -> None:
        config = VertexAIConfig(
            model="gemini-3.1-pro-preview",
            project="proj",
            location="us-central1",
            thinking_level="low",
        )
        built = config._build_create_config()["thinking_config"]
        assert isinstance(built, types.ThinkingConfig)
        assert built.thinking_level == types.ThinkingLevel.LOW

    def test_vertex_ai_explicit_thinking_config_passes_through(self) -> None:
        thinking = types.ThinkingConfig(thinking_budget=512)
        config = VertexAIConfig(
            model="gemini-2.5-pro",
            project="proj",
            location="us-central1",
            thinking_config=thinking,
        )
        assert config._build_create_config()["thinking_config"] is thinking

    def test_copy_overrides_thinking_level(self) -> None:
        config = GeminiConfig(model="gemini-3.1-pro-preview", thinking_level="low")
        copied = config.copy(thinking_level="high")

        assert copied.thinking_level == "high"
        assert config.thinking_level == "low"
