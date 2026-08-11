# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

pytest.importorskip("google.genai")

from google.genai import Client
from google.genai.types import Modality

from ag2.live import gemini
from ag2.tools.final import FunctionDefinition, FunctionToolSchema

pytestmark = pytest.mark.gemini


@pytest.fixture
def gemini_client() -> Client:
    # google-genai's Client requires an api_key at construction time. The
    # tests in this module never open a network connection — they only
    # exercise the in-memory `_build_session` merge — so a dummy key is fine.
    return Client(api_key=os.environ.get("GOOGLE_API_KEY", "test-key-not-used"))


def _build(
    config: gemini.RealTimeConfig,
    *,
    instructions: tuple[str, ...] = (),
    tools: tuple[FunctionToolSchema, ...] = (),
) -> dict:
    return dict(config._build_session(instructions=instructions, tools=tools))


class TestModalities:
    def test_default_audio_output(self, gemini_client: Client) -> None:
        payload = _build(gemini.RealTimeConfig("gemini-2.0-flash-live-001", client=gemini_client))
        assert payload == {
            "response_modalities": [Modality.AUDIO],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}},
            "output_audio_transcription": {},
        }

    def test_text_output_drops_audio_config(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                output=gemini.TextOutput(),
                client=gemini_client,
            )
        )
        assert payload == {"response_modalities": [Modality.TEXT]}

    def test_config_override_replaces_response_modalities(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                config={"response_modalities": ["TEXT"]},
                client=gemini_client,
            )
        )
        assert payload["response_modalities"] == ["TEXT"]


class TestAudioOutput:
    def test_voice_only(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                output=gemini.AudioOutput(voice="Aoede"),
                client=gemini_client,
            )
        )
        assert payload["speech_config"] == {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}},
        }

    def test_voice_and_language(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                output=gemini.AudioOutput(voice="Charon", language_code="en-US"),
                client=gemini_client,
            )
        )
        assert payload["speech_config"] == {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Charon"}},
            "language_code": "en-US",
        }


class TestInputConfig:
    def test_no_user_transcription_by_default(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                input=gemini.InputConfig(),
                client=gemini_client,
            )
        )
        assert "input_audio_transcription" not in payload

    def test_user_transcription_opt_in(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                input=gemini.InputConfig(transcribe=True),
                client=gemini_client,
            )
        )
        assert payload["input_audio_transcription"] == {}

    def test_transcription_languages_propagate(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                input=gemini.InputConfig(transcribe=True, transcription_languages=["en-US", "fr-FR"]),
                client=gemini_client,
            )
        )
        assert payload["input_audio_transcription"] == {"language_codes": ["en-US", "fr-FR"]}

    def test_realtime_input_knobs_propagate(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                input=gemini.InputConfig(
                    automatic_activity_detection={"silence_duration_ms": 500},
                ),
                client=gemini_client,
            )
        )
        assert payload["realtime_input_config"] == {
            "automatic_activity_detection": {"silence_duration_ms": 500},
        }


class TestPromotedKwargs:
    def test_temperature_and_max_output_tokens(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                temperature=0.7,
                max_output_tokens=1024,
                client=gemini_client,
            )
        )
        assert payload["temperature"] == 0.7
        assert payload["max_output_tokens"] == 1024


class TestInstructions:
    def test_instructions_joined(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig("gemini-2.0-flash-live-001", client=gemini_client),
            instructions=("be helpful", "stay terse"),
        )
        assert payload["system_instruction"] == "be helpful\nstay terse"

    def test_no_system_instruction_when_empty(self, gemini_client: Client) -> None:
        payload = _build(gemini.RealTimeConfig("gemini-2.0-flash-live-001", client=gemini_client))
        assert "system_instruction" not in payload

    def test_config_overrides_instructions(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                config={"system_instruction": "raw override"},
                client=gemini_client,
            ),
            instructions=("from prompt",),
        )
        assert payload["system_instruction"] == "raw override"


class TestTools:
    def test_function_tool_serialized(self, gemini_client: Client) -> None:
        schema = FunctionToolSchema(
            function=FunctionDefinition(
                name="sum_numbers",
                description="Sum two integers",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
            )
        )
        payload = _build(
            gemini.RealTimeConfig("gemini-2.0-flash-live-001", client=gemini_client),
            tools=(schema,),
        )
        assert payload["tools"] == [
            {
                "function_declarations": [
                    {
                        "name": "sum_numbers",
                        "description": "Sum two integers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                        },
                    }
                ]
            }
        ]


class TestMergeOrder:
    def test_config_overrides_typed_config(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                temperature=0.2,
                config={"temperature": 0.99},
                client=gemini_client,
            )
        )
        assert payload["temperature"] == 0.99

    def test_config_extends_with_unrelated_keys(self, gemini_client: Client) -> None:
        payload = _build(
            gemini.RealTimeConfig(
                "gemini-2.0-flash-live-001",
                temperature=0.5,
                config={"seed": 42},
                client=gemini_client,
            )
        )
        assert payload["temperature"] == 0.5
        assert payload["seed"] == 42
