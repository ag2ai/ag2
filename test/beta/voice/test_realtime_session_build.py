# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.live import AudioConfig, InputConfig, OpenAIRealTimeConfig


def _build(config: OpenAIRealTimeConfig, *, instructions: str | None = None) -> dict:
    # `_build_session` is the public-by-convention entry point of the merge pipeline;
    # exposed via the realtime config and exercised here without opening a network
    # connection.
    return dict(config._build_session(instructions=instructions))


class TestModalities:
    def test_default_audio_only(self) -> None:
        payload = _build(OpenAIRealTimeConfig("gpt-4o-realtime-preview"))
        assert payload == {"modalities": ["audio"]}

    def test_transcribe_text_adds_text_modality(self) -> None:
        payload = _build(OpenAIRealTimeConfig("gpt-4o-realtime-preview", transcribe_text=True))
        assert payload == {"modalities": ["audio", "text"]}

    def test_session_override_replaces_modalities(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                transcribe_text=True,
                session={"modalities": ["text"]},
            )
        )
        assert payload["modalities"] == ["text"]


class TestAudioConfig:
    def test_serializes_set_fields(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                audio=AudioConfig(voice="ash", output_audio_format="pcm16", speed=1.2),
            )
        )
        assert payload == {
            "modalities": ["audio"],
            "voice": "ash",
            "output_audio_format": "pcm16",
            "speed": 1.2,
        }

    def test_skips_none_fields(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                audio=AudioConfig(voice="ash"),
            )
        )
        assert payload == {"modalities": ["audio"], "voice": "ash"}


class TestInputConfig:
    def test_serializes_set_fields(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                input=InputConfig(
                    input_audio_format="pcm16",
                    input_audio_transcription={"model": "whisper-1"},
                    input_audio_noise_reduction={"type": "near_field"},
                    turn_detection={
                        "type": "semantic_vad",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                ),
            )
        )
        assert payload == {
            "modalities": ["audio"],
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
            "input_audio_noise_reduction": {"type": "near_field"},
            "turn_detection": {
                "type": "semantic_vad",
                "create_response": True,
                "interrupt_response": True,
            },
        }


class TestPromotedKwargs:
    def test_temperature_max_tokens_tool_choice_tracing(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                temperature=0.8,
                max_response_output_tokens=4096,
                tool_choice="auto",
                tracing="auto",
            )
        )
        assert payload == {
            "modalities": ["audio"],
            "temperature": 0.8,
            "max_response_output_tokens": 4096,
            "tool_choice": "auto",
            "tracing": "auto",
        }


class TestInstructions:
    def test_instructions_from_protocol_added(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig("gpt-4o-realtime-preview"),
            instructions="be helpful",
        )
        assert payload == {"modalities": ["audio"], "instructions": "be helpful"}

    def test_no_instructions_key_when_none(self) -> None:
        payload = _build(OpenAIRealTimeConfig("gpt-4o-realtime-preview"), instructions=None)
        assert "instructions" not in payload

    def test_session_overrides_instructions(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                session={"instructions": "raw override"},
            ),
            instructions="from prompt",
        )
        assert payload["instructions"] == "raw override"


class TestMergeOrder:
    def test_session_overrides_typed_config(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                audio=AudioConfig(voice="ash"),
                session={"voice": "alloy"},
            )
        )
        assert payload["voice"] == "alloy"

    def test_session_extends_with_unrelated_keys(self) -> None:
        payload = _build(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                audio=AudioConfig(voice="ash"),
                session={"temperature": 0.6},
            )
        )
        assert payload == {
            "modalities": ["audio"],
            "voice": "ash",
            "temperature": 0.6,
        }
