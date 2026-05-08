# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency, missing_optional_dependency

from .observer import AudioPlayerObserver, TTSObserver
from .protocols import AudioPlayer, TTSConfig
from .realtime import LiveAgent, RealtimeConfig
from .stt import STTConfig, VoiceInput

try:
    from .sound_device import Player as SoundDevicePlayer
    from .sound_device import Recorder as SoundDeviceRecorder
except ImportError as e:
    SoundDevicePlayer = missing_additional_dependency("SoundDevicePlayer", "sounddevice[numpy]", e)  # type: ignore[misc]
    SoundDeviceRecorder = missing_additional_dependency("SoundDeviceRecorder", "sounddevice[numpy]", e)  # type: ignore[misc]

try:
    from . import openai
    from .openai import AudioOutput, InputConfig, TextOutput
    from .openai import RealTimeConfig as OpenAIRealTimeConfig
    from .openai import STTConfig as OpenAITranscriber
    from .openai import STTTranslationConfig as OpenAITranslationTranscriber
    from .openai import TTSConfig as OpenAISynthesizer
except ImportError as e:
    openai = missing_optional_dependency("openai", "openai", e)  # type: ignore[misc,assignment]
    AudioOutput = missing_optional_dependency("AudioOutput", "openai", e)  # type: ignore[misc]
    TextOutput = missing_optional_dependency("TextOutput", "openai", e)  # type: ignore[misc]
    InputConfig = missing_optional_dependency("InputConfig", "openai", e)  # type: ignore[misc]
    OpenAIRealTimeConfig = missing_optional_dependency("RealTimeConfig", "openai", e)  # type: ignore[misc]
    OpenAITranscriber = missing_optional_dependency("OpenAITranscriber", "openai", e)  # type: ignore[misc]
    OpenAITranslationTranscriber = missing_optional_dependency("OpenAITranslationTranscriber", "openai", e)  # type: ignore[misc]
    OpenAISynthesizer = missing_optional_dependency("OpenAISynthesizer", "openai", e)  # type: ignore[misc]


__all__ = (
    "AudioOutput",
    "AudioPlayer",
    "AudioPlayerObserver",
    "InputConfig",
    "LiveAgent",
    "OpenAIRealTimeConfig",
    "OpenAISynthesizer",
    "OpenAITranscriber",
    "OpenAITranslationTranscriber",
    "RealtimeConfig",
    "STTConfig",
    "SoundDevicePlayer",
    "SoundDeviceRecorder",
    "TTSConfig",
    "TTSObserver",
    "TextOutput",
    "VoiceInput",
    "openai",
)
