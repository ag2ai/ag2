# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_additional_dependency, missing_optional_dependency

from .observer import TTSObserver
from .realtime import LiveAgent

try:
    from .sound_device import Player as SoundDevicePlayer
    from .sound_device import Recorder as SoundDeviceRecorder
except ImportError as e:
    SoundDevicePlayer = missing_additional_dependency("SoundDevicePlayer", "sounddevice[numpy]", e)  # type: ignore[misc]
    SoundDeviceRecorder = missing_additional_dependency("SoundDeviceRecorder", "sounddevice[numpy]", e)  # type: ignore[misc]

try:
    from .openai import RealTimeConfig as OpenAIRealTimeConfig
    from .openai import STTConfig as OpenAITranscriber
    from .openai import STTTranslationConfig as OpenAITranslationTranscriber
    from .openai import TTSConfig as OpenAITTSConfig
except ImportError as e:
    OpenAIRealTimeConfig = missing_optional_dependency("RealTimeConfig", "openai", e)  # type: ignore[misc]
    OpenAITTSConfig = missing_optional_dependency("TTSConfig", "openai", e)  # type: ignore[misc]
    OpenAITranscriber = missing_optional_dependency("STTConfig", "openai", e)  # type: ignore[misc]
    OpenAITranslationTranscriber = missing_optional_dependency("STTTranslationConfig", "openai", e)  # type: ignore[misc]

try:
    from .gemini import RealTimeConfig as GeminiRealTimeConfig
except ImportError as e:
    GeminiRealTimeConfig = missing_optional_dependency("RealTimeConfig", "gemini", e)  # type: ignore[misc]


__all__ = (
    "GeminiRealTimeConfig",
    "LiveAgent",
    "OpenAIRealTimeConfig",
    "OpenAITTSConfig",
    "OpenAITranscriber",
    "OpenAITranslationTranscriber",
    "SoundDevicePlayer",
    "SoundDeviceRecorder",
    "TTSObserver",
)
