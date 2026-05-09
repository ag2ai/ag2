# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency, missing_optional_dependency

from .realtime import LiveAgent

try:
    from .sound_device import Player as SoundDevicePlayer
    from .sound_device import Recorder as SoundDeviceRecorder
except ImportError as e:
    SoundDevicePlayer = missing_additional_dependency("SoundDevicePlayer", "sounddevice[numpy]", e)  # type: ignore[misc]
    SoundDeviceRecorder = missing_additional_dependency("SoundDeviceRecorder", "sounddevice[numpy]", e)  # type: ignore[misc]

try:
    from .openai import RealTimeConfig as OpenAIRealTimeConfig
except ImportError as e:
    OpenAIRealTimeConfig = missing_optional_dependency("RealTimeConfig", "openai", e)  # type: ignore[misc]

try:
    from .gemini import RealTimeConfig as GeminiRealTimeConfig
except ImportError as e:
    GeminiRealTimeConfig = missing_optional_dependency("RealTimeConfig", "gemini", e)  # type: ignore[misc]


__all__ = (
    "GeminiRealTimeConfig",
    "LiveAgent",
    "OpenAIRealTimeConfig",
    "SoundDevicePlayer",
    "SoundDeviceRecorder",
)
