# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_additional_dependency, missing_optional_dependency

from .observer import TTSObserver
from .protocols import AudioPlayer, TTSConfig

try:
    from .opendevice_player import OpenDevicePlayer
except ImportError as e:
    OpenDevicePlayer = missing_additional_dependency("OpenDevicePlayer", "sounddevice[numpy]", e)  # type: ignore[misc]

try:
    from .openai_tts import OpenAITTSConfig
except ImportError as e:
    OpenAITTSConfig = missing_optional_dependency("OpenAITTSConfig", "openai", e)  # type: ignore[misc]


__all__ = (
    "AudioPlayer",
    "OpenAITTSConfig",
    "OpenDevicePlayer",
    "TTSConfig",
    "TTSObserver",
)
