# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_additional_dependency, missing_optional_dependency

from .cascade import CascadeConfig
from .observer import TTSObserver
from .realtime import LiveAgent
from .turn import SilenceTurnDetector, TurnDetector

# The fallback rebinds a name mypy has bound to a class, which it rejects; it
# sees only the real import. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .elevenlabs import STTConfig as ElevenLabsTranscriber
    from .elevenlabs import StreamingTTSConfig as ElevenLabsStreamingTTSConfig
    from .elevenlabs import TTSConfig as ElevenLabsTTSConfig
    from .gemini import RealTimeConfig as GeminiRealTimeConfig
    from .openai import RealTimeConfig as OpenAIRealTimeConfig
    from .openai import STTConfig as OpenAITranscriber
    from .openai import STTTranslationConfig as OpenAITranslationTranscriber
    from .openai import TTSConfig as OpenAITTSConfig
    from .sound_device import Player as SoundDevicePlayer
    from .sound_device import Recorder as SoundDeviceRecorder
else:
    try:
        from .sound_device import Player as SoundDevicePlayer
        from .sound_device import Recorder as SoundDeviceRecorder
    except ImportError as e:
        SoundDevicePlayer = missing_additional_dependency("SoundDevicePlayer", "sounddevice[numpy]", e)
        SoundDeviceRecorder = missing_additional_dependency("SoundDeviceRecorder", "sounddevice[numpy]", e)

    try:
        from .openai import RealTimeConfig as OpenAIRealTimeConfig
        from .openai import STTConfig as OpenAITranscriber
        from .openai import STTTranslationConfig as OpenAITranslationTranscriber
        from .openai import TTSConfig as OpenAITTSConfig
    except ImportError as e:
        OpenAIRealTimeConfig = missing_optional_dependency("RealTimeConfig", "openai", e)
        OpenAITTSConfig = missing_optional_dependency("TTSConfig", "openai", e)
        OpenAITranscriber = missing_optional_dependency("STTConfig", "openai", e)
        OpenAITranslationTranscriber = missing_optional_dependency("STTTranslationConfig", "openai", e)

    try:
        from .gemini import RealTimeConfig as GeminiRealTimeConfig
    except ImportError as e:
        GeminiRealTimeConfig = missing_optional_dependency("RealTimeConfig", "gemini", e)

    try:
        from .elevenlabs import STTConfig as ElevenLabsTranscriber
        from .elevenlabs import StreamingTTSConfig as ElevenLabsStreamingTTSConfig
        from .elevenlabs import TTSConfig as ElevenLabsTTSConfig
    except ImportError as e:
        ElevenLabsTTSConfig = missing_optional_dependency("TTSConfig", "elevenlabs", e)
        ElevenLabsStreamingTTSConfig = missing_optional_dependency("StreamingTTSConfig", "elevenlabs", e)
        ElevenLabsTranscriber = missing_optional_dependency("STTConfig", "elevenlabs", e)


__all__ = (
    "CascadeConfig",
    "ElevenLabsStreamingTTSConfig",
    "ElevenLabsTTSConfig",
    "ElevenLabsTranscriber",
    "GeminiRealTimeConfig",
    "LiveAgent",
    "OpenAIRealTimeConfig",
    "OpenAITTSConfig",
    "OpenAITranscriber",
    "OpenAITranslationTranscriber",
    "SilenceTurnDetector",
    "SoundDevicePlayer",
    "SoundDeviceRecorder",
    "TTSObserver",
    "TurnDetector",
)
