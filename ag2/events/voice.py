# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.events import BaseEvent, Field


class TranscriptionChunkEvent(BaseEvent):
    content: str = Field(kw_only=False)


class TranscriptionCompletedEvent(BaseEvent):
    content: str = Field(kw_only=False)


class SynthesizedAudioEvent(BaseEvent):
    content: bytes = Field(kw_only=False)


class RecordedAudioEvent(BaseEvent):
    content: bytes = Field(kw_only=False)


class AudioPlaybackStartedEvent(BaseEvent):
    """The speaker began playing a reply.

    Emitted by the player, not by whatever synthesized the audio: only the
    player knows when bytes actually reach the speaker, and a session that
    wants to stop listening while it talks has to gate on the sound in the
    room rather than on the moment synthesis finished.
    """


class AudioPlaybackCompletedEvent(BaseEvent):
    """The speaker went quiet — every queued chunk has been played.

    Paired with `AudioPlaybackStartedEvent`. Also emitted after a barge-in
    flush, since dropping the queue ends playback just as surely as draining
    it does.
    """


class AudioInterruptedEvent(BaseEvent):
    """Barge-in: discard audio that was queued for playback but not yet heard.

    Emitted when the user starts talking over a reply. Synthesis is stopped at
    the source, but whatever already reached the player is still about to be
    played — a player that ignores this keeps speaking the abandoned reply.
    """
