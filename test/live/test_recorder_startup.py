# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

pytest.importorskip("numpy")
sd = pytest.importorskip("sounddevice")

from ag2.live import SoundDeviceRecorder


@pytest.mark.asyncio
async def test_failed_recorder_start_leaves_no_background_task() -> None:
    recorder = SoundDeviceRecorder(channels=99)
    tasks = asyncio.all_tasks()

    with pytest.raises(sd.PortAudioError):
        async with recorder:
            pytest.fail("an impossible input configuration cannot record")

    assert asyncio.all_tasks() == tasks
