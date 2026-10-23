# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

import pytest

pytest.importorskip("numpy")
sd = pytest.importorskip("sounddevice")

from ag2.live import SoundDeviceRecorder


class FailingInputStream:
    def __init__(self, **kwargs: Any) -> None:
        self.closed = False
        self.fail_start = True

    def start(self) -> None:
        if self.fail_start:
            raise sd.PortAudioError("input device unavailable")

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_failed_recorder_start_closes_input_and_leaves_no_background_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FailingInputStream()
    monkeypatch.setattr(sd, "InputStream", lambda **kwargs: device)
    tasks = asyncio.all_tasks()

    with pytest.raises(sd.PortAudioError, match="input device unavailable"):
        async with SoundDeviceRecorder():
            pytest.fail("an unavailable input device cannot record")

    assert device.closed
    assert asyncio.all_tasks() == tasks


@pytest.mark.asyncio
async def test_recorder_can_start_after_a_failed_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    device = FailingInputStream()
    monkeypatch.setattr(sd, "InputStream", lambda **kwargs: device)
    recorder = SoundDeviceRecorder()
    tasks = asyncio.all_tasks()

    with pytest.raises(sd.PortAudioError):
        async with recorder:
            pytest.fail("an unavailable input device cannot record")

    device.fail_start = False
    async with recorder:
        await asyncio.sleep(0)

    assert device.closed
    assert asyncio.all_tasks() == tasks


@pytest.mark.asyncio
async def test_failed_input_construction_leaves_no_background_task(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**kwargs: Any) -> None:
        raise sd.PortAudioError("no input device")

    monkeypatch.setattr(sd, "InputStream", fail)
    tasks = asyncio.all_tasks()

    with pytest.raises(sd.PortAudioError, match="no input device"):
        async with SoundDeviceRecorder():
            pytest.fail("an unavailable input device cannot record")

    assert asyncio.all_tasks() == tasks
