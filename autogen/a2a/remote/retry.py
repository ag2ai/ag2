# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import time
from types import TracebackType
from typing import Protocol

import anyio


class RetryPolicyManager(Protocol):
    def __enter__(self) -> None:
        pass

    async def __aenter__(self) -> None:
        pass

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None | bool:
        pass

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None | bool:
        pass


class RetryPolicy(Protocol):
    def __call__(self) -> RetryPolicyManager: ...


class SleepRetryPolicy(RetryPolicy):
    def __init__(self, retry_interval: float = 10.0, retry_count: int = 3) -> None:
        self.retry_interval = retry_interval
        self.retry_count = retry_count

    def __call__(self) -> RetryPolicyManager:
        return _SleepRetryPolicy(self.retry_interval, self.retry_count)


class _SleepRetryPolicy(RetryPolicyManager):
    def __init__(self, retry_interval: float = 10.0, retry_count: int = 3) -> None:
        self.retry_interval = retry_interval
        self.retry_count = retry_count
        self.errors_count = 0

    def __enter__(self) -> None:
        pass

    async def __aenter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None | bool:
        if exc_type is not None:
            self.errors_count += 1
            should_suppress = self.errors_count < self.retry_count
            time.sleep(self.retry_interval)
            return should_suppress
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None | bool:
        if exc_type is not None:
            self.errors_count += 1
            should_suppress = self.errors_count < self.retry_count
            await anyio.sleep(self.retry_interval)
            return should_suppress
        return None


class NoRetryPolicy(RetryPolicyManager):
    def __enter__(self) -> None:
        pass

    async def __aenter__(self) -> None:
        pass

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None | bool:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None | bool:
        pass
