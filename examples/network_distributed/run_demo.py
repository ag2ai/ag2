# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the whole distributed network locally as three separate processes.

Spawns ``hub_server``, ``responder``, and ``initiator`` as independent
OS subprocesses on loopback (distinct PIDs, a real TCP/WebSocket port —
no in-process shortcuts), then confirms the initiator's question reached
the responder and the answer came back across the process boundary.

    python -m examples.network_distributed.run_demo
    DEMO_PROVIDER=openai python -m examples.network_distributed.run_demo
    DEMO_RESPONDER_PROVIDER=gemini DEMO_INITIATOR_PROVIDER=anthropic \\
        python -m examples.network_distributed.run_demo

The same scripts run unchanged across real servers — see README.md.
Requires the API key(s) for whichever provider(s) you select.
"""

import contextlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG = "examples.network_distributed"


def _spawn(module: str, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-u", "-m", f"{_PKG}.{module}", *args],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _wait_for(proc: subprocess.Popen[str], pattern: str, *, label: str, timeout: float) -> re.Match[str]:
    """Read ``proc`` stdout (echoing it) until ``pattern`` matches."""
    rx = re.compile(pattern)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline() if proc.stdout is not None else ""
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(f"{label} exited early (rc={proc.returncode})")
            continue
        print(line.rstrip())
        match = rx.search(line)
        if match is not None:
            return match
    raise TimeoutError(f"{label}: never saw /{pattern}/ within {timeout}s")


def main() -> None:
    responder_provider = os.getenv("DEMO_RESPONDER_PROVIDER", os.getenv("DEMO_PROVIDER", "anthropic"))
    initiator_provider = os.getenv("DEMO_INITIATOR_PROVIDER", os.getenv("DEMO_PROVIDER", "anthropic"))

    hub = _spawn("hub_server", "--port", "0")
    procs = [hub]
    try:
        url = _wait_for(hub, r"listening on (ws://\S+)", label="hub", timeout=20.0).group(1)

        responder = _spawn("responder", "--url", url, "--name", "bob", "--provider", responder_provider)
        procs.append(responder)
        _wait_for(responder, r"registered as", label="responder", timeout=30.0)

        initiator = _spawn(
            "initiator",
            "--url",
            url,
            "--name",
            "alice",
            "--provider",
            initiator_provider,
            "--target",
            "bob",
            "--ask",
            "What is 12 times 11? Reply with just the integer.",
        )
        procs.append(initiator)
        _wait_for(initiator, r"RESULT-OK", label="initiator", timeout=120.0)

        pids = {p.pid for p in procs}
        print()
        print(f"[demo] SUCCESS — request/reply crossed {len(pids)} separate OS processes over a real WebSocket")
        print(f"[demo] pids: hub={hub.pid} bob(responder)={responder.pid} alice(initiator)={initiator.pid}")
        print(f"[demo] providers: initiator={initiator_provider} responder={responder_provider}")
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)


if __name__ == "__main__":
    main()
