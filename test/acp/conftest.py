# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the ACP tests."""

import contextlib
import ipaddress
import socket

import pytest


@pytest.fixture(scope="session")
def routable_host() -> str:
    """A non-loopback address this host can actually bind, for gateway tests.

    Exercising the advertised-address path needs an address the gateway's
    loopback default would *not* have used and that the test can still dial;
    which one that is depends on the machine, so it is discovered rather than
    hardcoded (127.0.0.2 is a loopback alias on Linux but not on macOS).
    """
    for candidate in (_outbound_address, socket.gethostname):
        try:
            host = candidate()
        except OSError:  # pragma: no cover - depends on the host's networking
            continue
        with contextlib.suppress(ValueError, OSError), contextlib.closing(socket.socket()) as probe:
            resolved = socket.gethostbyname(host)
            if ipaddress.ip_address(resolved).is_loopback:
                continue
            probe.bind((resolved, 0))
            return resolved
    pytest.skip("no bindable non-loopback address on this host")  # pragma: no cover - host-dependent


def _outbound_address() -> str:
    """This host's address on the interface a default route would leave by.

    A UDP socket is "connected" to a documentation address that is never routed;
    no packet is sent, so this works offline as long as a default route exists.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as probe:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, reserved for documentation
        return str(probe.getsockname()[0])
