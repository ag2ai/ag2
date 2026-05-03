# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: run a real Docker container via DockerCodeEnvironment.

Requires Docker daemon running. Skipped (exit 2) if not.
"""

import asyncio
import sys

try:
    import docker as _docker_pkg

    _docker_pkg.from_env().ping()
except Exception as e:  # noqa: BLE001 — best-effort daemon check
    print(f"Docker daemon not reachable ({e}) — skipping smoke test")
    sys.exit(2)

from autogen.beta.extensions.docker import DockerCodeEnvironment


async def main() -> int:
    print("→ Creating DockerCodeEnvironment (container is lazy)")
    async with DockerCodeEnvironment(image="python:3.12-slim") as env:
        print(f"  supported languages: {env.supported_languages}")

        print("→ Running python: 2 + 2")
        py = await env.run("print(2 + 2)", "python")
        print(f"  exit_code={py.exit_code} output={py.output!r}")
        assert py.exit_code == 0 and "4" in py.output, "python execution failed"

        print("→ Running bash: echo + uname")
        sh = await env.run("echo hello; uname -a", "bash")
        print(f"  exit_code={sh.exit_code} output={sh.output!r}")
        assert sh.exit_code == 0 and "hello" in sh.output, "bash execution failed"

        print("→ Persistence check: write a file, then read it")
        await env.run("open('/workspace/marker.txt', 'w').write('persisted')", "python")
        rd = await env.run("print(open('/workspace/marker.txt').read())", "python")
        print(f"  exit_code={rd.exit_code} output={rd.output!r}")
        assert rd.exit_code == 0 and "persisted" in rd.output, "persistence failed"

        print("→ Network isolation check: default network=none")
        net = await env.run(
            "import urllib.request; urllib.request.urlopen('https://example.com', timeout=3)",
            "python",
        )
        print(f"  exit_code={net.exit_code} (expected non-zero — network is off)")
        assert net.exit_code != 0, "network was reachable but should have been blocked"

    print("✓ Docker smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
