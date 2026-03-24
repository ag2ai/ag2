# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Start the AG2 debug server.

Usage::

    python -m autogen.beta.debug
    python -m autogen.beta.debug --host 0.0.0.0 --port 9000
"""

import argparse

from .server import run_debug_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the AG2 debug server")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to (default: 8765)")
    args = parser.parse_args()

    print(f"Starting AG2 debug server at http://{args.host}:{args.port}")
    run_debug_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
