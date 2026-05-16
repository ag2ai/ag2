# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch


def test_main_default_args() -> None:
    """main() with no CLI args should call run_debug_server with defaults."""
    with (
        patch("autogen.beta.debug.__main__.run_debug_server") as mock_run,
        patch("sys.argv", ["__main__"]),
    ):
        from autogen.beta.debug.__main__ import main

        main()
        mock_run.assert_called_once_with(host="localhost", port=8765)


def test_main_custom_args() -> None:
    """main() should parse --host and --port flags."""
    with (
        patch("autogen.beta.debug.__main__.run_debug_server") as mock_run,
        patch("sys.argv", ["__main__", "--host", "0.0.0.0", "--port", "9000"]),
    ):
        from autogen.beta.debug.__main__ import main

        main()
        mock_run.assert_called_once_with(host="0.0.0.0", port=9000)
