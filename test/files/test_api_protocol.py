# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import inspect

from ag2 import FilesAPI
from ag2.files.protocol import FilesClient


class TestFilesAPIAgainstTheClientProtocol:
    """`FilesAPI` is a facade over a `FilesClient`, not one itself."""

    def test_it_still_satisfies_the_runtime_checkable_protocol(self) -> None:
        """The protocol is structural, so dropping the base changes nothing here."""
        assert isinstance(FilesAPI.__new__(FilesAPI), FilesClient)

    def test_its_upload_is_deliberately_not_the_protocol_s(self) -> None:
        """Keyword-only, takes a path, and forwards provider options."""
        params = inspect.signature(FilesAPI.upload).parameters
        assert "path" in params
        assert params["data"].kind is inspect.Parameter.KEYWORD_ONLY
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

        protocol_params = inspect.signature(FilesClient.upload).parameters
        assert "path" not in protocol_params
        assert protocol_params["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
