# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from autogen.a2a.constants import A2UI_MIME_TYPE
from autogen.agents.experimental.a2ui.a2a_helpers import (
    A2UI_EXTENSION_URI,
    create_a2ui_part,
    get_a2ui_agent_extension,
    get_a2ui_datapart,
    is_a2ui_part,
    try_activate_a2ui_extension,
)


class TestA2UIPartHelpers:
    def test_create_a2ui_part_single_dict(self) -> None:
        from a2a.types import DataPart

        data = {"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}
        part = create_a2ui_part(data)

        assert isinstance(part.root, DataPart)  # type: ignore[union-attr]
        # Single dict is stored directly as data
        assert part.root.data == data  # type: ignore[union-attr]
        assert part.root.metadata is not None  # type: ignore[union-attr]
        assert part.root.metadata["mimeType"] == A2UI_MIME_TYPE  # type: ignore[union-attr]

    def test_create_a2ui_part_list(self) -> None:
        from a2a.types import DataPart

        ops = [
            {"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}},
            {"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}},
        ]
        result = create_a2ui_part(ops)

        # Multiple ops return a list of Parts, one per operation
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0].root, DataPart)
        assert result[0].root.data == ops[0]
        assert isinstance(result[1].root, DataPart)
        assert result[1].root.data == ops[1]

    def test_is_a2ui_part_true(self) -> None:
        part = create_a2ui_part({"test": "data"})
        assert is_a2ui_part(part) is True  # type: ignore[arg-type]

    def test_is_a2ui_part_false_wrong_mime(self) -> None:
        from a2a.types import DataPart, Part

        part = Part(root=DataPart(data={"test": "data"}, metadata={"mimeType": "application/json"}))
        assert is_a2ui_part(part) is False

    def test_is_a2ui_part_false_no_metadata(self) -> None:
        from a2a.types import DataPart, Part

        part = Part(root=DataPart(data={"test": "data"}, metadata=None))
        assert is_a2ui_part(part) is False

    def test_is_a2ui_part_false_text_part(self) -> None:
        from a2a.types import Part, TextPart

        part = Part(root=TextPart(text="hello"))
        assert is_a2ui_part(part) is False

    def test_get_a2ui_datapart_present(self) -> None:
        part = create_a2ui_part({"test": "data"})
        datapart = get_a2ui_datapart(part)  # type: ignore[arg-type]
        assert datapart is not None
        assert datapart.data == {"test": "data"}  # type: ignore[union-attr]

    def test_get_a2ui_datapart_absent(self) -> None:
        from a2a.types import Part, TextPart

        part = Part(root=TextPart(text="hello"))
        assert get_a2ui_datapart(part) is None


class TestA2UIExtension:
    def test_extension_uri_is_v09(self) -> None:
        assert "v0.9" in A2UI_EXTENSION_URI

    def test_get_agent_extension_default(self) -> None:
        ext = get_a2ui_agent_extension()
        assert ext.uri == A2UI_EXTENSION_URI
        assert ext.description is not None and "A2UI" in ext.description
        assert ext.params is None

    def test_get_agent_extension_with_inline_catalog(self) -> None:
        ext = get_a2ui_agent_extension(accepts_inline_custom_catalog=True)
        assert ext.params is not None
        assert ext.params["acceptsInlineCustomCatalog"] is True

    def test_try_activate_success(self) -> None:
        class MockContext:
            def __init__(self) -> None:
                self.requested_extensions = {A2UI_EXTENSION_URI}
                self._activated: set[str] = set()

            def add_activated_extension(self, uri: str) -> None:
                self._activated.add(uri)

        ctx = MockContext()
        result = try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert result is True
        assert A2UI_EXTENSION_URI in ctx._activated

    def test_try_activate_not_requested(self) -> None:
        class MockContext:
            def __init__(self) -> None:
                self.requested_extensions: set[str] = set()
                self._activated: set[str] = set()

            def add_activated_extension(self, uri: str) -> None:
                self._activated.add(uri)

        ctx = MockContext()
        result = try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert result is False
        assert len(ctx._activated) == 0

    def test_try_activate_wrong_version_uri(self) -> None:
        """Client requests a different A2UI version — should not activate."""

        class MockContext:
            def __init__(self) -> None:
                # Client requests a hypothetical v1.0, but server only supports v0.9
                self.requested_extensions = {"https://a2ui.org/a2a-extension/a2ui/v1.0"}
                self._activated: set[str] = set()

            def add_activated_extension(self, uri: str) -> None:
                self._activated.add(uri)

        ctx = MockContext()
        result = try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert result is False
        assert len(ctx._activated) == 0

    def test_try_activate_multiple_extensions_only_a2ui_activated(self) -> None:
        """Client requests multiple extensions — only A2UI should be activated."""

        class MockContext:
            def __init__(self) -> None:
                self.requested_extensions = {
                    A2UI_EXTENSION_URI,
                    "https://example.com/some-other-extension/v1.0",
                }
                self._activated: set[str] = set()

            def add_activated_extension(self, uri: str) -> None:
                self._activated.add(uri)

        ctx = MockContext()
        result = try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert result is True
        assert ctx._activated == {A2UI_EXTENSION_URI}

    def test_try_activate_empty_extensions(self) -> None:
        """Client sends no extensions at all."""

        class MockContext:
            def __init__(self) -> None:
                self.requested_extensions: set[str] = set()
                self._activated: set[str] = set()

            def add_activated_extension(self, uri: str) -> None:
                self._activated.add(uri)

        ctx = MockContext()
        result = try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert result is False
