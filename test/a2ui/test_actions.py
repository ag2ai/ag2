# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.a2ui import A2UIAction, a2ui_action
from ag2.a2ui.actions import A2UIEventAction, collect_action_declarations, collect_server_actions


class TestDecorator:
    def test_bare_decorator_produces_action(self) -> None:
        @a2ui_action
        def add_to_basket(good_id: str) -> dict:
            """Add to cart."""
            return {"ok": True}

        assert isinstance(add_to_basket, A2UIAction)
        action = add_to_basket.action
        assert isinstance(action, A2UIEventAction)
        assert action.name == "add_to_basket"
        assert action.description == "Add to cart."
        assert action.action_type == "event"

    def test_call_with_overrides(self) -> None:
        @a2ui_action(name="add", description="Add an item")
        def add_to_basket(good_id: str) -> dict:
            return {"ok": True}

        assert add_to_basket.action.name == "add"
        assert add_to_basket.action.description == "Add an item"


class TestExampleContext:
    def test_auto_derived_from_schema(self) -> None:
        @a2ui_action
        def add(good_id: str, qty: int) -> dict:
            return {"ok": True}

        # Declared to the LLM so it can render the button with the right context.
        assert add.action.example_context == {"good_id": "<string>", "qty": "<integer>"}

    def test_mixed_scalar_types(self) -> None:
        @a2ui_action
        def mixed(name: str, count: int, ok: bool) -> str:
            return name

        assert mixed.action.example_context == {
            "name": "<string>",
            "count": "<integer>",
            "ok": "<boolean>",
        }

    def test_optional_and_containers(self) -> None:
        @a2ui_action
        def complex_args(maybe: str | None, tags: list[str], meta: dict[str, str]) -> str:
            return maybe or ""

        # Optional[str] (anyOf with null) resolves to the non-null branch;
        # containers get fresh empty placeholders.
        assert complex_args.action.example_context == {
            "maybe": "<string>",
            "tags": [],
            "meta": {},
        }

    def test_explicit_override_wins(self) -> None:
        @a2ui_action(example_context={"good_id": "SKU-123"})
        def add(good_id: str) -> dict:
            return {"ok": True}

        assert add.action.example_context == {"good_id": "SKU-123"}

    def test_no_params_yields_empty(self) -> None:
        @a2ui_action
        def refresh() -> str:
            return "ok"

        assert refresh.action.example_context == {}


class TestCollectors:
    def test_collect_declarations_and_actions(self) -> None:
        @a2ui_action(description="Add to cart")
        def add_to_basket(good_id: str) -> dict:
            return {"ok": True}

        @a2ui_action
        def remove(good_id: str) -> dict:
            return {"ok": True}

        objs = [add_to_basket, remove]

        assert {a.name for a in collect_action_declarations(objs)} == {"add_to_basket", "remove"}
        # The collector maps name → the A2UIAction itself (which owns its execution).
        assert collect_server_actions(objs) == {"add_to_basket": add_to_basket, "remove": remove}

    def test_collectors_ignore_non_actions(self) -> None:
        def plain() -> None: ...

        assert collect_action_declarations([plain]) == ()
        assert collect_server_actions([plain]) == {}
