# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The A2UI section of the system prompt: response format, message types, components, rules, actions, schema."""

import json

from .._types import JsonObject, JsonSchema, JsonValue
from .._versions import A2UIProtocol
from ..actions import A2UIEventAction
from ..constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG
from ._assembly import _AgentCatalogs
from ._json import _as_object, _pointer_key
from ._spec import _VendoredSpec, _instructions_of


def _component_description(comp_def: JsonSchema, envelope: frozenset[str]) -> str:
    """A custom component's prompt line: its ``description``, else the properties it adds to ``envelope``."""
    raw_desc = comp_def.get("description")
    if isinstance(raw_desc, str) and raw_desc:
        return raw_desc

    props: set[str] = set()
    all_of = comp_def.get("allOf")
    if isinstance(all_of, list):
        for item in all_of:
            item_props = _as_object(item.get("properties")) if isinstance(item, dict) else {}
            props.update(k for k in item_props if k not in envelope)
    props.update(k for k in _as_object(comp_def.get("properties")) if k not in envelope)
    return f"Properties: {', '.join(sorted(props))}" if props else ""


_BASIC_COMPONENT_REF_PREFIX = "basic_catalog.json#/components/"
"""How a catalog used in place of the basic one (before v1.0) reaches a basic component."""


def _example_component_types(messages: JsonValue) -> set[str]:
    """The ``component`` types the prompt example's messages use."""
    types: set[str] = set()
    for message in messages if isinstance(messages, list) else ():
        for payload in _as_object(message).values():
            components = _as_object(payload).get("components")
            for component in components if isinstance(components, list) else ():
                kind = _as_object(component).get("component")
                if isinstance(kind, str):
                    types.add(kind)
    return types


def _build_prompt_example(protocol: A2UIProtocol, spec: _VendoredSpec, catalog: _AgentCatalogs) -> str:
    """Build the prompt example JSON string from the version-specific template.

    The example's components come from the basic catalog; a catalog used in place of it may lack them, and then
    the example shows only the messages that carry no components.
    """
    example = spec.prompt_example
    available = _as_object(catalog.surface_catalog.get("components"))
    if isinstance(example, list) and not _example_component_types(example) <= available.keys():
        example = [message for message in example if not _example_component_types([message])]
    raw = json.dumps(example, separators=(",", ": "))
    return _fill_template(raw, protocol, catalog)


def _component_lines(spec: _VendoredSpec, components: JsonObject, envelope: frozenset[str]) -> str:
    """One described line per component; a bare ``$ref`` into the basic catalog is described by its target."""
    basic = _as_object(spec.basic_catalog.get("components"))
    lines = ""
    for name in sorted(components):
        definition = _as_object(components.get(name))
        ref = definition.get("$ref")
        if isinstance(ref, str) and ref.startswith(_BASIC_COMPONENT_REF_PREFIX):
            definition = _as_object(basic.get(_pointer_key(ref.removeprefix(_BASIC_COMPONENT_REF_PREFIX))))
        if desc := _component_description(definition, envelope):
            lines += f"- **{name}**: {desc}\n"
    return lines + "\n" if lines else ""


def _fill_template(text: str, protocol: A2UIProtocol, catalog: _AgentCatalogs) -> str:
    return text.replace("{catalog_id}", catalog.id).replace("{version_string}", protocol.wire_version)


def _prompt_section(
    protocol: A2UIProtocol,
    spec: _VendoredSpec,
    catalog: _AgentCatalogs,
    common_types: JsonSchema,
    custom_catalog_rules: str,
    *,
    include_schema: bool,
    include_rules: bool,
    actions: list[A2UIEventAction] | None,
) -> str:
    """Generate the A2UI portion of the system prompt."""
    protocol_version = protocol.version
    sections: list[str] = []

    example_json = _build_prompt_example(protocol, spec, catalog)

    sections.append(
        f"## A2UI Response Format ({protocol_version})\n\n"
        f"You can generate rich UI responses using the A2UI {protocol_version} protocol. "
        "When you want to display a UI, write your conversational text first, "
        f"then wrap a JSON array of A2UI message objects between the "
        f"`{A2UI_JSON_OPEN_TAG}` and `{A2UI_JSON_CLOSE_TAG}` tags.\n\n"
        "Example response format:\n"
        "```\n"
        "Here is the UI you requested.\n"
        f"{A2UI_JSON_OPEN_TAG}\n"
        f"{example_json}\n"
        f"{A2UI_JSON_CLOSE_TAG}\n"
        "```"
    )

    message_types = _fill_template(spec.prompt_message_types, protocol, catalog)
    sections.append(f"\n\n## A2UI Message Types\n\n{message_types}")

    # The properties every component shares: the ``component`` discriminator and ``ComponentCommon``
    # (``id`` in every version, then v0.9 ``accessibility`` or v1.0 ``@catalogId``, ...).
    common = _as_object(_as_object(common_types.get("$defs")).get("ComponentCommon"))
    envelope = frozenset({"component", *_as_object(common.get("properties"))})
    default_components = _as_object(catalog.surface_catalog.get("components"))
    mixes_catalogs = catalog.mixable and catalog.basic is not None and catalog.custom is not None
    mixed_in = catalog.custom_components if mixes_catalogs else {}
    if default_components or mixed_in:
        comp_section = "\n\n## Available Components\n\n"
        if catalog.custom is None:
            comp_section += f"**Basic catalog:** {', '.join(sorted(default_components))}\n\n"
        elif mixes_catalogs and catalog.custom is not None:
            comp_section += f"**Basic catalog** (the surface default): {', '.join(sorted(default_components))}\n\n"
            if mixed_in:
                custom_id = str(catalog.custom["$id"])
                comp_section += (
                    f"**Custom catalog** (`{custom_id}`): {', '.join(sorted(mixed_in))}\n\n"
                    "These are not in the surface's default catalog: "
                    f'give each of them `"@catalogId": "{custom_id}"`, '
                    f'and each call to one of this catalog\'s functions `"catalogId": "{custom_id}"`.\n\n'
                )
                comp_section += _component_lines(spec, mixed_in, envelope)
        else:
            comp_section += f"**Components:** {', '.join(sorted(default_components))}\n\n"
            comp_section += _component_lines(spec, default_components, envelope)

        comp_section += (
            "Each component uses a flat discriminator format:\n"
            '```json\n{"id": "myId", "component": "Text", "text": "Hello"}\n```\n\n'
            "Components are referenced by `id`. Use `children` (array of ids) for layout components (Column, Row). "
            "Use `child` (single id) for wrapper components (Card, Button)."
        )
        sections.append(comp_section)

    if include_rules:
        rules_parts: list[str] = []
        # A catalog used in place of the basic one replaces its rules too.
        if spec.catalog_rules and catalog.surface_catalog is catalog.basic:
            rules_parts.append(spec.catalog_rules)
        if custom_catalog_rules:
            rules_parts.append(custom_catalog_rules)
        elif catalog.custom is not None and (custom_instructions := _instructions_of(catalog.custom)):
            rules_parts.append(custom_instructions)
        if rules_parts:
            sections.append("\n\n## Component Rules\n\n" + "\n\n".join(rules_parts))

    if actions:
        action_lines = ["\n\n## Available Actions\n"]
        action_lines.append("The following actions can be triggered by buttons in the UI:\n")

        action_lines.append("### Server Events\n")
        for a in actions:
            desc = f": {a.description}" if a.description else ""
            action_lines.append(f"- `{a.name}`{desc}")
            if a.example_context:
                action_lines.append(f"  Context: {json.dumps(a.example_context)}")
        action_lines.append(
            "\nTo create a button that triggers a server event:\n"
            "```json\n"
            '{"id": "action_btn", "component": "Button", "child": "action_btn_text", '
            '"action": {"event": {"name": "<action_name>", "context": {...}}}},\n'
            '{"id": "action_btn_text", "component": "Text", "text": "Button Label"}\n'
            "```"
        )

        sections.append("\n".join(action_lines))

    if include_schema:
        schema_str = json.dumps(spec.server_to_client, indent=2)
        sections.append(f"\n\n## A2UI Message Schema ({protocol_version})\n\n```json\n{schema_str}\n```")

    return "\n".join(sections)
