# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

from a2ui.core import Catalog, get_agent_to_renderer_schema_json
from a2ui.core.basic_catalog import BasicCatalog
from a2ui.core.catalog import CatalogApi
from a2ui.core.common import to_protocol_version

from ._types import A2UIVersion, JsonObject, JsonSchema, JsonValue
from .actions import A2UIEventAction
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_SUPPORTED_VERSIONS: tuple[A2UIVersion, ...] = ("v0.9", "v0.9.1", "v1.0")

# v0.9.1 shares v0.9's prompt text: it only widens the ``version`` enum.
_PROMPT_DIR_BY_VERSION: dict[A2UIVersion, str] = {"v0.9": "v0_9", "v0.9.1": "v0_9_1", "v1.0": "v1_0"}


def _as_object(value: "JsonValue | None") -> JsonObject:
    """Narrow a ``JsonValue`` to a JSON object, returning ``{}`` when it is not a dict."""
    return value if isinstance(value, dict) else {}


class A2UISchemaManager:
    """Resolves the A2UI catalog for a protocol version through ``a2ui-core`` and
    builds the system-prompt section that instructs an LLM to emit valid A2UI output.

    A custom catalog (file path or dict) is a self-contained unit: it replaces the
    basic catalog for the surface instead of extending it, exactly as in
    ``a2ui-core``. A catalog that wants ``Column`` next to its own ``Badge`` lists both.
    """

    def __init__(
        self,
        protocol_version: A2UIVersion = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
    ) -> None:
        """Initialize the schema manager.

        Args:
            protocol_version: The A2UI protocol version: "v0.9" (default), "v0.9.1", or "v1.0".
            custom_catalog: A self-contained custom catalog. Can be:
                - A file path (str or PathLike) to a JSON catalog file
                - A dict with the catalog schema directly
                Must include a ``$id`` (or ``catalogId``) field, used as the catalogId in
                A2UI messages.
            custom_catalog_rules: Plain-text rules for the custom catalog components,
                appended to the system prompt.
        """
        if protocol_version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported A2UI protocol version: {protocol_version!r}. "
                f"Supported versions: {', '.join(_SUPPORTED_VERSIONS)}"
            )

        self._protocol_version = protocol_version

        # Raw, unread inputs. File I/O and catalog construction are deferred to
        # ``_ensure_loaded`` so construction stays side-effect-free per the project
        # rule "no side effects in initialization methods".
        self._custom_catalog_input = custom_catalog
        self._custom_catalog_rules = custom_catalog_rules or ""

        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Build the catalog and read the prompt files on first use (idempotent)."""
        if self._loaded:
            return

        prompt_dir = _PROMPTS_DIR / _PROMPT_DIR_BY_VERSION[self._protocol_version]
        self._catalog_rules = _read_text(prompt_dir / "basic_catalog_rules.txt")
        self._prompt_example: JsonValue = json.loads((prompt_dir / "prompt_example.json").read_text(encoding="utf-8"))
        self._prompt_message_types = _read_text(prompt_dir / "prompt_message_types.md")

        self._custom_catalog: JsonSchema | None = self._read_custom_catalog()
        self._catalog: CatalogApi
        if self._custom_catalog is None:
            self._catalog = BasicCatalog(self._protocol_version)
        else:
            self._catalog = Catalog.from_json(
                self._custom_catalog,
                protocol_version=self._protocol_version,
                catalog_id=self._custom_catalog_id(self._custom_catalog),
            )

        self._loaded = True

    def _read_custom_catalog(self) -> JsonSchema | None:
        custom_catalog = self._custom_catalog_input
        if custom_catalog is None:
            return None
        if isinstance(custom_catalog, dict):
            return custom_catalog
        path = Path(custom_catalog)
        try:
            with open(path, encoding="utf-8") as f:
                loaded: JsonSchema = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Custom catalog file not found: {path}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Custom catalog file {path} is not valid JSON: {e}") from e
        return loaded

    @staticmethod
    def _custom_catalog_id(custom_catalog: JsonSchema) -> str:
        catalog_id = custom_catalog.get("$id", custom_catalog.get("catalogId"))
        if catalog_id is None:
            raise ValueError(
                "Custom catalog must include a '$id' field. "
                "This is used as the catalogId in A2UI createSurface messages. "
                'Example: {"$id": "https://mycompany.com/my_catalog.json", ...}'
            )
        if not isinstance(catalog_id, str):
            raise ValueError(
                f"Custom catalog '$id' must be a string, got {type(catalog_id).__name__!r}. "
                "It is used verbatim as the catalogId."
            )
        return catalog_id

    @property
    def protocol_version(self) -> A2UIVersion:
        return self._protocol_version

    @property
    def version_string(self) -> A2UIVersion:
        """The version string used in A2UI messages (e.g., 'v0.9')."""
        return self._protocol_version

    @property
    def catalog(self) -> CatalogApi:
        """The active ``a2ui-core`` catalog: the basic one, or the custom catalog on its own."""
        self._ensure_loaded()
        return self._catalog

    @property
    def catalog_id(self) -> str:
        return self.catalog.id

    @property
    def server_to_client_schema(self) -> JsonSchema:
        """The server→client (agent→renderer) message schema published by ``a2ui-core``."""
        schema: JsonSchema = json.loads(get_agent_to_renderer_schema_json(to_protocol_version(self._protocol_version)))
        return schema

    def _build_prompt_example(self) -> str:
        """Build the prompt example JSON string from the version-specific template."""
        raw = json.dumps(self._prompt_example, separators=(",", ": "))
        return raw.replace("{catalog_id}", self.catalog_id)

    def generate_prompt_section(
        self,
        include_schema: bool = True,
        include_rules: bool = True,
        actions: list[A2UIEventAction] | None = None,
    ) -> str:
        """Generate the A2UI portion of the system prompt."""
        self._ensure_loaded()
        v = self._protocol_version
        sections: list[str] = []

        example_json = self._build_prompt_example()

        sections.append(
            f"## A2UI Response Format ({v})\n\n"
            f"You can generate rich UI responses using the A2UI {v} protocol. "
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

        message_types = self._prompt_message_types.replace("{catalog_id}", self.catalog_id).replace(
            "{version_string}", v
        )
        sections.append(f"\n\n## A2UI Message Types\n\n{message_types}")

        component_names = sorted(self._catalog.components)
        if component_names:
            comp_section = "\n\n## Available Components\n\n"
            if self._custom_catalog is None:
                comp_section += f"**Basic catalog:** {', '.join(component_names)}\n\n"
            else:
                comp_section += f"**Custom catalog components:** {', '.join(component_names)}\n\n"
                custom_comp_defs = _as_object(self._custom_catalog.get("components"))
                for comp_name in component_names:
                    comp_def = _as_object(custom_comp_defs.get(comp_name))
                    desc = self._extract_component_description(comp_def)
                    if desc:
                        comp_section += f"- **{comp_name}**: {desc}\n"
                comp_section += "\n"

            if self._custom_catalog is None:
                example = '{"id": "myId", "component": "Text", "text": "Hello"}'
                reference_rule = (
                    "Components are referenced by `id`. Use `children` (array of ids) for layout components "
                    "(Column, Row). Use `child` (single id) for wrapper components (Card, Button)."
                )
            else:
                example = '{"id": "myId", "component": "<ComponentName>", ...}'
                reference_rule = (
                    "Components are referenced by `id`. Child components are referenced by the ids "
                    "their component's properties ask for."
                )
            comp_section += (
                f"Each component uses a flat discriminator format:\n```json\n{example}\n```\n\n{reference_rule}"
            )
            sections.append(comp_section)

        if include_rules:
            rules_parts: list[str] = []
            # The basic rules describe basic components; a standalone custom catalog does not have them.
            if self._catalog_rules and self._custom_catalog is None:
                rules_parts.append(self._catalog_rules)
            if self._custom_catalog_rules:
                rules_parts.append(self._custom_catalog_rules)
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
            schema_str = json.dumps(self.server_to_client_schema, indent=2)
            sections.append(f"\n\n## A2UI Message Schema ({v})\n\n```json\n{schema_str}\n```")

        return "\n".join(sections)

    @staticmethod
    def _extract_component_description(comp_def: JsonSchema) -> str:
        """Extract a human-readable description of a custom component for the prompt."""
        raw_desc = comp_def.get("description")
        if isinstance(raw_desc, str) and raw_desc:
            return raw_desc

        props: set[str] = set()
        all_of = comp_def.get("allOf")
        if isinstance(all_of, list):
            for item in all_of:
                item_props = _as_object(item.get("properties")) if isinstance(item, dict) else {}
                props.update(k for k in item_props if k not in ("id", "component", "accessibility"))
        own_props = _as_object(comp_def.get("properties"))
        props.update(k for k in own_props if k not in ("id", "component", "accessibility"))

        if props:
            return f"Properties: {', '.join(sorted(props))}"
        return ""


def _read_text(path: Path) -> str:
    """Read a packaged prompt file, returning ``""`` when the version ships none."""
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""
