# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import TypedDict

from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from ._types import A2UIVersion, JsonObject, JsonSchema, JsonValue  # noqa: F401
from .actions import A2UIAction
from .constants import A2UI_DEFAULT_CATALOG_ID_BY_VERSION, A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG

_VERSIONS_DIR = Path(__file__).parent


class _VersionConfigEntry(TypedDict):
    """Per-version identifiers used to load specs and stamp wire messages."""

    default_catalog_id: str
    schema_base_uri: str
    version_string: A2UIVersion


# Catalog ids / schema base uris are the canonical A2UI identifiers (they
# resolve on a2ui.org and must match what renderers advertise in
# ``supportedCatalogIds``). v0.9.1 is a backward-compatible patch over v0.9: it
# reuses v0.9's schema ``$id``s and catalog, and only widens the ``version``
# enum to accept ``"v0.9.1"`` — hence its identifiers point at ``v0_9``.
_VERSION_CONFIG: dict[A2UIVersion, _VersionConfigEntry] = {
    "v0.9": {
        "default_catalog_id": A2UI_DEFAULT_CATALOG_ID_BY_VERSION["v0.9"],
        "schema_base_uri": "https://a2ui.org/specification/v0_9/",
        "version_string": "v0.9",
    },
    "v0.9.1": {
        "default_catalog_id": A2UI_DEFAULT_CATALOG_ID_BY_VERSION["v0.9.1"],
        "schema_base_uri": "https://a2ui.org/specification/v0_9/",
        "version_string": "v0.9.1",
    },
    "v1.0": {
        "default_catalog_id": A2UI_DEFAULT_CATALOG_ID_BY_VERSION["v1.0"],
        "schema_base_uri": "https://a2ui.org/specification/v1_0/",
        "version_string": "v1.0",
    },
}

_SUPPORTED_VERSIONS = tuple(_VERSION_CONFIG.keys())


class A2UISchemaManager:
    """Manages A2UI schema loading and system prompt generation.

    Loads the vendored JSON schemas for a given protocol version and generates
    system prompt sections that instruct an LLM to produce valid A2UI output.

    Supports custom catalogs that extend the basic catalog with additional
    components. Custom catalogs are loaded alongside the basic catalog and
    both are included in the system prompt.

    One directory per supported version (``v0_9/``, ``v0_9_1/``, ``v1_0/``),
    each with the same layout (shown for ``v0_9/``)::

        v0_9/
        ├── prompt_example.json     # Our prompt example (version-specific)
        ├── prompt_message_types.md # Message type examples to help the LLM (version-specific)
        └── spec/                   # Vendored as-is from a2ui-project/a2ui
            ├── server_to_client.json
            ├── basic_catalog.json
            ├── common_types.json
            ├── basic_catalog_rules.txt
            └── ...
    """

    def __init__(
        self,
        protocol_version: A2UIVersion = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
        spec_dir: "str | os.PathLike[str] | None" = None,
    ) -> None:
        """Initialize the schema manager.

        Args:
            protocol_version: The A2UI protocol version: "v0.9" (default), "v0.9.1", or "v1.0".
            custom_catalog: A custom catalog that extends the basic catalog. Can be:
                - A file path (str or PathLike) to a JSON catalog file
                - A dict with the catalog schema directly
                Must include a ``$id`` field (used as the catalogId in A2UI messages).
            custom_catalog_rules: Plain-text rules for the custom catalog components,
                appended to the basic catalog rules in the system prompt.
            spec_dir: Override the spec file directory.
        """
        if protocol_version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported A2UI protocol version: {protocol_version!r}. "
                f"Supported versions: {', '.join(_SUPPORTED_VERSIONS)}"
            )

        self._protocol_version = protocol_version

        version_dir = protocol_version.replace(".", "_")
        self._version_dir = (Path(spec_dir) / version_dir) if spec_dir else (_VERSIONS_DIR / version_dir)
        self._spec_dir = self._version_dir / "spec"

        self._server_to_client = self._load_spec_json("server_to_client.json")
        self._client_to_server = self._load_spec_json("client_to_server.json")
        self._common_types = self._load_spec_json("common_types.json")
        self._basic_catalog = self._load_spec_json("basic_catalog.json")
        self._catalog_rules = self._load_spec_text("basic_catalog_rules.txt")

        self._prompt_example = self._load_version_json("prompt_example.json")
        self._prompt_message_types = self._load_version_text("prompt_message_types.md")

        self._custom_catalog: JsonSchema | None = None
        self._custom_catalog_rules = custom_catalog_rules or ""

        if custom_catalog is not None:
            if isinstance(custom_catalog, dict):
                self._custom_catalog = custom_catalog
            else:
                path = Path(custom_catalog)
                try:
                    with open(path, encoding="utf-8") as f:
                        self._custom_catalog = json.load(f)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Custom catalog file not found: {path}") from e
                except json.JSONDecodeError as e:
                    raise ValueError(f"Custom catalog file {path} is not valid JSON: {e}") from e

        if self._custom_catalog is not None:
            if "$id" not in self._custom_catalog:
                raise ValueError(
                    "Custom catalog must include a '$id' field. "
                    "This is used as the catalogId in A2UI createSurface messages. "
                    'Example: {"$id": "https://mycompany.com/my_catalog.json", ...}'
                )
            self._catalog_id: str = self._custom_catalog["$id"]
        else:
            self._catalog_id = str(_VERSION_CONFIG[protocol_version]["default_catalog_id"])

    @property
    def protocol_version(self) -> A2UIVersion:
        return self._protocol_version

    @property
    def catalog_id(self) -> str:
        return self._catalog_id

    @property
    def server_to_client_schema(self) -> JsonSchema:
        return self._server_to_client

    @property
    def client_to_server_schema(self) -> JsonSchema:
        return self._client_to_server

    @property
    def basic_catalog_schema(self) -> JsonSchema:
        return self._basic_catalog

    @property
    def custom_catalog_schema(self) -> JsonSchema | None:
        return self._custom_catalog

    @property
    def common_types_schema(self) -> JsonSchema:
        return self._common_types

    @property
    def version_string(self) -> A2UIVersion:
        """The version string used in A2UI messages (e.g., 'v0.9')."""
        return _VERSION_CONFIG[self._protocol_version]["version_string"]

    @property
    def catalog_rules(self) -> str:
        return self._catalog_rules

    @property
    def custom_catalog_rules(self) -> str:
        return self._custom_catalog_rules

    def _get_active_catalog(self) -> JsonSchema:
        """Return the active catalog used as ``catalog.json`` in the registry.

        When a custom catalog is configured, returns a *merged* catalog that
        unions basic + custom components and functions, so ``$ref`` lookups
        like ``catalog.json#/components/Text`` resolve even for catalogs that
        only declare custom components. Custom catalog values override basic
        ones on key collision. ``anyComponent`` / ``anyFunction`` discriminator
        defs are rebuilt to cover the union.
        """
        if self._custom_catalog is None:
            return self._basic_catalog

        merged: JsonSchema = dict(self._basic_catalog)
        custom_id = self._custom_catalog.get("$id")
        if custom_id:
            merged["$id"] = custom_id

        components: JsonSchema = dict(self._basic_catalog.get("components", {}))
        components.update(self._custom_catalog.get("components", {}))
        merged["components"] = components

        functions: JsonSchema = dict(self._basic_catalog.get("functions", {}))
        functions.update(self._custom_catalog.get("functions", {}))
        merged["functions"] = functions

        defs: JsonSchema = dict(self._basic_catalog.get("$defs", {}))
        defs.update(self._custom_catalog.get("$defs", {}))
        if components:
            defs["anyComponent"] = {
                "oneOf": [{"$ref": f"#/components/{name}"} for name in components],
                "discriminator": {"propertyName": "component"},
            }
        if functions:
            defs["anyFunction"] = {
                "oneOf": [{"$ref": f"#/functions/{name}"} for name in functions],
            }
        merged["$defs"] = defs

        return merged

    def _get_all_components(self) -> JsonSchema:
        """Return all available components from both basic and custom catalogs."""
        components: JsonSchema = {}
        components.update(self._basic_catalog.get("components", {}))
        if self._custom_catalog is not None:
            components.update(self._custom_catalog.get("components", {}))
        return components

    def get_component_schemas(self) -> dict[str, JsonSchema]:
        """Get resolved schemas for all components, keyed by component type name."""
        schemas: dict[str, JsonSchema] = {}
        for catalog in [self._basic_catalog, self._custom_catalog]:
            if catalog is not None:
                components = catalog.get("components", {})
                for name, defn in components.items():
                    if isinstance(defn, dict) and name not in schemas:
                        schemas[name] = defn
        return schemas

    def build_schema_registry(self) -> "Registry":
        """Build a jsonschema referencing Registry for cross-file $ref resolution."""
        active_catalog = self._get_active_catalog()
        base = _VERSION_CONFIG[self._protocol_version]["schema_base_uri"]
        resources: list[tuple[str, Resource]] = [
            (f"{base}catalog.json", Resource.from_contents(active_catalog, default_specification=DRAFT202012)),
            (f"{base}common_types.json", Resource.from_contents(self._common_types, default_specification=DRAFT202012)),
            ("catalog.json", Resource.from_contents(active_catalog, default_specification=DRAFT202012)),
            ("common_types.json", Resource.from_contents(self._common_types, default_specification=DRAFT202012)),
            (
                f"{base}basic_catalog.json",
                Resource.from_contents(self._basic_catalog, default_specification=DRAFT202012),
            ),
            ("basic_catalog.json", Resource.from_contents(self._basic_catalog, default_specification=DRAFT202012)),
        ]
        for schema in [self._server_to_client, self._basic_catalog, self._common_types]:
            schema_id = schema.get("$id")
            if schema_id:
                resources.append((schema_id, Resource.from_contents(schema, default_specification=DRAFT202012)))
        if self._custom_catalog is not None:
            custom_id = self._custom_catalog.get("$id")
            if custom_id:
                # Register the MERGED catalog at the custom catalog's $id,
                # so refs like "<custom_id>#/components/Text" resolve.
                resources.append((
                    custom_id,
                    Resource.from_contents(active_catalog, default_specification=DRAFT202012),
                ))
        return Registry().with_resources(resources)

    def _load_json_file(self, path: Path) -> "list[JsonValue] | JsonSchema":
        """Load and parse a JSON file, raising a clear error on failure."""
        try:
            with open(path, encoding="utf-8") as f:
                result: list[JsonValue] | JsonSchema = json.load(f)
                return result
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"A2UI spec file not found for protocol version {self._protocol_version!r}: {path}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(f"A2UI spec file {path} is not valid JSON: {e}") from e

    def _load_spec_json(self, filename: str) -> JsonSchema:
        """Load a JSON file from the spec/ subdirectory (upstream A2UI files)."""
        data: JsonSchema = self._load_json_file(self._spec_dir / filename)  # type: ignore[assignment]
        return data

    def _load_spec_text(self, filename: str) -> str:
        """Load a text file from the spec/ subdirectory (upstream A2UI files)."""
        path = self._spec_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _load_version_json(self, filename: str) -> "list[JsonValue] | JsonSchema":
        """Load a JSON file from the version directory (our files)."""
        return self._load_json_file(self._version_dir / filename)

    def _load_version_text(self, filename: str) -> str:
        """Load a text file from the version directory (our files)."""
        path = self._version_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _build_prompt_example(self) -> str:
        """Build the prompt example JSON string from the version-specific template."""
        raw = json.dumps(self._prompt_example, separators=(",", ": "))
        return raw.replace("{catalog_id}", self._catalog_id)

    def generate_prompt_section(
        self,
        include_schema: bool = True,
        include_rules: bool = True,
        actions: list[A2UIAction] | None = None,
    ) -> str:
        """Generate the A2UI portion of the system prompt."""
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

        message_types = self._prompt_message_types.replace("{catalog_id}", self._catalog_id).replace(
            "{version_string}", v
        )
        sections.append(f"\n\n## A2UI Message Types\n\n{message_types}")

        all_components = self._get_all_components()
        if all_components:
            basic_components = sorted(self._basic_catalog.get("components", {}).keys())
            custom_components = (
                sorted(self._custom_catalog.get("components", {}).keys()) if self._custom_catalog else []
            )

            comp_section = "\n\n## Available Components\n\n"
            if basic_components:
                comp_section += f"**Basic catalog:** {', '.join(basic_components)}\n\n"
            if custom_components:
                comp_section += f"**Custom components:** {', '.join(custom_components)}\n\n"
                for comp_name in custom_components:
                    comp_def = self._custom_catalog.get("components", {}).get(comp_name, {})  # type: ignore[union-attr]
                    desc = self._extract_component_description(comp_name, comp_def)
                    if desc:
                        comp_section += f"- **{comp_name}**: {desc}\n"
                comp_section += "\n"

            comp_section += (
                "Each component uses a flat discriminator format:\n"
                '```json\n{"id": "myId", "component": "Text", "text": "Hello"}\n```\n\n'
                "Components are referenced by `id`. Use `children` (array of ids) for layout components (Column, Row). "
                "Use `child` (single id) for wrapper components (Card, Button)."
            )
            sections.append(comp_section)

        if include_rules:
            rules_parts: list[str] = []
            if self._catalog_rules:
                rules_parts.append(self._catalog_rules)
            if self._custom_catalog_rules:
                rules_parts.append(self._custom_catalog_rules)
            if rules_parts:
                sections.append("\n\n## Component Rules\n\n" + "\n\n".join(rules_parts))

        if actions:
            action_lines = ["\n\n## Available Actions\n"]
            action_lines.append("The following actions can be triggered by buttons in the UI:\n")

            event_actions = [a for a in actions if a.action_type == "event"]
            func_actions = [a for a in actions if a.action_type == "functionCall"]

            if event_actions:
                action_lines.append("### Server Events\n")
                for a in event_actions:
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

            if func_actions:
                action_lines.append("\n### Client Functions\n")
                action_lines.append(
                    "Client functions execute on the client without a server round-trip. "
                    "Use the **exact** function name, argument structure, and returnType shown below. "
                    "Do not add extra properties.\n"
                )
                for a in func_actions:
                    desc = f": {a.description}" if a.description else ""
                    example_args = json.dumps(a.example_args) if a.example_args else "{}"
                    action_lines.append(f"- **`{a.name}`**{desc}")
                    action_lines.append(
                        f"  ```json\n"
                        f'  {{"id": "action_btn", "component": "Button", "child": "action_btn_text", '
                        f'"action": {{"functionCall": {{"call": "{a.name}", "args": {example_args}, "returnType": "void"}}}}}}\n'
                        f"  ```"
                    )

            sections.append("\n".join(action_lines))

        if include_schema:
            schema_str = json.dumps(self._server_to_client, indent=2)
            sections.append(f"\n\n## A2UI Message Schema ({v})\n\n```json\n{schema_str}\n```")

        return "\n".join(sections)

    def _extract_component_description(self, name: str, comp_def: JsonSchema) -> str:
        """Extract a human-readable description of a custom component for the prompt."""
        desc: str = comp_def.get("description", "")
        if desc:
            return desc

        props: set[str] = set()
        for item in comp_def.get("allOf", []):
            if "properties" in item:
                props.update(k for k in item["properties"] if k not in ("id", "component", "accessibility"))
        if "properties" in comp_def:
            props.update(k for k in comp_def["properties"] if k not in ("id", "component", "accessibility"))

        if props:
            return f"Properties: {', '.join(sorted(props))}"
        return ""
