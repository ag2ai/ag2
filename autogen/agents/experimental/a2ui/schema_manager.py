# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import json
from pathlib import Path
from typing import Any

from ....import_utils import optional_import_block
from .a2a_helpers import A2UI_DEFAULT_DELIMITER

with optional_import_block():
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

_VERSIONS_DIR = Path(__file__).parent

_VERSION_CONFIG: dict[str, dict[str, str]] = {
    "v0.9": {
        "default_catalog_id": "https://a2ui.org/specification/v0_9/basic_catalog.json",
        "schema_base_uri": "https://a2ui.org/specification/v0_9/",
        "version_string": "v0.9",
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

    Directory structure per version::

        v0_9/
        ├── prompt_example.json     # Our prompt example (version-specific)
        ├── prompt_message_types.md # Message type examples to help the LLM (version-specific)
        └── spec/                   # Copied directly from google/A2UI
            ├── server_to_client.json
            ├── basic_catalog.json
            ├── common_types.json
            ├── basic_catalog_rules.txt
            └── ...
    """

    def __init__(
        self,
        protocol_version: str = "v0.9",
        custom_catalog: str | Path | dict[str, Any] | None = None,
        custom_catalog_rules: str | None = None,
        spec_dir: str | Path | None = None,
    ) -> None:
        """Initialize the schema manager.

        Args:
            protocol_version: The A2UI protocol version. Currently only "v0.9".
            custom_catalog: A custom catalog that extends the basic catalog. Can be:
                - A file path (str or Path) to a JSON catalog file
                - A dict with the catalog schema directly
                Must include a ``$id`` field (used as the catalogId in A2UI messages).
                The custom catalog should reference ``basic_catalog.json`` in its
                ``$defs/anyComponent`` to inherit all basic components.
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

        version_dir = protocol_version.replace(".", "_")  # "v0.9" -> "v0_9"
        self._version_dir = (Path(spec_dir) / version_dir) if spec_dir else (_VERSIONS_DIR / version_dir)
        self._spec_dir = self._version_dir / "spec"

        # Load spec files (from google/A2UI)
        self._server_to_client = self._load_spec_json("server_to_client.json")
        self._common_types = self._load_spec_json("common_types.json")
        self._basic_catalog = self._load_spec_json("basic_catalog.json")
        self._catalog_rules = self._load_spec_text("basic_catalog_rules.txt")

        # Load our version-specific files
        self._prompt_example = self._load_version_json("prompt_example.json")
        self._prompt_message_types = self._load_version_text("prompt_message_types.md")

        # Load custom catalog
        self._custom_catalog: dict[str, Any] | None = None
        self._custom_catalog_rules = custom_catalog_rules or ""

        if custom_catalog is not None:
            if isinstance(custom_catalog, dict):
                self._custom_catalog = custom_catalog
            else:
                path = Path(custom_catalog)
                with open(path) as f:
                    self._custom_catalog = json.load(f)

        # Set catalog ID from custom catalog's $id or default
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
    def protocol_version(self) -> str:
        return self._protocol_version

    @property
    def catalog_id(self) -> str:
        return self._catalog_id

    @property
    def server_to_client_schema(self) -> dict[str, Any]:
        return self._server_to_client

    @property
    def basic_catalog_schema(self) -> dict[str, Any]:
        return self._basic_catalog

    @property
    def custom_catalog_schema(self) -> dict[str, Any] | None:
        return self._custom_catalog

    @property
    def common_types_schema(self) -> dict[str, Any]:
        return self._common_types

    @property
    def version_string(self) -> str:
        """The version string used in A2UI messages (e.g., 'v0.9')."""
        return _VERSION_CONFIG[self._protocol_version]["version_string"]

    @property
    def catalog_rules(self) -> str:
        return self._catalog_rules

    @property
    def custom_catalog_rules(self) -> str:
        return self._custom_catalog_rules

    def _get_active_catalog(self) -> dict[str, Any]:
        """Return the custom catalog if set, otherwise the basic catalog."""
        return self._custom_catalog if self._custom_catalog is not None else self._basic_catalog

    def _get_all_components(self) -> dict[str, Any]:
        """Return all available components from both basic and custom catalogs."""
        components: dict[str, Any] = {}
        components.update(self._basic_catalog.get("components", {}))
        if self._custom_catalog is not None:
            components.update(self._custom_catalog.get("components", {}))
        return components

    def get_component_schemas(self) -> dict[str, dict[str, Any]]:
        """Get resolved schemas for all components, keyed by component type name.

        Returns:
            A dict mapping component type names (e.g. "Button") to their
            JSON Schema definitions from the catalog ``components`` section.
        """
        schemas: dict[str, dict[str, Any]] = {}
        for catalog in [self._basic_catalog, self._custom_catalog]:
            if catalog is not None:
                components = catalog.get("components", {})
                for name, defn in components.items():
                    if isinstance(defn, dict) and name not in schemas:
                        schemas[name] = defn
        return schemas

    def build_schema_registry(self) -> Any:
        """Build a jsonschema referencing Registry for cross-file $ref resolution.

        The v0.9 ``server_to_client.json`` schema references ``catalog.json``
        and ``common_types.json`` via relative ``$ref``. This method creates a
        registry that maps those URIs to the loaded schemas so that
        ``jsonschema.validate()`` can resolve them locally.

        When a custom catalog is provided, it is registered as ``catalog.json``
        so that the envelope schema validates against the custom components.

        Returns:
            A ``referencing.Registry`` instance.
        """
        active_catalog = self._get_active_catalog()
        base = _VERSION_CONFIG[self._protocol_version]["schema_base_uri"]
        resources: list[tuple[str, Any]] = [
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
                resources.append((
                    custom_id,
                    Resource.from_contents(self._custom_catalog, default_specification=DRAFT202012),
                ))
        return Registry().with_resources(resources)

    def _load_spec_json(self, filename: str) -> dict[str, Any]:
        """Load a JSON file from the spec/ subdirectory (upstream A2UI files)."""
        path = self._spec_dir / filename
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
            return data

    def _load_spec_text(self, filename: str) -> str:
        """Load a text file from the spec/ subdirectory (upstream A2UI files)."""
        path = self._spec_dir / filename
        if path.exists():
            with open(path) as f:
                return f.read().strip()
        return ""

    def _load_version_json(self, filename: str) -> list[Any] | dict[str, Any]:
        """Load a JSON file from the version directory (our files)."""
        path = self._version_dir / filename
        with open(path) as f:
            result: list[Any] | dict[str, Any] = json.load(f)
            return result

    def _load_version_text(self, filename: str) -> str:
        """Load a text file from the version directory (our files)."""
        path = self._version_dir / filename
        if path.exists():
            with open(path) as f:
                return f.read().strip()
        return ""

    def _build_prompt_example(self) -> str:
        """Build the prompt example JSON string from the version-specific template.

        Substitutes ``{catalog_id}`` in the template with the active catalog ID.
        """
        raw = json.dumps(self._prompt_example, separators=(",", ": "))
        return raw.replace("{catalog_id}", self._catalog_id)

    def generate_prompt_section(
        self,
        include_schema: bool = True,
        include_rules: bool = True,
        response_delimiter: str = A2UI_DEFAULT_DELIMITER,
        actions: list[Any] | None = None,
    ) -> str:
        """Generate the A2UI portion of the system prompt.

        Args:
            include_schema: Whether to include the JSON schema for validation reference.
            include_rules: Whether to include the plain-text catalog rules.
            response_delimiter: The delimiter the LLM should use to separate text from A2UI JSON.
            actions: List of ``A2UIAction`` objects. If provided, an "Available Actions"
                section is injected into the prompt listing the actions and how to
                create buttons for them.

        Returns:
            A string to append to the agent's system prompt.
        """
        v = self._protocol_version
        sections: list[str] = []

        example_json = self._build_prompt_example()

        sections.append(
            f"## A2UI Response Format ({v})\n\n"
            f"You can generate rich UI responses using the A2UI {v} protocol. "
            "When you want to display a UI, respond with your text message first, "
            f"then the delimiter `{response_delimiter}`, "
            "then a JSON array of A2UI message objects.\n\n"
            "Example response format:\n"
            "```\n"
            "Here is the UI you requested.\n"
            f"{response_delimiter}\n"
            f"{example_json}\n"
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

    def _extract_component_description(self, name: str, comp_def: dict[str, Any]) -> str:
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
