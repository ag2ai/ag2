# website/ Documentation Guidelines

## Beta Documentation

Beta docs sources live in `website/docs/beta/*.mdx`. Each page is an MDX file.

### Frontmatter

Every MDX file must include YAML frontmatter with at least a `title`. Use `sidebarTitle` for a shorter sidebar label when the full title is long:

```yaml
---
title: "Full Page Title"
sidebarTitle: "Short Name"
---
```

### Writing Style

- Start each page with a brief 1-2 sentence overview of the concept before diving into details.
- Use MkDocs Material admonition syntax for callouts: `!!! note`, `!!! warning`, `!!! tip`. Do **not** use Docusaurus-style `:::` fences or HTML `<Note>`/`<Warning>` JSX components.
- Use ```` ```python linenums="1" ```` for code blocks that benefit from line numbers.
- All Python imports in examples must use the `autogen.beta` module path (e.g., `from autogen.beta import Agent, tool`).
- When showing the same concept across multiple providers or alternatives, use `<Tabs>` and `<Tab title="...">` components. Each tab should be self-contained with its own code block.

Example:

````mdx
<Tabs>
  <Tab title="OpenAI">
```python linenums="1"
from autogen.beta.config import OpenAIConfig

config = OpenAIConfig(model="gpt-5")
```
  </Tab>

  <Tab title="Anthropic">
```python linenums="1"
from autogen.beta.config import AnthropicConfig

config = AnthropicConfig(model="claude-haiku-4-5-20251001")
```
  </Tab>
</Tabs>
````

### Code Blocks

Code blocks use MkDocs Material syntax with these attributes:

- Always specify the language (e.g., `python`, `json`, `bash`).
- **Inline code highlighting**: prefix inline code with `#!python` to get syntax highlighting: `` `#!python await reply.content()` ``.
- **Line numbers**: add `linenums="1"` to show numbered lines.
- **Line highlighting**: add `hl_lines="..."` to highlight specific lines. Supports single lines (`"6"`), multiple lines (`"4 9 15"`), and ranges (`"15-18"`). Can be combined (`"3-6 8"`).

Examples:

````
```python linenums="1"
from autogen.beta import Agent
```
````

````
```python linenums="1" hl_lines="3"
from autogen.beta import Agent

agent = Agent(name="Assistant")  # this line is highlighted
```
````

````
```python linenums="1" hl_lines="1-2 5"
from autogen.beta import Agent, tool  # highlighted
from autogen.beta import Context       # highlighted

@tool
def greet(name: str) -> str:  # highlighted
    """Greets a person by name."""
    return f"Hello, {name}!"
```
````

### Code Examples

- Examples should be self-contained: a reader should be able to copy-paste and run them.
- Show the simplest working version first, then progressively add complexity.
- Use realistic but concise variable names and docstrings — the LLM reads these, so they matter.
- When demonstrating a tool or feature, show both the definition and how it's wired into an Agent.

### Navigation

Page navigation is defined in `website/mint-json-template.json.jinja` under the `"navigation"` key. Beta pages live under the `"Beta"` group.

To add a new page:

1. Create the MDX file in `website/docs/beta/` (or a subdirectory).
2. Add its path to the `"Beta"` group's `"pages"` array in `mint-json-template.json.jinja`. Paths are relative to the `website/` root and omit the `.mdx` extension (e.g., `"docs/beta/my_new_page"`).
3. For nested groups, wrap pages in a `{"group": "Group Name", "pages": [...]}` object.

**Subfolder rule**: navigation groups with subpages must be backed by a matching subfolder in the source tree. Place pages for a group inside `docs/beta/{group_name}/`. Do **not** use `index.mdx` — subfolders do not support index files. Instead, name the main page explicitly (e.g., `docs/beta/tools/tools.mdx`). For example, the "Tools" group maps to `docs/beta/tools/tools.mdx`, `docs/beta/tools/toolkits.mdx`, etc.

Example — adding a standalone page:

```json
{
  "group": "Beta",
  "pages": [
    "docs/beta/motivation",
    "docs/beta/agents",
    "docs/beta/my_new_page"
  ]
}
```

Example — adding a page inside a nested group:

```json
{
  "group": "Tools",
  "pages": [
    "docs/beta/tools/tools",
    "docs/beta/tools/toolkits",
    "docs/beta/tools/builtin_tools"
  ]
}
```

### Links

- **Internal links**: use absolute paths from the docs root: `[Agent Tools](/docs/beta/tools/)`. Do not use relative paths.
- **External links**: append `{.external-link target="_blank"}` after the markdown link. Use descriptive anchor text — do **not** use bare URLs or generic text like "here" or "this link".

Good: `[OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/){.external-link target="_blank"}`
Bad: `[OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)` (missing attribute syntax)
Bad: `[click here](https://example.com){.external-link target="_blank"}` (generic anchor text)

## Notebooks

Notebooks live in the `notebook/` directory (project root, not under `website/`). They are rendered to MDX by Quarto via `process_notebooks.py`.

### Required Metadata

Notebooks must include a `front_matter` key in notebook-level metadata with:
- `description` (required) — short summary of the notebook
- `tags` (required) — list of string tags

The `title` is auto-extracted from the first `# Heading` in the notebook.

### Skipping

- Set `skip_render: true` in notebook metadata to exclude from the docs build.
- Set `skip_test: true` to exclude from CI test execution.
