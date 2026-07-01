# website/ Documentation Guidelines

## Documentation

Documentation sources live in `website/docs/user-guide/*.mdx`. Each page is an MDX file.

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
- All Python imports in examples must use the `ag2` module path (e.g., `from ag2 import Agent, tool`).
- When showing the same concept across multiple providers or alternatives, use `<Tabs>` and `<Tab title="...">` components. Each tab should be self-contained with its own code block.

Example:

````mdx
<Tabs>
  <Tab title="OpenAI">
```python linenums="1"
from ag2.config import OpenAIConfig

config = OpenAIConfig(model="gpt-5")
```
  </Tab>

  <Tab title="Anthropic">
```python linenums="1"
from ag2.config import AnthropicConfig

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

!!! warning "Single blank lines only — MkDocs collapses consecutive blanks"

    MkDocs Material **collapses two or more consecutive blank lines inside a code block to a single blank line** when it renders. The source still counts every blank, so any `hl_lines` you author against the source silently mis-highlights once rendered — every highlighted line below a collapsed run shifts up. **Never put two or more blank lines in a row inside a code example.** Keep all code blocks to single blank-line separators so the source line numbers match the rendered output and `hl_lines` stays accurate. (This includes PEP 8's two-blank-lines-between-top-level-defs convention — use one blank line in docs examples.)

Examples:

````
```python linenums="1"
from ag2 import Agent
```
````

````
```python linenums="1" hl_lines="3"
from ag2 import Agent

agent = Agent(name="Assistant")  # this line is highlighted
```
````

````
```python linenums="1" hl_lines="1-2 5"
from ag2 import Agent, tool  # highlighted
from ag2 import Context       # highlighted

@tool
def greet(name: str) -> str:  # highlighted
    """Greets a person by name."""
    return f"Hello, {name}!"
```
````

**Maintaining `hl_lines`** — these line numbers are easy to break:

- **No double blank lines in the block.** Two or more consecutive blanks are collapsed at render time (see the warning above), which shifts every highlight below them. Confirm the block has only single blank-line separators before trusting any `hl_lines`.
- **Line numbers are 1-indexed from the first code line** — the line immediately after the opening ```` ``` ```` fence is line `1`. The fence itself is not counted, and the count is independent of what `linenums="1"` starts the *displayed* numbering at.
- **Recompute `hl_lines` whenever you edit a block.** Inserting or removing a line shifts every range below it, so an unrelated edit silently mis-highlights. After editing, re-count from the first code line and update the ranges.
- **Highlight the lines that carry the point** — the construction, the call, the changed lines. Do not highlight blank lines or unrelated imports; a range that includes them is a sign the numbers have drifted.
- **Keep brackets balanced.** A highlight group must not open a bracket it doesn't close. If a highlighted line opens a multi-line `(`/`[`/`{`, either extend the range through its matching closing line (highlight the whole construct, closing bracket included) **or** highlight only the lines *inside* the brackets (excluding both the opener and the closer). A range that ends on a dangling open bracket — or a lone closing bracket with no matching open in the same group — renders as a visually unbalanced block.
- **Verify before committing.** Open the block, count to each highlighted line, and confirm it lands on what the surrounding prose says it does.

### Code Examples

- Examples should be self-contained: a reader should be able to copy-paste and run them.
- Show the simplest working version first, then progressively add complexity.
- Use realistic but concise variable names and docstrings — the LLM reads these, so they matter.
- When demonstrating a tool or feature, show both the definition and how it's wired into an Agent.

### Navigation

Page navigation is defined in `website/mint-json-template.json.jinja` under the `"navigation"` key. User Guide pages live under the `"User Guide"` group.

To add a new page:

1. Create the MDX file in `website/docs/user-guide/` (or a subdirectory).
2. Add its path to the `"User Guide"` group's `"pages"` array in `mint-json-template.json.jinja`. Paths are relative to the `website/` root and omit the `.mdx` extension (e.g., `"docs/user-guide/my_new_page"`).
3. For nested groups, wrap pages in a `{"group": "Group Name", "pages": [...]}` object.

**Subfolder rule**: navigation groups with subpages must be backed by a matching subfolder in the source tree. Place pages for a group inside `docs/user-guide/{group_name}/`. Do **not** use `index.mdx` — subfolders do not support index files. Instead, name the main page explicitly (e.g., `docs/user-guide/tools/tools.mdx`). For example, the "Tools" group maps to `docs/user-guide/tools/tools.mdx`, `docs/user-guide/tools/toolkits.mdx`, etc.

Example — adding a standalone page:

```json
{
  "group": "User Guide",
  "pages": [
    "docs/user-guide/motivation",
    "docs/user-guide/agents",
    "docs/user-guide/my_new_page"
  ]
}
```

Example — adding a page inside a nested group:

```json
{
  "group": "Tools",
  "pages": [
    "docs/user-guide/tools/tools",
    "docs/user-guide/tools/toolkits",
    "docs/user-guide/tools/builtin_tools"
  ]
}
```

### Links

- **Internal links**: use absolute paths from the docs root: `[Agent Tools](/docs/user-guide/tools/)`. Do not use relative paths.
- **External links**: append `{.external-link target="_blank"}` after the markdown link. Use descriptive anchor text — do **not** use bare URLs or generic text like "here" or "this link".

Good: `[OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/){.external-link target="_blank"}`
Bad: `[OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)` (missing attribute syntax)
Bad: `[click here](https://example.com){.external-link target="_blank"}` (generic anchor text)
