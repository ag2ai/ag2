---
name: docs-writer
description: AG2 documentation conventions — MDX format, frontmatter, admonitions, code blocks, and how to update the roadmap.
version: 1.0.0
license: Apache-2.0
---

# Docs Writer

Guidelines for writing and updating AG2 documentation under `website/docs/beta/`.

## File format

Docs are MDX files rendered by [Mintlify](https://mintlify.com). Every file starts with YAML frontmatter:

```mdx
---
title: My Feature
sidebarTitle: My Feature        # optional — if different from title
---

# My Feature

Content here.
```

`title` is the `<title>` tag in the browser. `sidebarTitle` appears in the left navigation.

## Code blocks

Always add `linenums="1"` to production code examples:

````mdx
```python linenums="1"
from autogen.beta import Agent

agent = Agent("assistant", config=config)
```
````

Use ` ```bash ` for shell commands (no `linenums`). Use ` ```python linenums="1" ` for all Python examples.

To highlight specific lines, add `hl_lines="2 3"` after the language:

````mdx
```python linenums="1" hl_lines="2 3"
from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig   # (1)!

agent = Agent("assistant", config=OpenAIConfig("gpt-4o-mini"))
```
````

## Inline code

Use single backticks for inline code: `` `agent.ask()` ``, `` `MemoryToolkit` ``.

When inlining Python syntax that should be rendered with syntax highlighting in Mintlify, prefix with `#!python`:

```mdx
Pass `#!python read_only=True` to restrict the toolkit to read operations.
```

## Admonitions

Admonitions use `!!!`:

```mdx
!!! tip
    Short helpful note here.

!!! note
    An important note.

!!! warning
    A warning the reader must not miss.
```

## External links

Always add `{.external-link target="_blank"}` to external URLs:

```mdx
[Mintlify](https://mintlify.com){.external-link target="_blank"}
```

## Tables

Use pipe syntax with a header-separator row:

```mdx
| Tool | Description |
| :--- | :--- |
| `read_file` | Read the contents of a file. |
| `write_file` | Create or overwrite a file. |
```

Left-align columns with `:---`.

## Docstrings

Python docstrings in `autogen.beta` use plain reStructuredText (not Google style):

```python
def fanout(agents, prompts, *, max_concurrent=None):
    """Run agent.ask() calls in parallel and return results in input order.

    Args:
        agents: A single :class:`Agent`, a list of agents, or a list of
            ``(agent, prompt)`` pairs.
        prompts: Depends on *agents* — see calling conventions.
        max_concurrent: Maximum number of concurrent agent calls.
            ``None`` means unlimited.

    Returns:
        A list of :class:`AgentReply` in the same order as the inputs.

    Raises:
        ValueError: If the argument combination is invalid.
    """
```

Rules:
- First line: one-sentence summary ending in a period.
- Blank line before Args / Returns / Raises.
- Wrap at 88 characters.
- Only document public APIs; private helpers (`_foo`) need no docstring.

## Updating the navigation

`website/mint-json-template.json.jinja` controls the sidebar. To add a new page, find the relevant `"group"` and add the path (without `.mdx`) to its `"pages"` array:

```json
{
  "group": "Advanced",
  "pages": [
    "docs/beta/advanced/fanout",
    "docs/beta/advanced/scheduler",
    "docs/beta/advanced/my-new-feature"
  ]
}
```

## Updating the roadmap

`website/docs/beta/roadmap.mdx` has three sections: **Completed**, **In Progress**, and **Future Priorities**.

When shipping a feature:
1. Remove the item from **In Progress** or the relevant **P-level** block.
2. Add it to **Completed** with a short name matching the existing style.
