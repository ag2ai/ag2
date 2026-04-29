# Built-in tools — provider matrix and per-tool parameters

## Provider-native tools (server-side execution)

| Tool | Anthropic | OpenAI | Gemini |
|---|:---:|:---:|:---:|
| `CodeExecutionTool` | ✓ | ✓ | ✓ |
| `WebSearchTool` | ✓ | ✓ | ✓ |
| `WebFetchTool` | ✓ | ✗ | ✓ |
| `ShellTool` | ✓ | ✓ | ✗ |
| `MCPServerTool` | ✓ | ✓ | ✗ |
| `ImageGenerationTool` | ✗ | ✓ | ✗ |
| `MemoryTool` | ✓ | ✗ | ✗ |

Unsupported combinations raise `UnsupportedToolError` at request time.

### `WebSearchTool` parameters

| Parameter | Anthropic | OpenAI | Gemini |
|---|:---:|:---:|:---:|
| `max_uses` | ✓ | ✓ | ✗ |
| `user_location` | ✓ | ✓ | ✗ |
| `search_context_size` | ✗ | ✓ | ✗ |
| `allowed_domains` | ✓ | ✓ | ✗ |
| `blocked_domains` | ✓ | ✗ | ✓ |

Unsupported parameters are silently ignored.

### `WebFetchTool` parameters (Anthropic / Gemini only)

| Parameter | Anthropic | Gemini |
|---|:---:|:---:|
| `max_uses` | ✓ | ✗ |
| `allowed_domains` | ✓ | ✗ |
| `blocked_domains` | ✓ | ✗ |
| `citations` | ✓ | ✗ |
| `max_content_tokens` | ✓ | ✗ |

### `MCPServerTool` parameters

| Parameter | Anthropic | OpenAI |
|---|:---:|:---:|
| `server_url` | ✓ | ✓ |
| `server_label` | ✓ | ✓ |
| `authorization_token` | ✓ | ✗ |
| `description` | ✓ | ✗ |
| `allowed_tools` | ✓ | ✓ |
| `blocked_tools` | ✓ | ✗ |
| `headers` | ✗ | ✓ |

### `ImageGenerationTool` parameters (OpenAI Responses only)

| Parameter | Description |
|---|---|
| `quality` | `"low"`, `"medium"`, `"high"`, `"auto"` |
| `size` | e.g. `"1024x1024"`, `"1536x1024"`, `"auto"` |
| `background` | `"transparent"`, `"opaque"`, `"auto"` |
| `output_format` | `"png"`, `"jpeg"`, `"webp"` |
| `output_compression` | 0–100, jpeg/webp only |
| `partial_images` | 1–3, partial-stream count |

Generated images surface on `reply.images: list[bytes]`.

## Anthropic tool versions

Newer Anthropic tool revisions support dynamic filtering (the model writes code to filter results before they reach context) but require Opus 4.6 / Sonnet 4.6.

```python
from autogen.beta.tools import WebFetchTool, WebSearchTool

tools = [
    WebSearchTool(version="web_search_20260209"),  # default: web_search_20250305
    WebFetchTool(version="web_fetch_20260209"),    # default: web_fetch_20250910
]
```

Default versions are compatible with all Claude models including Haiku.

## Common toolkits (local execution, all providers)

### `FilesystemToolkit`

| Tool | Description |
|---|---|
| `read_file` | Read a file |
| `write_file` | Create / overwrite (auto-creates parent dirs) |
| `update_file` | Replace first occurrence of a string |
| `delete_file` | Delete a file |
| `find_files` | Glob search (supports `**`) |

Constructor:

```python
FilesystemToolkit(base_path="/tmp/workspace", read_only=False)
```

`base_path` is enforced — escape attempts raise `PermissionError`. Individual tools available as `fs.read_file()`, `fs.find_files()`, etc.

### `DuckDuckSearchTool`

```python
DuckDuckSearchTool(
    max_results=5,         # default
    region="us-en",        # default
    safesearch="moderate", # "on" | "moderate" | "off"
)
```

All parameters accept `Variable(...)` for deferred resolution.

### `ExaToolkit`

| Tool factory | Description |
|---|---|
| `toolkit.search(...)` | Neural web search with filters |
| `toolkit.find_similar(...)` | Find pages similar to a URL |
| `toolkit.get_contents(...)` | Fetch full text |
| `toolkit.answer(...)` | LLM answer with citations |

Constructor:

```python
ExaToolkit(api_key=..., num_results=10, max_characters=2000)
```

Per-call factory params include `search_type`, `category`, `include_domains`, `exclude_domains`, `start_published_date`, `end_published_date`, `use_autoprompt`, `livecrawl`. All accept `Variable`.

### `TavilySearchTool`

```python
TavilySearchTool(
    api_key=...,
    max_results=5,
    search_depth="advanced",   # "basic" | "advanced" | "fast" | "ultra-fast"
    topic="news",              # "general" | "news" | "finance"
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    time_range="week",         # "day" | "week" | "month" | "year"
    include_domains=[...],
    exclude_domains=[...],
)
```

### `SkillsToolkit` / `SkillSearchToolkit`

Discovers and runs skills following the [agentskills.io](https://agentskills.io) convention. By default reads from `.agents/skills/` in the current working directory.

```python
from autogen.beta.tools import SkillsToolkit
from autogen.beta.tools.skills import LocalRuntime

skills = SkillsToolkit()                              # uses .agents/skills/
skills = SkillsToolkit(runtime="./my-skills")         # custom dir
skills = SkillsToolkit(runtime=LocalRuntime("./my-skills", extra_paths=["./shared-skills"]))
```

Three-step progressive disclosure: `list_skills` (catalog) → `load_skill` (full SKILL.md) → `run_skill_script` (execute a script).

`SkillSearchToolkit` adds `search_skills`, `install_skill`, `remove_skill` against the [skills.sh](https://skills.sh) registry. Set `GITHUB_TOKEN` (env or `SkillsClientConfig`) to lift the GitHub API rate limit from 60 → 5000 requests/hour.

## Required extras

| Tool | `pip install` |
|---|---|
| `DuckDuckSearchTool` | `ag2[ddgs]` |
| `ExaToolkit` | `ag2[exa]` |
| `TavilySearchTool` | `ag2[tavily]` |
| Provider-native tools | the provider's own extra (`ag2[anthropic]`, `ag2[openai]`, `ag2[gemini]`) |
