# ag2 convert

> Convert between agent frameworks — migrate from CrewAI, LangChain, and others to AG2.

## Problem

AG2 already has interoperability adapters (`autogen.interop`) for CrewAI,
LangChain, and Pydantic-AI. But using them requires understanding both
frameworks. A CLI command that does the conversion automatically would
make migration effortless.

## Commands

```bash
# Convert a CrewAI project to AG2
ag2 convert ./crewai-project --from crewai

# Convert LangChain tools to AG2 tools
ag2 convert ./tools.py --from langchain

# Convert Pydantic-AI agents
ag2 convert ./my_agent.py --from pydantic-ai

# Convert AG2 agent to an MCP server
ag2 convert my_agent.py --to mcp --output mcp_server.py

# Convert AG2 agent to an A2A endpoint
ag2 convert my_agent.py --to a2a --output a2a_server.py

# Preview changes without writing
ag2 convert ./crewai-project --from crewai --dry-run

# Convert with AI assistance for complex cases
ag2 convert ./crewai-project --from crewai --ai-assist
```

## Conversion Maps

### CrewAI → AG2

| CrewAI | AG2 |
|--------|-----|
| `Agent(role=..., goal=..., backstory=...)` | `AssistantAgent(name=..., system_message=...)` |
| `Task(description=..., agent=...)` | Part of `initiate_chat()` message |
| `Crew(agents=..., tasks=..., process=sequential)` | `run_group_chat()` with sequential pattern |
| `Crew(process=hierarchical)` | `run_group_chat()` with auto pattern |
| `@tool` decorator | `@tool` decorator (different import) |
| `crew.kickoff()` | `initiate_chat()` or `run_group_chat()` |

### LangChain → AG2

| LangChain | AG2 |
|-----------|-----|
| `BaseTool` subclass | `@tool` decorated function |
| `StructuredTool.from_function()` | `Tool(func=...)` |
| `AgentExecutor` | `ConversableAgent` with tools |
| `ChatPromptTemplate` | `system_message` |

### AG2 → MCP

| AG2 | MCP Server |
|-----|------------|
| `@tool` functions | MCP tool handlers |
| Agent description | Server capabilities |
| `initiate_chat()` | `chat` MCP tool |

## Example Output

```bash
ag2 convert ./crewai-project --from crewai
```

```
╭─ AG2 Convert ─ CrewAI → AG2 ──────────────────────╮
│ Analyzing CrewAI project...                         │
╰─────────────────────────────────────────────────────╯

  Found:
    3 agents (researcher, writer, editor)
    5 tools (search, scrape, format, save, review)
    2 tasks (research_task, writing_task)
    1 crew (sequential process)

  Converting:
    ✓ agents/researcher.py    → agents/researcher.py
    ✓ agents/writer.py        → agents/writer.py
    ✓ agents/editor.py        → agents/editor.py
    ✓ tools/search_tool.py    → tools/search.py
    ✓ tools/scrape_tool.py    → tools/scrape.py
    ✓ crew.py                 → main.py (run_group_chat)
    ✓ pyproject.toml          → updated dependencies

  ⚠ Manual review needed:
    - tools/review.py: CrewAI's `cache_function` has no AG2 equivalent.
      Generated code uses a simple dict cache instead.
    - crew.py: CrewAI's `memory=True` was removed. See AG2 docs for
      persistent memory options.

  Run `ag2 doctor main.py` to verify the converted code.
```

## --ai-assist Mode

For complex conversions with custom logic, use `--ai-assist` to have an
LLM analyze the source code and generate idiomatic AG2 equivalents:

```bash
ag2 convert ./complex-crewai-app --from crewai --ai-assist
```

The AI:
1. Reads all source files in the project
2. Understands the intent (not just syntax mapping)
3. Generates idiomatic AG2 code that may use different patterns
4. Explains non-trivial conversion decisions

## Implementation Notes

### AST-Based Conversion
For simple cases, use Python AST transformation:
1. Parse the source file
2. Identify framework-specific patterns (imports, class instantiations)
3. Apply transformation rules
4. Generate new source code with `ast.unparse()` or template rendering

### AI-Assisted Conversion
For complex cases, send the source code to an LLM with a conversion prompt
that includes AG2 API documentation (from the skills pack).

### Validation
After conversion, automatically run `ag2 doctor` on the output to catch
any issues the conversion missed.
