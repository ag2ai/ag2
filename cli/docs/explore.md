# ag2 explore

> AI-powered analysis of codebases, APIs, and databases to auto-generate agents and tools.

## Problem

Building agents requires manually identifying what tools are needed, writing
tool wrappers, and designing agent architectures. For common targets (REST
APIs, databases, codebases), this is predictable work that an AI can do.

## Commands

```bash
# Analyze a codebase and recommend an agent team
ag2 explore ./my-project

# Generate AG2 tools from an OpenAPI spec
ag2 explore --api https://api.example.com/openapi.json

# Generate tools from a database schema
ag2 explore --db postgresql://localhost/mydb

# Generate tools from a GraphQL schema
ag2 explore --graphql https://api.example.com/graphql

# Generate MCP tool wrappers from a CLI tool
ag2 explore --cli kubectl
```

## Codebase Exploration

```bash
ag2 explore ./my-ecommerce-app
```

```
╭─ AG2 Explore ─ ./my-ecommerce-app ────────────────╮
│ Analyzing project structure...                      │
╰─────────────────────────────────────────────────────╯

  Detected: Django 5.0 / PostgreSQL / Redis / Stripe API
  Files: 342 Python, 89 templates, 45 tests

╭─ Recommended Agent Team ───────────────────────────╮
│                                                     │
│  1. CodeReviewAgent                                 │
│     Review PRs against your Django coding standards │
│     Tools: git_diff, django_check, run_tests        │
│                                                     │
│  2. TestGeneratorAgent                              │
│     Generate tests for uncovered views and models   │
│     Tools: coverage_report, pytest_run, file_write  │
│                                                     │
│  3. DBQueryAgent                                    │
│     Answer questions about your data                │
│     Tools: sql_query (read-only), schema_inspect    │
│                                                     │
│  4. APIDocsAgent                                    │
│     Keep API docs in sync with your DRF views       │
│     Tools: openapi_generate, file_read, file_write  │
│                                                     │
╰─────────────────────────────────────────────────────╯

  Scaffold these agents? [Y/n] y
  → Created agents/code_review.py
  → Created agents/test_generator.py
  → Created agents/db_query.py
  → Created agents/api_docs.py
  → Created tools/git_diff.py (4 tools)
  → Created tools/db_tools.py (2 tools)
```

## API Exploration

```bash
ag2 explore --api https://petstore.swagger.io/v2/swagger.json
```

```
╭─ API Analysis ─ Petstore API ──────────────────────╮
│ Base URL: https://petstore.swagger.io/v2            │
│ Endpoints: 18 | Auth: API key                       │
╰─────────────────────────────────────────────────────╯

  Generated tools:
    find_pets_by_status     GET  /pet/findByStatus
    get_pet_by_id           GET  /pet/{petId}
    add_pet                 POST /pet
    update_pet              PUT  /pet
    delete_pet              DELETE /pet/{petId}
    place_order             POST /store/order
    get_order_by_id         GET  /store/order/{orderId}
    get_inventory           GET  /store/inventory
    create_user             POST /user
    get_user_by_name        GET  /user/{username}
    ... (8 more)

  → Created tools/petstore_api.py (18 tools)
  → Created tools/petstore_auth.py (auth helper)
```

Generated tool example:
```python
from autogen.tools import tool

@tool(name="find_pets_by_status", description="Finds pets by their status (available, pending, sold)")
def find_pets_by_status(status: str = "available") -> str:
    """Find pets in the store by status.

    Args:
        status: Pet status to filter by. One of: available, pending, sold.
    """
    import httpx
    response = httpx.get(
        "https://petstore.swagger.io/v2/pet/findByStatus",
        params={"status": status},
        headers={"api_key": os.environ.get("PETSTORE_API_KEY", "")},
    )
    response.raise_for_status()
    return json.dumps(response.json(), indent=2)
```

## Database Exploration

```bash
ag2 explore --db postgresql://localhost/mydb
```

```
╭─ Database Analysis ─ mydb (PostgreSQL) ────────────╮
│ Tables: 24 | Views: 3 | Size: 2.1 GB               │
╰─────────────────────────────────────────────────────╯

  Generated tools (read-only by default):
    query_users          SELECT on users (12 columns)
    query_orders         SELECT on orders (8 columns)
    query_products       SELECT on products (15 columns)
    get_table_schema     Inspect any table's columns and types
    run_sql              Execute arbitrary read-only SQL

  → Created tools/db_tools.py (5 tools)
  → Created config/db_schema.yaml (full schema reference)

  ⚠ All tools use READ-ONLY connections by default.
    Pass --write to generate mutation tools.
```

## CLI Tool Exploration

```bash
ag2 explore --cli kubectl
```

Analyzes the CLI tool's `--help` output (recursively for subcommands) and
generates AG2 tools:

```
  Generated tools:
    kubectl_get_pods        kubectl get pods
    kubectl_get_services    kubectl get services
    kubectl_describe        kubectl describe <resource>
    kubectl_logs            kubectl logs <pod>
    kubectl_apply           kubectl apply -f <file>
    ... (12 more)

  → Created tools/kubectl_tools.py (17 tools)
```

## Implementation Notes

### Codebase Analysis
1. Walk the directory tree, identify framework (Django, FastAPI, Flask, etc.)
2. Parse key config files (requirements.txt, pyproject.toml, docker-compose.yml)
3. Use an LLM to analyze the project structure and suggest agent teams
4. Generate agent and tool code using AG2's code generation templates

### API Tool Generation
1. Parse OpenAPI/Swagger spec (use `httpx` to fetch if URL)
2. For each endpoint, generate an `@tool`-decorated function
3. Handle auth (API key, OAuth, Bearer token) via env vars
4. Generate proper type annotations from the schema
5. Add response parsing and error handling

### Database Tool Generation
1. Connect and introspect schema (tables, columns, types, relationships)
2. Generate typed query tools for each major table
3. Generate a general `run_sql` tool with read-only enforcement
4. Export schema as YAML for LLM context (so agents understand the data model)

### Safety
- Database tools are read-only by default
- Generated API tools never hardcode credentials
- Code execution tools always suggest Docker isolation
- All generated code includes `# AUTO-GENERATED by ag2 explore` headers
