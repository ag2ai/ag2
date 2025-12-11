---
title: Apply Patch Tool
description: Apply code patches with GPT-5.1 using structured diffs
---

# Apply Patch Tool

The `ApplyPatchTool` enables agents to create, update, and delete files using structured diffs from OpenAI's Responses API with GPT-5.1. Unlike traditional code execution methods, the apply_patch tool provides structured, controlled file operations that are safer and more precise than raw code generation.

## Features

- **File Creation**: Generate new files with specified content
- **File Updates**: Modify existing files using unified diff format
- **File Deletion**: Remove files from the workspace
- **Path Security**: Control file operations with `allowed_paths` patterns
- **Workspace Management**: Organize operations within a dedicated workspace directory
- **Async Support**: Apply patch operations asynchronously

## Requirements

- Python >= 3.10
- GPT-5.1 access (currently in beta)
- OpenAI API key
- AG2 installed with OpenAI support

## Installation

Install AG2 with OpenAI support:

```bash
pip install ag2[openai]
```

For more information, refer to the [installation guide](/docs/user-guide/basic-concepts/installing-ag2).

## Quick Start

The simplest way to use the apply_patch tool is through the LLM configuration:

```python
import os
from autogen import ConversableAgent, LLMConfig
from dotenv import load_dotenv

load_dotenv()

# Configure LLM with apply_patch tool
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["apply_patch"],
    },
)

# Create agent - tool is automatically registered
coding_agent = ConversableAgent(
    name="coding_assistant",
    llm_config=llm_config,
    system_message="""You are a helpful coding assistant. You can create, edit, and delete files
    using the apply_patch tool. When making changes, always use the apply_patch tool rather than
    writing raw code blocks. Be precise with your file operations and explain what you're doing.""",
)

# Use the agent
result = coding_agent.initiate_chat(
    recipient=coding_agent,
    message="Create a Python file called hello.py with a hello_world function",
    max_turns=2,
)
```

## Configuration Options

### Workspace Directory

Specify a dedicated workspace directory for file operations:

```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["apply_patch"],
        "workspace_dir": "./my_project_folder",  # Root directory for operations
    },
)
```

All file operations will be relative to this workspace directory.

### Allowed Paths

Control which paths can be accessed for security:

```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["apply_patch"],
        "workspace_dir": "./my_project_folder",
        "allowed_paths": ["src/**", "tests/**", "*.py"],  # Only allow these paths
    },
)
```

**Path Pattern Examples:**
- `["**"]` - Allow all paths (default)
- `["src/**"]` - Allow all files in `src/` and subdirectories
- `["*.py"]` - Allow Python files in root directory
- `["src/**", "tests/**"]` - Allow paths in multiple directories

The `allowed_paths` parameter supports glob-style patterns with `**` for recursive matching on local filesystem paths.

### Async Patches

Apply patches asynchronously for better performance:
```python
llm_config = LLMConfig(
    config_list={
        "api_type": "responses",
        "model": "gpt-5.1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "built_in_tools": ["apply_patch_async"],  # Use async version
        "workspace_dir": "./my_project_folder",
    },
)
```

## Usage Examples

### Example 1: Creating a New Project

Create a simple Python project with multiple files:
```python
result = coding_agent.initiate_chat(
    recipient=coding_agent,
    message="""
    Create a new Python project folder called 'calculator' with the following structure:
    1. Create a main.py file with a Calculator class that has methods for add, subtract, multiply, and divide
    """,
    max_turns=2,
    clear_history=True,
)
```

### Example 2: Modifying Existing Files

Update files using the apply_patch tool:

```python
result = coding_agent.initiate_chat(
    recipient=coding_agent,
    message="""
    In calculator folder, add power and square root methods to the Calculator class in main.py
    """,
    max_turns=2,
)
```

## Understanding Apply Patch Operations

The apply_patch tool uses three types of operations:

1. **create_file** / **a_create_file**: Creates a new file with the specified content
2. **update_file** / **a_update_file**: Updates an existing file using unified diff format
3. **delete_file** / **a_delete_file**: Deletes a file from the workspace
