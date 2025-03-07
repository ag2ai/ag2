# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from autogen._website.generate_mkdocs import (
    add_api_ref_to_mkdocs_template,
    filter_excluded_files,
    fix_asset_path,
    format_navigation,
    generate_mkdocs_navigation,
    process_and_copy_files,
    transform_card_grp_component,
    transform_tab_component,
)
from autogen._website.utils import NavigationGroup
from autogen.import_utils import optional_import_block, run_for_optional_imports

with optional_import_block():
    import jinja2

    assert jinja2


def test_exclude_files() -> None:
    files = [
        Path("/tmp/ag2/ag2/website/docs/user-guide/advanced-concepts/groupchat/groupchat.mdx"),
        Path("/tmp/ag2/ag2/website/docs/user-guide/advanced-concepts/groupchat/chat.txt"),
        Path("/tmp/ag2/ag2/website/docs/_blogs/2023-04-21-LLM-tuning-math/index.mdx"),
        Path("/tmp/ag2/ag2/website/docs/home/home.mdx"),
        Path("/tmp/ag2/ag2/website/docs/home/quick-start.mdx"),
    ]

    exclusion_list = ["docs/_blogs", "docs/home"]
    website_dir = Path("/tmp/ag2/ag2/website")

    actual = filter_excluded_files(files, exclusion_list, website_dir)
    expected = files[:2]
    assert actual == expected


def test_process_and_copy_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source directory structure
        src_dir = Path(tmpdir) / "src"
        src_dir.mkdir()

        files = [
            src_dir / "user-guide" / "advanced-concepts" / "groupchat" / "groupchat.mdx",
            src_dir / "user-guide" / "advanced-concepts" / "groupchat" / "chat.txt",
            src_dir / "home" / "agent.png",
            src_dir / "home" / "quick-start.mdx",
        ]
        # Create the content for quick-start.mdx
        quick_start_content = dedent("""
            <Tip>
            It is important to never hard-code secrets into your code, therefore we read the OpenAI API key from an environment variable.
            ```bash
            pip install -U autogen
            ```
            </Tip>

            <Warning>
            It is important to never hard-code secrets into your code, therefore we read the OpenAI API key from an environment variable.
            </Warning>

            <Note>
            It is important to never hard-code secrets into your code, therefore we read the OpenAI API key from an environment variable.
            </Note>

            <img src="https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/logo/banner-badge.png?raw=true" style={{ width: '40%' }} alt="AgentOps logo"/>

            """).lstrip()

        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)
            if file.name == "quick-start.mdx":
                file.write_text(quick_start_content)
            else:
                file.touch()

        mkdocs_output_dir = Path(tmpdir) / "mkdocs_output"
        mkdocs_output_dir.mkdir()

        process_and_copy_files(src_dir, mkdocs_output_dir, files)

        actual = list(filter(lambda x: x.is_file(), mkdocs_output_dir.rglob("*")))
        expected = [
            mkdocs_output_dir / "home" / "agent.png",
            mkdocs_output_dir / "home" / "quick-start.md",
            mkdocs_output_dir / "user-guide" / "advanced-concepts" / "groupchat" / "chat.txt",
            mkdocs_output_dir / "user-guide" / "advanced-concepts" / "groupchat" / "groupchat.md",
        ]
        assert len(actual) == len(expected)
        assert sorted(actual) == sorted(actual)

        # Assert the content of the transformed markdown file
        expected_quick_start_content = dedent("""
            !!! tip
                It is important to never hard-code secrets into your code, therefore we read the OpenAI API key from an environment variable.
                ```bash
                pip install -U autogen
                ```

            !!! warning
                It is important to never hard-code secrets into your code, therefore we read the OpenAI API key from an environment variable.

            !!! note
                It is important to never hard-code secrets into your code, therefore we read the OpenAI API key from an environment variable.

            <img src="https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/logo/banner-badge.png?raw=true" style={ width: '40%' } alt="AgentOps logo"/>

            """).lstrip()

        with open(mkdocs_output_dir / "home" / "quick-start.md") as f:
            actual_quick_start_content = f.read()

        assert actual_quick_start_content == expected_quick_start_content


def test_transform_tab_component() -> None:
    content = dedent("""This is a sample quick start page.
<Tabs>
    <Tab title="Chat with an agent">
```python
# 1. Import our agent class
from autogen import ConversableAgent

# 2. Define our LLM configuration for OpenAI's GPT-4o mini
#    uses the OPENAI_API_KEY environment variable
llm_config = {"api_type": "openai", "model": "gpt-4o-mini"}

# 3. Create our LLM agent
my_agent = ConversableAgent(
    name="helpful_agent",
    llm_config=llm_config,
    system_message="You are a poetic AI assistant, respond in rhyme.",
)

# 4. Run the agent with a prompt
chat_result = my_agent.run("In one sentence, what's the big deal about AI?")

# 5. Print the chat
print(chat_result.chat_history)
```
    </Tab>
    <Tab title="Two agent chat">
    example code
```python
llm_config = {"api_type": "openai", "model": "gpt-4o-mini"}
```


    </Tab>
</Tabs>

Some conclusion
""")

    expected = dedent("""This is a sample quick start page.
=== "Chat with an agent"
    ```python
    # 1. Import our agent class
    from autogen import ConversableAgent

    # 2. Define our LLM configuration for OpenAI's GPT-4o mini
    #    uses the OPENAI_API_KEY environment variable
    llm_config = {"api_type": "openai", "model": "gpt-4o-mini"}

    # 3. Create our LLM agent
    my_agent = ConversableAgent(
        name="helpful_agent",
        llm_config=llm_config,
        system_message="You are a poetic AI assistant, respond in rhyme.",
    )

    # 4. Run the agent with a prompt
    chat_result = my_agent.run("In one sentence, what's the big deal about AI?")

    # 5. Print the chat
    print(chat_result.chat_history)
    ```

=== "Two agent chat"
    example code
    ```python
    llm_config = {"api_type": "openai", "model": "gpt-4o-mini"}
    ```

Some conclusion
""")
    actual = transform_tab_component(content)
    assert actual == expected


def test_transform_card_grp_component() -> None:
    content = dedent("""This is a sample quick start page.
        <div class="popular-resources">
            <div class="card-group not-prose grid gap-x-4 sm:grid-cols-2">
                <CardGroup cols={2}>
                <Card>
                    <p>Hello World</p>
                </Card>
                <Card title="Quick Start" href="/docs/home/quick-start">
                    <p>Hello World</p>
                </Card>
                </CardGroup>
            </div>
        </div>
        """)

    expected = dedent("""This is a sample quick start page.
        <div class="popular-resources">
            <div class="card-group not-prose grid gap-x-4 sm:grid-cols-2">
                <div class="card">
                    <p>Hello World</p>
                </div>
                <a class="card" href="/docs/home/quick-start">
<h2>Quick Start</h2>
                    <p>Hello World</p>
                </a>
            </div>
        </div>
        """)
    actual = transform_card_grp_component(content)
    assert actual == expected


def test_fix_asset_path() -> None:
    content = dedent("""This is a sample quick start page.
<div class="key-feature">
    <img noZoom src="/static/img/conv_2.svg" alt="Multi-Agent Conversation Framework" />
    <a class="hero-btn" href="/docs/home/quick-start">
        <div>Getting Started - 3 Minute</div>
    </a>
</div>""")
    expected = dedent("""This is a sample quick start page.
<div class="key-feature">
    <img noZoom src="/assets/img/conv_2.svg" alt="Multi-Agent Conversation Framework" />
    <a class="hero-btn" href="/docs/home/quick-start">
        <div>Getting Started - 3 Minute</div>
    </a>
</div>""")
    actual = fix_asset_path(content)
    assert actual == expected


@pytest.fixture
def navigation() -> list[NavigationGroup]:
    return [
        {"group": "Home", "pages": ["docs/home/home", "docs/home/quick-start"]},
        {
            "group": "User Guide",
            "pages": [
                {
                    "group": "Basic Concepts",
                    "pages": [
                        "docs/user-guide/basic-concepts/installing-ag2",
                        {
                            "group": "LLM Configuration",
                            "pages": [
                                "docs/user-guide/basic-concepts/llm-configuration/llm-configuration",
                                "docs/user-guide/basic-concepts/llm-configuration/structured-outputs",
                            ],
                        },
                        "docs/user-guide/basic-concepts/conversable-agent",
                        "docs/user-guide/basic-concepts/human-in-the-loop",
                        {
                            "group": "Orchestrating Agents",
                            "pages": [
                                "docs/user-guide/basic-concepts/orchestration/orchestrations",
                                "docs/user-guide/basic-concepts/orchestration/sequential-chat",
                            ],
                        },
                    ],
                },
                {"group": "Advanced Concepts", "pages": ["docs/user-guide/advanced-concepts/rag"]},
            ],
        },
        {
            "group": "Contributor Guide",
            "pages": [
                "docs/contributing/contributing",
            ],
        },
    ]


@pytest.fixture
def expected_nav() -> str:
    return """- [Home](index.md)
    - [Home](docs/home/home.md)
    - [Quick Start](docs/home/quick-start.md)
- User Guide
    - Basic Concepts
        - [Installing AG2](docs/user-guide/basic-concepts/installing-ag2.md)
        - LLM Configuration
            - [LLM Configuration](docs/user-guide/basic-concepts/llm-configuration/llm-configuration.md)
            - [Structured Outputs](docs/user-guide/basic-concepts/llm-configuration/structured-outputs.md)
        - [Conversable Agent](docs/user-guide/basic-concepts/conversable-agent.md)
        - [Human In The Loop](docs/user-guide/basic-concepts/human-in-the-loop.md)
        - Orchestrating Agents
            - [Orchestrations](docs/user-guide/basic-concepts/orchestration/orchestrations.md)
            - [Sequential Chat](docs/user-guide/basic-concepts/orchestration/sequential-chat.md)
    - Advanced Concepts
        - [RAG](docs/user-guide/advanced-concepts/rag.md)
- Contributor Guide
    - [Contributing](docs/contributing/contributing.md)"""


def test_format_navigation(navigation: list[NavigationGroup], expected_nav: str) -> None:
    actual = format_navigation(navigation)
    assert actual == expected_nav


def test_add_api_ref_to_mkdocs_template() -> None:
    mkdocs_nav = """- Home
    - [Home](docs/home/home.md)
- User Guide
    - Basic Concepts
        - [Installing AG2](docs/user-guide/basic-concepts/installing-ag2.md)
        - LLM Configuration
            - [LLM Configuration](docs/user-guide/basic-concepts/llm-configuration/llm-configuration.md)
        - [Websurferagent](docs/user-guide/reference-agents/websurferagent.md)
- Contributor Guide
    - [Contributing](docs/contributor-guide/contributing.md)
"""

    expected = """- Home
    - [Home](docs/home/home.md)
- User Guide
    - Basic Concepts
        - [Installing AG2](docs/user-guide/basic-concepts/installing-ag2.md)
        - LLM Configuration
            - [LLM Configuration](docs/user-guide/basic-concepts/llm-configuration/llm-configuration.md)
        - [Websurferagent](docs/user-guide/reference-agents/websurferagent.md)
- API References
{api}
- Contributor Guide
    - [Contributing](docs/contributor-guide/contributing.md)
"""
    section_to_follow = "Contributor Guide"
    actual = add_api_ref_to_mkdocs_template(mkdocs_nav, section_to_follow)
    assert actual == expected


@run_for_optional_imports(["jinja2"], "docs")
def test_generate_mkdocs_navigation(navigation: list[NavigationGroup], expected_nav: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source directory structure
        website_dir = Path(tmpdir) / "website_root"
        website_dir.mkdir()

        # Create mkdocs directory
        mkdocs_root_dir = Path(tmpdir) / "mkdocs_root"
        mkdocs_root_dir.mkdir()

        mintlify_nav_template_path = website_dir / "mint-json-template.json.jinja"
        mkdocs_nav_path = mkdocs_root_dir / "docs" / "navigation_template.txt"
        mkdocs_nav_path.parent.mkdir(parents=True, exist_ok=True)
        mkdocs_nav_path.touch()

        summary_md_path = mkdocs_root_dir / "docs" / "SUMMARY.md"

        mintlify_nav_content = (
            """
        {
  "$schema": "https://mintlify.com/schema.json",
  "name": "AG2",
    "navigation": """
            + json.dumps(navigation)
            + """ }"""
        )

        mintlify_nav_template_path.write_text(mintlify_nav_content)

        nav_exclusions = ["Contributor Guide"]
        generate_mkdocs_navigation(website_dir, mkdocs_root_dir, nav_exclusions)
        actual = mkdocs_nav_path.read_text()
        expected = (
            """---
search:
  exclude: true
---
"""
            + expected_nav.replace(
                """
- Contributor Guide
    - [Contributing](docs/contributing/contributing.md)""",
                "",
            )
            + "\n"
        )
        assert actual == expected
        assert summary_md_path.read_text() == expected
