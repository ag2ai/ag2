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
    add_authors_info_to_user_stories,
    add_excerpt_marker,
    add_notebooks_nav,
    filter_excluded_files,
    fix_asset_path,
    fix_snippet_imports,
    format_navigation,
    generate_mkdocs_navigation,
    generate_url_slug,
    generate_user_stories_nav,
    process_and_copy_files,
    process_blog_contents,
    process_blog_files,
    remove_mdx_code_blocks,
    transform_admonition_blocks,
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


def test_process_blog_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source directory structure

        mkdocs_dir = Path(tmpdir) / "mkdocs"
        mkdocs_dir.mkdir()

        src_dir = mkdocs_dir / "_blogs"
        src_dir.mkdir()

        # Create snippets directory
        snippets_dir = mkdocs_dir / "snippets"
        snippets_dir.mkdir()

        files = [
            src_dir / "2023-04-21-LLM-tuning-math" / "index.md",
            src_dir / "2023-04-21-LLM-tuning-math" / "cover.jpg",
            src_dir / "2023-04-21-LLM-tuning-math" / "cover.png",
            snippets_dir / "2023-04-21-LLM-tuning-math" / "index.md",
            snippets_dir / "2023-04-21-LLM-tuning-math" / "cover.jpg",
            snippets_dir / "2023-04-21-LLM-tuning-math" / "cover.png",
        ]

        # Create the files
        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()

        target_blog_dir = mkdocs_dir / "blog"

        authors_yml_path = mkdocs_dir / ".authors.yml"
        authors_yml_path.touch()

        # Assert the target_blog_dir should have posts directory and index.md file and .authors.yml file
        process_blog_files(mkdocs_dir, Path(authors_yml_path), snippets_dir)

        actual = list(filter(lambda x: x.is_file(), target_blog_dir.rglob("*")))
        expected = [
            target_blog_dir / "posts" / "2023-04-21-LLM-tuning-math" / "index.md",
            target_blog_dir / "posts" / "2023-04-21-LLM-tuning-math" / "cover.jpg",
            target_blog_dir / "posts" / "2023-04-21-LLM-tuning-math" / "cover.png",
            target_blog_dir / "index.md",
            target_blog_dir / ".authors.yml",
        ]
        assert len(actual) == len(expected)
        assert sorted(actual) == sorted(expected)

        target_snippet_dir = Path(tmpdir) / "snippets"
        assert target_snippet_dir.exists()
        assert len(list(filter(lambda x: x.is_file(), target_snippet_dir.rglob("*")))) == 3


def test_process_blog_contents() -> None:
    contents = dedent("""
    ---
    title: Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH
    authors: [sonichi]
    tags: [LLM, GPT, research]
    ---

    ![level 2 algebra](img/level2algebra.png)

    **TL;DR:**
    """)

    expected = "---" + dedent("""
    title: Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH
    authors: [sonichi]
    tags:
        - LLM
        - GPT
        - research
    categories:
        - LLM
        - GPT
        - research
    date: 2025-01-10
    slug: WebSockets
    ---

    ![level 2 algebra](img/level2algebra.png)

    **TL;DR:**

    <!-- more -->
    """)
    file = Path("tmp/ag2/ag2/website/mkdocs/docs/docs/_blogs/2025-01-10-WebSockets/index.md")
    actual = process_blog_contents(contents, file)
    assert actual == expected


def test_add_excerpt_marker() -> None:
    content = dedent("""
    ## Welcome DiscordAgent, SlackAgent, and TelegramAgent

    We want to help you focus on building workflows and enhancing agents

    ### New agents need new tools

    some content

    """)
    expected = dedent("""
    ## Welcome DiscordAgent, SlackAgent, and TelegramAgent

    We want to help you focus on building workflows and enhancing agents


    <!-- more -->

    ### New agents need new tools

    some content

    """)
    actual = add_excerpt_marker(content)
    assert actual == expected

    content = dedent("""
    ## Welcome DiscordAgent, SlackAgent, and TelegramAgent

    We want to help you focus on building workflows and enhancing agents

    """)
    expected = dedent("""
    ## Welcome DiscordAgent, SlackAgent, and TelegramAgent

    We want to help you focus on building workflows and enhancing agents

    <!-- more -->
    """)
    actual = add_excerpt_marker(content)
    assert actual == expected

    content = dedent(r"""
    ## Welcome DiscordAgent, SlackAgent, and TelegramAgent

    \<!-- more -->

    We want to help you focus on building workflows and enhancing agents

    ### New agents need new tools

    some content

    """)
    expected = dedent("""
    ## Welcome DiscordAgent, SlackAgent, and TelegramAgent

    <!-- more -->

    We want to help you focus on building workflows and enhancing agents

    ### New agents need new tools

    some content

    """)
    actual = add_excerpt_marker(content)
    assert actual == expected


def test_fix_snippet_imports() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_snippets_dir = Path(tmpdir) / "snippets"
        expected_path = "{!" + tmp_snippets_dir.as_posix() + "/reference-agents/deep-research.mdx" + " !}"

        content = dedent("""
        ## Introduction

        import DeepResearch from "/snippets/reference-agents/deep-research.mdx";

        <DeepResearch/>

        ## Conclusion
        """)
        expected = dedent(f"""
        ## Introduction

        {expected_path}


        <DeepResearch/>

        ## Conclusion
        """)

        actual = fix_snippet_imports(content, tmp_snippets_dir)
        assert actual == expected

        content = dedent("""
        ## Introduction

        import DeepResearch from "/some-other-dir/reference-agents/deep-research.mdx";

        <DeepResearch/>

        ## Conclusion
        """)
        expected = dedent("""
        ## Introduction

        import DeepResearch from "/some-other-dir/reference-agents/deep-research.mdx";

        <DeepResearch/>

        ## Conclusion
        """)

        actual = fix_snippet_imports(content, tmp_snippets_dir)
        assert actual == expected


def test_generate_url_slug() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        blog_dir = Path(tmpdir) / "2025-02-13-DeepResearchAgent"
        blog_dir.mkdir(parents=True, exist_ok=True)

        tmpfile = blog_dir / "somefile.txt"
        tmpfile.touch()

        actual = generate_url_slug(tmpfile)
        expected = "\nslug: DeepResearchAgent"

        assert actual == expected


def test_add_notebooks_nav() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create source directory structure
        metadata_yml_path = Path(tmpdir) / "notebooks_metadata.yml"

        # Add content
        metadata_yml_path.write_text(
            dedent("""
- title: "Run a standalone AssistantAgent"
  link: "/docs/use-cases/notebooks/notebooks/agentchat_assistant_agent_standalone"
  description: "Run a standalone AssistantAgent, browsing the web using the BrowserUseTool"
  image: ""
  tags:
    - "assistantagent"
    - "run"
    - "browser-use"
    - "webscraping"
    - "function calling"
  source: "/notebook/agentchat_assistant_agent_standalone.ipynb"

- title: "Mitigating Prompt hacking with JSON Mode in Autogen"
  link: "/docs/use-cases/notebooks/notebooks/JSON_mode_example"
  description: "Use JSON mode and Agent Descriptions to mitigate prompt manipulation and control speaker transition."
  image: ""
  tags:
    - "JSON"
    - "description"
    - "prompt hacking"
    - "group chat"
    - "orchestration"
  source: "/notebook/JSON_mode_example.ipynb"
""")
        )

        mkdocs_nav_path = Path(tmpdir) / "navigation_template.txt"

        mkdocs_nav_path.write_text(
            dedent("""
- Use Cases
    - Use cases
        - [Customer Service](docs/use-cases/use-cases/customer-service.md)
        - [Game Design](docs/use-cases/use-cases/game-design.md)
        - [Travel Planning](docs/use-cases/use-cases/travel-planning.md)
    - Notebooks
        - [Notebooks](docs/use-cases/notebooks/Notebooks.md)
    - [Community Gallery](docs/use-cases/community-gallery/community-gallery.md)
- API References
{api}
""")
        )

        add_notebooks_nav(mkdocs_nav_path, metadata_yml_path)

        expected = dedent("""
- Use Cases
    - Use cases
        - [Customer Service](docs/use-cases/use-cases/customer-service.md)
        - [Game Design](docs/use-cases/use-cases/game-design.md)
        - [Travel Planning](docs/use-cases/use-cases/travel-planning.md)
    - Notebooks
        - [Notebooks](docs/use-cases/notebooks/Notebooks.md)
        - [Run a standalone AssistantAgent](docs/use-cases/notebooks/notebooks/agentchat_assistant_agent_standalone)
        - [Mitigating Prompt hacking with JSON Mode in Autogen](docs/use-cases/notebooks/notebooks/JSON_mode_example)
    - [Community Gallery](docs/use-cases/community-gallery/community-gallery.md)
- API References
{api}
""")

        actual = mkdocs_nav_path.read_text()
        assert actual == expected


def test_generate_user_stories_nav() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create User Stories directory
        user_stories_dir = Path(tmpdir) / "docs" / "user-stories"

        file_1 = user_stories_dir / "2025-03-11-NOVA" / "index.md"
        file_1.parent.mkdir(parents=True, exist_ok=True)
        file_1.write_text("""---
title: Unlocking the Power of Agentic Workflows at Nexla with AG2
authors:
  - sonichi
  - qingyunwu
tags: [data automation, agents, AG2, Nexla]
---

> AG2 has been instrumental in helping Nexla build NOVA,
""")

        file_2 = user_stories_dir / "2025-02-11-NOVA" / "index.md"
        file_2.parent.mkdir(parents=True, exist_ok=True)
        file_2.write_text("""---
title: Some other text
authors:
  - sonichi
  - qingyunwu
tags: [data automation, agents, AG2, Nexla]
---

> AG2 has been instrumental in helping Nexla build NOVA,
""")

        # Create source directory structure
        mkdocs_nav_path = Path(tmpdir) / "navigation_template.txt"

        mkdocs_nav_path.write_text(
            dedent("""
- Use Cases
    - Use cases
        - [Customer Service](docs/use-cases/use-cases/customer-service.md)
        - [Game Design](docs/use-cases/use-cases/game-design.md)
        - [Travel Planning](docs/use-cases/use-cases/travel-planning.md)
    - Notebooks
        - [Notebooks](docs/use-cases/notebooks/Notebooks.md)
    - [Community Gallery](docs/use-cases/community-gallery/community-gallery.md)
- Contributor Guide
    - [Contributing](docs/contributor-guide/contributing.md)
    - [Setup Development Environment](docs/contributor-guide/setup-development-environment.md)
""")
        )

        generate_user_stories_nav(Path(tmpdir), mkdocs_nav_path)

        expected = dedent("""
- Use Cases
    - Use cases
        - [Customer Service](docs/use-cases/use-cases/customer-service.md)
        - [Game Design](docs/use-cases/use-cases/game-design.md)
        - [Travel Planning](docs/use-cases/use-cases/travel-planning.md)
    - Notebooks
        - [Notebooks](docs/use-cases/notebooks/Notebooks.md)
    - [Community Gallery](docs/use-cases/community-gallery/community-gallery.md)
- User Stories
    - [Unlocking the Power of Agentic Workflows at Nexla with AG2](docs/user-stories/2025-03-11-NOVA)
    - [Some other text](docs/user-stories/2025-02-11-NOVA)
- Contributor Guide
    - [Contributing](docs/contributor-guide/contributing.md)
    - [Setup Development Environment](docs/contributor-guide/setup-development-environment.md)
""")

        actual = mkdocs_nav_path.read_text()
        assert actual == expected


def test_add_authors_info_to_user_stories() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create User Stories directory
        mkdocs_output_dir = Path(tmpdir) / "mkdocs" / "docs" / "docs"
        user_stories_dir = mkdocs_output_dir / "user-stories"

        file_1 = user_stories_dir / "2025-03-11-NOVA" / "index.md"
        file_1.parent.mkdir(parents=True, exist_ok=True)
        file_1.write_text("""---
title: Unlocking the Power of Agentic Workflows at Nexla with AG2
authors:
  - sonichi
  - qingyunwu
tags: [data automation, agents, AG2, Nexla]
---

> AG2 has been instrumental in helping Nexla build NOVA,
""")

        file_2 = user_stories_dir / "2025-02-11-NOVA" / "index.md"
        file_2.parent.mkdir(parents=True, exist_ok=True)
        file_2.write_text("""---
title: Some other text
authors:
  - sonichi
  - qingyunwu
tags: [data automation, agents, AG2, Nexla]
---

> AG2 has been instrumental in helping Nexla build NOVA,
""")

        authors_yml = Path(tmpdir) / "blogs_and_user_stories_authors.yml"

        authors_yml.write_text("""
authors:
  sonichi:
    name: Chi Wang
    description: Founder of AutoGen (now AG2) & FLAML
    url: https://www.linkedin.com/in/chi-wang-autogen/
    avatar: https://github.com/sonichi.png

  qingyunwu:
    name: Qingyun Wu
    description: Co-Founder of AutoGen/AG2 & FLAML, Assistant Professor at Penn State University
    url: https://qingyun-wu.github.io/
    avatar: https://github.com/qingyun-wu.png
""")

        add_authors_info_to_user_stories(Path(tmpdir))

        actual = file_1.read_text()
        expected = dedent("""---
title: Unlocking the Power of Agentic Workflows at Nexla with AG2
authors:
  - sonichi
  - qingyunwu
tags: [data automation, agents, AG2, Nexla]
---

<div class="blog-authors">
<p class="authors">Authors:</p>
<div class="card-group">

<div class="card">
    <div class="col card">
      <div class="img-placeholder">
        <img noZoom src="https://github.com/sonichi.png" />
      </div>
      <div>
        <p class="name">Chi Wang</p>
        <p>Founder of AutoGen (now AG2) & FLAML</p>
      </div>
    </div>
</div>

<div class="card">
    <div class="col card">
      <div class="img-placeholder">
        <img noZoom src="https://github.com/qingyun-wu.png" />
      </div>
      <div>
        <p class="name">Qingyun Wu</p>
        <p>Co-Founder of AutoGen/AG2 & FLAML, Assistant Professor at Penn State University</p>
      </div>
    </div>
</div>

</div>
</div>

> AG2 has been instrumental in helping Nexla build NOVA,
""")
        assert actual == expected


class TestRemoveMdxCodeBlocks:
    def test_simple_example(self) -> None:
        """Test with the example provided in the requirements."""
        input_content = """````mdx-code-block
!!! info "Requirements"
    Some extra dependencies are needed for this notebook, which can be installed via pip:

    ```bash
    pip install pyautogen[openai,lmm]
    ```

    For more information, please refer to the [installation guide](/docs/user-guide/basic-concepts/installing-ag2).
````"""

        expected_output = """!!! info "Requirements"
    Some extra dependencies are needed for this notebook, which can be installed via pip:

    ```bash
    pip install pyautogen[openai,lmm]
    ```

    For more information, please refer to the [installation guide](/docs/user-guide/basic-concepts/installing-ag2)."""

        assert remove_mdx_code_blocks(input_content) == expected_output

    def test_multiple_blocks(self) -> None:
        """Test with multiple mdx-code-blocks in the content."""
        input_content = """Some text before

````mdx-code-block
!!! note
    This is a note
````

Some text in between

````mdx-code-block
!!! warning
    This is a warning
````

Some text after"""

        expected_output = """Some text before

!!! note
    This is a note

Some text in between

!!! warning
    This is a warning

Some text after"""

        assert remove_mdx_code_blocks(input_content) == expected_output

    def test_no_mdx_blocks(self) -> None:
        """Test with content that doesn't have any mdx-code-blocks."""
        input_content = """# Regular Markdown

This is some regular markdown content.

```python
def regular_code():
    return "not inside mdx-code-block"
```"""

        assert remove_mdx_code_blocks(input_content) == input_content


class TestTransformAdmonitionBlocks:
    def test_basic_admonition(self) -> None:
        """Test basic admonition block without a title."""
        content = """
Some text before

:::note
This is a simple note.
:::

Some text after
"""
        expected = """
Some text before

!!! note
    This is a simple note.

Some text after
"""
        actual = transform_admonition_blocks(content)
        assert actual == expected

    def test_admonition_with_title(self) -> None:
        """Test admonition block with a title."""
        content = """
:::warning Important Alert
This is a warning with a title.
:::
"""
        expected = """
!!! warning "Important Alert"
    This is a warning with a title.
"""
        actual = transform_admonition_blocks(content)
        assert actual == expected

    def test_multiple_admonitions(self) -> None:
        """Test multiple admonition blocks in the same content."""
        content = """
:::tip
Tip content
:::

Some text in between

:::danger Caution
Danger content
:::
"""
        expected = """
!!! tip
    Tip content

Some text in between

!!! danger "Caution"
    Danger content
"""
        actual = transform_admonition_blocks(content)
        assert actual == expected

    def test_admonition_with_multiline_content(self) -> None:
        """Test admonition with multiple lines of content."""
        content = """
:::note
Line 1
Line 2
Line 3
:::
"""
        expected = """
!!! note
    Line 1
    Line 2
    Line 3
"""
        assert transform_admonition_blocks(content) == expected

    def test_admonition_with_indented_content(self) -> None:
        """Test admonition with indented content."""
        content = """
:::note
    This line is indented.
        This line is more indented.
    Back to first level.
:::
"""
        expected = """
!!! note
    This line is indented.
        This line is more indented.
    Back to first level.
"""
        assert transform_admonition_blocks(content) == expected

    def test_admonition_with_code_block(self) -> None:
        """Test admonition containing a code block."""
        content = """
:::tip Code Example
Here's some code:

```python
def hello():
    print("Hello world")
```
:::
"""
        expected = """
!!! tip "Code Example"
    Here's some code:

    ```python
    def hello():
        print("Hello world")
    ```
"""
        actual = transform_admonition_blocks(content)
        assert actual == expected

    def test_admonition_with_lists(self) -> None:
        """Test admonition containing lists."""
        content = """
:::note
- Item 1
- Item 2
  - Nested item
- Item 3
:::
"""
        expected = """
!!! note
    - Item 1
    - Item 2
      - Nested item
    - Item 3
"""
        assert transform_admonition_blocks(content) == expected

    def test_admonition_with_blockquotes(self) -> None:
        """Test admonition containing blockquotes."""
        content = """
:::info
Here's a quote:

> This is a blockquote
> Multiple lines
:::
"""
        expected = """
!!! info
    Here's a quote:

    > This is a blockquote
    > Multiple lines
"""
        actual = transform_admonition_blocks(content)
        assert actual == expected

    def test_admonition_type_mapping(self) -> None:
        """Test mapping of admonition types."""
        content = """
:::Tip
This should map to lowercase 'tip'
:::

:::warning
This should stay as 'warning'
:::

:::custom
This should stay as 'custom'
:::
"""
        expected = """
!!! tip
    This should map to lowercase 'tip'

!!! warning
    This should stay as 'warning'

!!! custom
    This should stay as 'custom'
"""
        assert transform_admonition_blocks(content) == expected

    def test_invalid_syntax_admonition(self) -> None:
        """Test that original content is preserved for malformed admonition syntax."""
        content = """
    Some text before

    :::
    This is missing a type specifier
    :::

    Some text after
    """
        # The output should be identical to the input
        assert transform_admonition_blocks(content) == content


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
            + "\n- Blog\n    - [Blog](docs/blog)"
            + "\n"
        )

        assert actual == expected
        assert summary_md_path.read_text() == expected
