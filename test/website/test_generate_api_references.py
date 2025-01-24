# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Add the ../../website directory to sys.path
website_path = Path(__file__).resolve().parents[2] / "website"
assert website_path.exists()
assert website_path.is_dir()
sys.path.append(str(website_path))

from generate_api_references import SplitReferenceFilesBySymbols, generate_mint_json_from_template


@pytest.fixture
def api_dir(tmp_path: Path) -> Path:
    """Helper function to create test directory structure"""
    # Create autogen directory
    autogen_dir = tmp_path / "autogen"
    autogen_dir.mkdir()

    # Create some files in autogen
    (autogen_dir / "browser_utils.md").write_text("browser_utils")
    (autogen_dir / "code_utils.md").write_text("code_utils")
    (autogen_dir / "version.md").write_text("version")

    # Create subdirectories with files
    subdir1 = autogen_dir / "agentchat"
    subdir1.mkdir()
    (subdir1 / "assistant_agent.md").write_text("assistant_agent")
    (subdir1 / "conversable_agent.md").write_text("conversable_agent")
    (subdir1 / "agentchat.md").write_text("agentchat")
    (subdir1 / "index.md").write_text("index")

    # nested subdirectory
    nested_subdir = subdir1 / "contrib"
    nested_subdir.mkdir()
    (nested_subdir / "agent_builder.md").write_text("agent_builder")
    (nested_subdir / "index.md").write_text("index")

    subdir2 = autogen_dir / "cache"
    subdir2.mkdir()
    (subdir2 / "cache_factory.md").write_text("cache_factory")
    (subdir2 / "disk_cache.md").write_text("disk_cache")
    (subdir2 / "index.md").write_text("index")

    return tmp_path


@pytest.fixture
def template_content() -> str:
    """Fixture providing the template JSON content."""
    template = """
    {
        "name": "AG2",
        "logo": {"dark": "/logo/ag2-white.svg", "light": "/logo/ag2.svg"},
        "navigation": [
            {"group": "", "pages": ["docs/Home", "docs/Getting-Started"]},
            {
                "group": "Installation",
                "pages": [
                    "docs/installation/Installation",
                    "docs/installation/Docker",
                    "docs/installation/Optional-Dependencies"
                ]
            },
            {"group": "API Reference", "pages": ["PLACEHOLDER"]}
        ]
    }
    """
    return template


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Fixture providing a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def template_file(temp_dir: Path, template_content: str) -> Path:
    """Fixture creating a template file in a temporary directory."""
    template_path = temp_dir / "mint-json-template.json.jinja"
    with open(template_path, "w") as f:
        f.write(template_content)
    return template_path


@pytest.fixture
def target_file(temp_dir: Path) -> Path:
    """Fixture providing the target mint.json path."""
    return temp_dir / "mint.json"


def test_generate_mint_json_from_template(template_file: Path, target_file: Path, template_content: str) -> None:
    """Test that mint.json is generated correctly from template."""
    # Run the function
    generate_mint_json_from_template(template_file, target_file)

    # Verify the file exists
    assert target_file.exists()

    # Verify the contents
    with open(target_file) as f:
        actual = json.load(f)

    expected = json.loads(template_content)
    assert actual == expected


def test_generate_mint_json_existing_file(template_file: Path, target_file: Path, template_content: str) -> None:
    """Test that function works when mint.json already exists."""
    # Create an existing mint.json with different content
    existing_content = {"name": "existing"}
    with open(target_file, "w") as f:
        json.dump(existing_content, f)

    # Run the function
    generate_mint_json_from_template(template_file, target_file)

    # Verify the contents were overwritten
    with open(target_file) as f:
        actual = json.load(f)

    expected = json.loads(template_content)
    assert actual == expected


def test_generate_mint_json_missing_template(target_file: Path) -> None:
    """Test handling of missing template file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        nonexistent_template = Path(tmp_dir) / "nonexistent.template"
        with pytest.raises(FileNotFoundError):
            generate_mint_json_from_template(nonexistent_template, target_file)


class TestSplitReferenceFilesBySymbol:
    AUTOGEN_INDEX_CONTENT = """---
sidebarTitle: autogen
title: autogen
---

**** SUBMODULE_START ****

## Sub-modules

* autogen.agentchat
* autogen.browser_utils

**** SUBMODULE_END ****

## Functions

**** SYMBOL_START ****

<code class="doc-symbol doc-symbol-heading doc-symbol-function"></code>
#### gather_usage_summary
<a href="#autogen..gather_usage_summary" class="headerlink" title="Permanent link"></a>

```python
gather_usage_summary(agents: list[autogen.Agent]) -> dict[dict[str, dict], dict[str, dict]]
```

    Gather usage summary from all agents.

<b>Parameters:</b>
| Name | Description |
|--|--|
| `agents` | (list): List of agents.<br/><br/>**Type:** `list[autogen.Agent]` |

<b>Returns:</b>
| Type | Description |
|--|--|
| `dict[dict[str, dict], dict[str, dict]]` | dictionary: A dictionary containing two keys: - "usage_including_cached_inference": Cost information on the total usage, including the tokens in cached inference. - "usage_excluding_cached_inference": Cost information on the usage of tokens, excluding the tokens in cache. No larger than "usage_including_cached_inference". Example: ```python \\{ "usage_including_cached_inference": \\{ "total_cost": 0.0006090000000000001, "gpt-35-turbo": \\{ "cost": 0.0006090000000000001, "prompt_tokens": 242, "completion_tokens": 123, "total_tokens": 365, }, }, "usage_excluding_cached_inference": \\{ "total_cost": 0.0006090000000000001, "gpt-35-turbo": \\{ "cost": 0.0006090000000000001, "prompt_tokens": 242, "completion_tokens": 123, "total_tokens": 365, }, }, } ``` |

<br />

**** SYMBOL_END ****
**** SYMBOL_START ****

<code class="doc-symbol doc-symbol-heading doc-symbol-function"></code>
#### config_list_from_dotenv
<a href="#autogen..config_list_from_dotenv" class="headerlink" title="Permanent link"></a>

```python
config_list_from_dotenv(
    dotenv_file_path: str | None = None,
) -> list[dict[str, str | set[str]]]
```

    Load API configurations

<b>Parameters:</b>
| Name | Description |
|--|--|
| `dotenv_file_path` | The path to the .env file.<br/><br/>Defaults to None.<br/><br/>**Type:** `str \\| None`<br/><br/>**Default:** None |

<b>Returns:</b>
| Type | Description |
|--|--|
| `list[dict[str, str \\| set[str]]]` | List[Dict[str, Union[str, Set[str]]]]: A list of configuration dictionaries for each model. |

<br />

**** SYMBOL_END ****
**** SYMBOL_START ****

    <h2 id="autogen.ConversableAgent" class="doc doc-heading">
        <code class="doc-symbol doc-symbol-heading doc-symbol-class"></code>
        <span class="doc doc-object-name doc-class-name">ConversableAgent</span>
        <a href="#autogen.ConversableAgent" class="headerlink" title="Permanent link"></a>
    </h2>

`autogen.ConversableAgent`

```python
ConversableAgent(
    name: str,
    system_message: str | list | None = 'You are a helpful AI Assistant.',
    is_termination_msg: Callable[[dict], bool] | None = None,
)
```

    A class for generic conversable agents which can be configured as assistant or user proxy.

<b>Parameters:</b>
| Name | Description |
|--|--|
| `name` | name of the agent.<br/><br/>**Type:** `str` |
| `system_message` | system message for the ChatCompletion inference.<br/><br/>**Type:** `str \\| list \\| None`<br/><br/>**Default:** 'You are a helpful AI Assistant.' |
| `is_termination_msg` | a function that takes a message in the form of a dictionary and returns a boolean value indicating if this received message is a termination message.<br/><br/>The dict can contain the following keys: "content", "role", "name", "function_call".<br/><br/>**Type:** `Callable[[dict], bool] \\| None`<br/><br/>**Default:** None |

### Class Attributes

<code class="doc-symbol doc-symbol-heading doc-symbol-attribute"></code>
#### DEFAULT_CONFIG
<a href="#autogen.ConversableAgent.DEFAULT_CONFIG" class="headerlink" title="Permanent link"></a>

    <br />

<code class="doc-symbol doc-symbol-heading doc-symbol-attribute"></code>
#### DEFAULT_SUMMARY_METHOD
<a href="#autogen.ConversableAgent.DEFAULT_SUMMARY_METHOD" class="headerlink" title="Permanent link"></a>

    <br />

### Instance Attributes

<code class="doc-symbol doc-symbol-heading doc-symbol-attribute"></code>
#### chat_messages
<a href="#autogen.ConversableAgent.chat_messages" class="headerlink" title="Permanent link"></a>

    A dictionary of conversations from agent to list of messages.

<code class="doc-symbol doc-symbol-heading doc-symbol-attribute"></code>
#### code_executor
<a href="#autogen.ConversableAgent.code_executor" class="headerlink" title="Permanent link"></a>

    The code executor used by this agent. Returns None if code execution is disabled.

### Instance Methods

<code class="doc-symbol doc-symbol-heading doc-symbol-method"></code>
#### a_check_termination_and_human_reply
<a href="#autogen.ConversableAgent.a_check_termination_and_human_reply" class="headerlink" title="Permanent link"></a>

```python
a_check_termination_and_human_reply(
    self,
    messages: list[dict] | None = None,
) -> tuple[bool, str | None]
```

    Check if the conversation should be terminated, and if human reply is provided.

<b>Parameters:</b>
| Name | Description |
|--|--|
| `messages` | A list of message dictionaries, representing the conversation history.<br/><br/>**Type:** `list[dict] \\| None`<br/><br/>**Default:** None |

<b>Returns:</b>
| Type | Description |
|--|--|
| `tuple[bool, str \\| None]` | Tuple[bool, Union[str, Dict, None]]: A tuple containing a boolean indicating if the conversation should be terminated, and a human reply which can be a string, a dictionary, or None. |

<br />

<code class="doc-symbol doc-symbol-heading doc-symbol-method"></code>
#### a_execute_function
<a href="#autogen.ConversableAgent.a_execute_function" class="headerlink" title="Permanent link"></a>

```python
a_execute_function(
    self,
    func_call,
) -> tuple[bool, dict[str, typing.Any]]
```

    Execute an async function call and return the result.

<b>Parameters:</b>
| Name | Description |
|--|--|
| `func_call` | a dictionary extracted from openai message at key "function_call" or "tool_calls" with keys "name" and "arguments".<br/><br/> |

<b>Returns:</b>
| Type | Description |
|--|--|
| `tuple[bool, dict[str, typing.Any]]` | A tuple of (is_exec_success, result_dict). is_exec_success (boolean): whether the execution is successful. result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function". "function_call" deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0) See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call |

<br />

<code class="doc-symbol doc-symbol-heading doc-symbol-method"></code>
#### a_generate_function_call_reply
<a href="#autogen.ConversableAgent.a_generate_function_call_reply" class="headerlink" title="Permanent link"></a>

```python
a_generate_function_call_reply(
    self,
    messages: list[dict] | None = None,
) -> tuple[bool, dict | None]
```

    Generate a reply using async function call.

<b>Parameters:</b>
| Name | Description |
|--|--|
| `messages` | **Type:** `list[dict] \\| None`<br/><br/>**Default:** None |

<br />
**** SYMBOL_END ****
    """

    AGENTCHAT_INDEX_CONTENT = """
---
sidebarTitle: agentchat
title: autogen.agentchat
---

**** SYMBOL_START ****

    <h2 id="autogen.agentchat.MyClass" class="doc doc-heading">
        <code class="doc-symbol doc-symbol-heading doc-symbol-class"></code>
        <span class="doc doc-object-name doc-class-name">MyClass</span>
        <a href="#autogen.MyClass" class="headerlink" title="Permanent link"></a>
    </h2>

`autogen.MyClass`

```python
MyClass(
    name: str,
)
```

    A class for generic conversable agents which can be configured as assistant or user proxy.

<b>Parameters:</b>
| Name | Description |
|--|--|
| `name` | name of the agent.<br/><br/>**Type:** `str` |

**** SYMBOL_END ****

"""

    @pytest.fixture
    def api_dir(self) -> Generator[Path, None, None]:
        """Fixture providing the api reference directory."""
        # create a tmp directory and add some files, finally yield the path
        with tempfile.TemporaryDirectory() as tmp_dir:
            api_dir = Path(tmp_dir)
            autogen_dir = api_dir / "autogen"
            autogen_dir.mkdir(parents=True, exist_ok=True)
            (autogen_dir / "index.md").write_text(TestSplitReferenceFilesBySymbol.AUTOGEN_INDEX_CONTENT)
            agentchat_path = autogen_dir / "agentchat"
            agentchat_path.mkdir(parents=True, exist_ok=True)
            (agentchat_path / "index.md").write_text(TestSplitReferenceFilesBySymbol.AGENTCHAT_INDEX_CONTENT)
            yield api_dir

    @pytest.fixture
    def expected_files(self) -> list[str]:
        """Fixture providing the expected directories."""
        return [
            # "overview.md",
            "gather_usage_summary.md",
            "config_list_from_dotenv.md",
            "ConversableAgent.md",
            "agentchat/MyClass.md",
            # "agentchat/overview.md",
        ]

    def test_split_reference_by_symbols(self, api_dir: Path, expected_files: list[str]) -> None:
        """Test that files are split correctly."""
        all_files_relative_to_api_dir = [str(p.relative_to(api_dir)) for p in api_dir.rglob("*.md")]
        print("*" * 80)
        print(f"{all_files_relative_to_api_dir=}")

        symbol_files_generator = SplitReferenceFilesBySymbols(api_dir)
        symbol_files_generator.generate()

        # Verify the files exist
        actual_files = [str(p.relative_to(api_dir)) for p in api_dir.rglob("*.md")]
        assert len(actual_files) == len(expected_files)
