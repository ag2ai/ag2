# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Union

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.code_utils import (
    decide_use_docker,
    is_docker_running,
)
from autogen.coding.base import CodeBlock, CodeExecutor
from autogen.coding.factory import CodeExecutorFactory
from autogen.coding.jupyter import (
    DockerJupyterServer,
    EmbeddedIPythonCodeExecutor,
    JupyterCodeExecutor,
    LocalJupyterServer,
)
from autogen.coding.jupyter.import_utils import skip_on_missing_jupyter_kernel_gateway
from autogen.import_utils import optional_import_block, run_for_optional_imports

from ..conftest import MOCK_OPEN_AI_API_KEY

# needed for skip_on_missing_imports to work
with optional_import_block():
    import ipykernel  # noqa: F401
    import jupyter_client  # noqa: F401
    import requests  # noqa: F401
    import websocket  # noqa: F401


class DockerJupyterExecutor(JupyterCodeExecutor):
    def __init__(self, **kwargs):
        jupyter_server = DockerJupyterServer()
        super().__init__(jupyter_server=jupyter_server, **kwargs)


class LocalJupyterCodeExecutor(JupyterCodeExecutor):
    def __init__(self, **kwargs):
        jupyter_server = LocalJupyterServer()
        super().__init__(jupyter_server=jupyter_server, **kwargs)


# Skip on windows due to kernelgateway bug https://github.com/jupyter-server/kernel_gateway/issues/398
if sys.platform == "win32":
    classes_to_test = [EmbeddedIPythonCodeExecutor]
else:
    classes_to_test = [EmbeddedIPythonCodeExecutor, LocalJupyterCodeExecutor]

skip_docker_test = not (is_docker_running() and decide_use_docker(use_docker=None))

if not skip_docker_test:
    classes_to_test.append(DockerJupyterExecutor)


@run_for_optional_imports(
    [
        "websocket",
        "requests",
        "jupyter_client",
        "ipykernel",
    ],
    "jupyter-executor",
)
@skip_on_missing_jupyter_kernel_gateway()
@pytest.mark.jupyter_executor
class TestCodeExecutor:
    def test_import_utils(self) -> None:
        pass

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_is_code_executor(self, cls) -> None:
        assert isinstance(cls, CodeExecutor)

    def test_create_dict(self) -> None:
        config: dict[str, Union[str, CodeExecutor]] = {"executor": "ipython-embedded"}
        executor = CodeExecutorFactory.create(config)
        assert isinstance(executor, EmbeddedIPythonCodeExecutor)

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_create(self, cls) -> None:
        config = {"executor": cls()}
        executor = CodeExecutorFactory.create(config)
        assert executor is config["executor"]

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_init(self, cls) -> None:
        executor = cls(timeout=10, kernel_name="python3", output_dir=".")
        assert executor._timeout == 10 and executor._kernel_name == "python3" and executor._output_dir == Path()

        # Try invalid output directory.
        with pytest.raises(ValueError, match="Output directory .* does not exist."):
            executor = cls(timeout=111, kernel_name="python3", output_dir="/invalid/directory")

        # Try invalid kernel name.
        with pytest.raises(ValueError, match="Kernel .* is not installed."):
            executor = cls(timeout=111, kernel_name="invalid_kernel_name", output_dir=".")

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_execute_code_single_code_block(self, cls) -> None:
        executor = cls()
        code_blocks = [CodeBlock(code="import sys\nprint('hello world!')", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and "hello world!" in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_execute_code_multiple_code_blocks(self, cls) -> None:
        executor = cls()
        code_blocks = [
            CodeBlock(code="import sys\na = 123 + 123\n", language="python"),
            CodeBlock(code="print(a)", language="python"),
        ]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and "246" in code_result.output

        msg = """
def test_function(a, b):
    return a + b
"""
        code_blocks = [
            CodeBlock(code=msg, language="python"),
            CodeBlock(code="test_function(431, 423)", language="python"),
        ]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and "854" in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_execute_code_bash_script(self, cls) -> None:
        executor = cls()
        # Test bash script.
        code_blocks = [CodeBlock(code='!echo "hello world!"', language="bash")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and "hello world!" in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_timeout(self, cls) -> None:
        executor = cls(timeout=1)
        code_blocks = [CodeBlock(code="import time; time.sleep(10); print('hello world!')", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code and "Timeout" in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_silent_pip_install(self, cls) -> None:
        executor = cls(timeout=600)
        code_blocks = [CodeBlock(code="!pip install matplotlib numpy", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and code_result.output.strip() == ""

        none_existing_package = uuid.uuid4().hex
        code_blocks = [CodeBlock(code=f"!pip install matplotlib_{none_existing_package}", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and "ERROR: " in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_restart(self, cls) -> None:
        executor = cls()
        code_blocks = [CodeBlock(code="x = 123", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and code_result.output.strip() == ""

        executor.restart()
        code_blocks = [CodeBlock(code="print(x)", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code and "NameError" in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_save_image(self, cls) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = cls(output_dir=temp_dir)
            # Install matplotlib.
            code_blocks = [CodeBlock(code="!pip install matplotlib", language="python")]
            code_result = executor.execute_code_blocks(code_blocks)
            assert code_result.exit_code == 0 and code_result.output.strip() == ""

            # Test saving image.
            code_blocks = [
                CodeBlock(code="import matplotlib.pyplot as plt\nplt.plot([1, 2, 3, 4])\nplt.show()", language="python")
            ]
            code_result = executor.execute_code_blocks(code_blocks)
            assert code_result.exit_code == 0
            assert os.path.exists(code_result.output_files[0])
            assert f"Image data saved to {code_result.output_files[0]}" in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_timeout_preserves_kernel_state(self, cls: type[CodeExecutor]) -> None:
        executor = cls(timeout=1)
        code_blocks = [CodeBlock(code="x = 123", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and code_result.output.strip() == ""

        code_blocks = [CodeBlock(code="import time; time.sleep(10)", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code != 0 and "Timeout" in code_result.output

        code_blocks = [CodeBlock(code="print(x)", language="python")]
        code_result = executor.execute_code_blocks(code_blocks)
        assert code_result.exit_code == 0 and "123" in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_save_html(self, cls) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executor = cls(output_dir=temp_dir)
            # Test saving html.
            code_blocks = [
                CodeBlock(code="from IPython.display import HTML\nHTML('<h1>Hello, world!</h1>')", language="python")
            ]
            code_result = executor.execute_code_blocks(code_blocks)
            assert code_result.exit_code == 0
            assert os.path.exists(code_result.output_files[0])
            assert f"HTML data saved to {code_result.output_files[0]}" in code_result.output

    @pytest.mark.parametrize("cls", classes_to_test)
    def test_conversable_agent_code_execution(self, cls) -> None:
        agent = ConversableAgent(
            "user_proxy",
            llm_config=False,
            code_execution_config={"executor": cls()},
        )
        msg = """
Run this code:
```python
def test_function(a, b):
    return a * b
```
And then this:
```python
print(test_function(123, 4))
```
"""
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("OPENAI_API_KEY", MOCK_OPEN_AI_API_KEY)
            reply = agent.generate_reply(
                [{"role": "user", "content": msg}],
                sender=ConversableAgent("user", llm_config=False, code_execution_config=False),
            )
            assert "492" in reply  # type: ignore[operator]
