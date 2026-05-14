# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from autogen.coding.base import CodeBlock
from autogen.coding.func_with_reqs import FunctionWithRequirements, with_requirements
from autogen.coding.local_commandline_code_executor import LocalCommandLineCodeExecutor
from autogen.import_utils import optional_import_block, run_for_optional_imports

with optional_import_block() as result:
    import pandas


classes_to_test = [LocalCommandLineCodeExecutor]


def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@with_requirements(python_packages=["pandas"], global_imports=["pandas"])
def load_data() -> "pandas.DataFrame":
    """Load some sample data.

    Returns:
        pandas.DataFrame: A DataFrame with the following columns: name(str), location(str), age(int)
    """
    data = {
        "name": ["John", "Anna", "Peter", "Linda"],
        "location": ["New York", "Paris", "Berlin", "London"],
        "age": [24, 13, 53, 33],
    }
    return pandas.DataFrame(data)


@with_requirements(global_imports=["NOT_A_REAL_PACKAGE"])
def function_incorrect_import() -> "pandas.DataFrame":
    return pandas.DataFrame()


@with_requirements(python_packages=["NOT_A_REAL_PACKAGE"])
def function_incorrect_dep() -> "pandas.DataFrame":
    return pandas.DataFrame()


def function_missing_reqs() -> "pandas.DataFrame":
    return pandas.DataFrame()


@pytest.mark.parametrize("cls", classes_to_test)
@run_for_optional_imports(["pandas"], "test")
def test_can_load_function_with_reqs(cls) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = cls(work_dir=temp_dir, functions=[load_data])
        code = f"""from {executor.functions_module} import load_data
import pandas

# Get first row's name
print(load_data().iloc[0]['name'])"""

        result = executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ]
        )
        assert result.output == "John\n"
        assert result.exit_code == 0


@pytest.mark.parametrize("cls", classes_to_test)
@run_for_optional_imports(["pandas"], "test")
def test_can_load_function(cls) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = cls(work_dir=temp_dir, functions=[add_two_numbers])
        code = f"""from {executor.functions_module} import add_two_numbers
print(add_two_numbers(1, 2))"""

        result = executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ]
        )
        assert result.output == "3\n"
        assert result.exit_code == 0


# TODO - only run this test for containerized executors, as the environment is not guaranteed to have pandas installed
# It is common for the local environment to have pandas installed, so this test will not work as expected
# @pytest.mark.parametrize("cls", classes_to_test)
# @skip_on_missing_imports(["pandas"], "test")
# def test_fails_for_missing_reqs(cls) -> None:
#     with tempfile.TemporaryDirectory() as temp_dir:
#         executor = cls(work_dir=temp_dir, functions=[function_missing_reqs])
#         code = f"""from {executor.functions_module} import function_missing_reqs
# function_missing_reqs()"""

#         with pytest.raises(ValueError):
#             executor.execute_code_blocks(
#                 code_blocks=[
#                     CodeBlock(language="python", code=code),
#                 ]
#             )


@pytest.mark.parametrize("cls", classes_to_test)
@run_for_optional_imports(["pandas"], "test")
def test_fails_for_function_incorrect_import(cls) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = cls(work_dir=temp_dir, functions=[function_incorrect_import])
        code = f"""from {executor.functions_module} import function_incorrect_import
function_incorrect_import()"""

        with pytest.raises(ValueError):
            executor.execute_code_blocks(
                code_blocks=[
                    CodeBlock(language="python", code=code),
                ]
            )


@pytest.mark.parametrize("cls", classes_to_test)
@run_for_optional_imports(["pandas"], "test")
def test_fails_for_function_incorrect_dep(cls) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = cls(work_dir=temp_dir, functions=[function_incorrect_dep])
        code = f"""from {executor.functions_module} import function_incorrect_dep
function_incorrect_dep()"""

        with pytest.raises(ValueError):
            executor.execute_code_blocks(
                code_blocks=[
                    CodeBlock(language="python", code=code),
                ]
            )


@pytest.mark.parametrize("cls", classes_to_test)
def test_setup_functions_writes_module_as_utf8(cls) -> None:
    """The generated functions module must be written as UTF-8 regardless
    of the host's `locale.getpreferredencoding()` (cp1252 on most Windows
    installs). Without an explicit encoding kwarg, a user-supplied
    function whose source contains any non-cp1252 glyph — non-Latin
    docstring, emoji, smart quotes in an Annotated description — fails
    `Path.write_text` mid-write with UnicodeEncodeError. Same class of
    bug as #1731.
    """
    # FunctionWithRequirements.from_str avoids the inspect.getsource path
    # so the function source survives even when this file is exec'd from
    # an unusual location.
    func = FunctionWithRequirements.from_str(
        '''
def greet(name: str) -> str:
    """Café — say hello (☕) — 例 — émoji 🚀"""
    return f"Hello, {name}!"
'''
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        executor = cls(work_dir=temp_dir, functions=[func])
        # _setup_functions runs lazily on the first execute_code_blocks
        # call. Drive it directly so the assertions below can read the
        # generated module off disk. _setup_functions also spawns the
        # configured Python executable to syntax-check the generated
        # module; the file is written first, so we tolerate that
        # downstream subprocess failure (e.g. when `python` resolves to
        # python3 only on the host) and assert the on-disk bytes.
        try:
            executor._setup_functions()
        except (FileNotFoundError, ValueError, OSError):
            pass

        func_file = executor._work_dir / f"{executor.functions_module}.py"
        assert func_file.exists(), "functions module should be on disk after _setup_functions"

        # Bytes round-trip as UTF-8.
        body = func_file.read_bytes().decode("utf-8")
        assert "Café" in body
        assert "🚀" in body


@pytest.mark.parametrize("cls", classes_to_test)
def test_setup_functions_passes_utf8_encoding_to_write_text(cls) -> None:
    """Companion to test_setup_functions_writes_module_as_utf8: on POSIX
    the bytes-equal check passes regardless of whether `encoding="utf-8"`
    was specified, because the platform default is already UTF-8. This
    test patches `Path.write_text` directly so the contract holds on
    every platform: a future refactor that drops the kwarg fails here
    immediately, even on POSIX.
    """
    func = FunctionWithRequirements.from_str(
        '''
def greet(name: str) -> str:
    """A simple greeter."""
    return f"Hello, {name}!"
'''
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        executor = cls(work_dir=temp_dir, functions=[func])

        original_write_text = Path.write_text
        captured_kwargs: list[dict[str, object]] = []

        def recording_write_text(self: Path, *args: object, **kwargs: object) -> int:
            captured_kwargs.append(dict(kwargs))
            return original_write_text(self, *args, **kwargs)  # type: ignore[arg-type]

        with patch.object(Path, "write_text", recording_write_text):
            try:
                executor._setup_functions()
            except (FileNotFoundError, ValueError, OSError):
                # downstream subprocess check (python on PATH) can fail —
                # the write call we care about has already happened.
                pass

        # First write_text call inside _setup_functions writes the
        # generated module. Subsequent calls (e.g. via the syntax-check
        # subprocess preamble) are not relevant for this assertion.
        assert captured_kwargs, "expected Path.write_text to be invoked during _setup_functions"
        assert captured_kwargs[0].get("encoding") == "utf-8", (
            f"functions module must be written with encoding='utf-8', got kwargs={captured_kwargs[0]!r}"
        )


@pytest.mark.parametrize("cls", classes_to_test)
def test_formatted_prompt(cls) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        executor = cls(work_dir=temp_dir, functions=[add_two_numbers])

        result = executor.format_functions_for_prompt()
        assert (
            '''def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
'''
            in result
        )


@pytest.mark.parametrize("cls", classes_to_test)
def test_formatted_prompt_str_func(cls) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        func = FunctionWithRequirements.from_str(
            '''
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
        )
        executor = cls(work_dir=temp_dir, functions=[func])

        result = executor.format_functions_for_prompt()
        assert (
            '''def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
'''
            in result
        )


@pytest.mark.parametrize("cls", classes_to_test)
def test_can_load_str_function_with_reqs(cls) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        func = FunctionWithRequirements.from_str(
            '''
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
        )

        executor = cls(work_dir=temp_dir, functions=[func])
        code = f"""from {executor.functions_module} import add_two_numbers
print(add_two_numbers(1, 2))"""

        result = executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ]
        )
        assert result.output == "3\n"
        assert result.exit_code == 0


@pytest.mark.parametrize("cls", classes_to_test)
def test_cant_load_broken_str_function_with_reqs(cls) -> None:
    with pytest.raises(ValueError):
        _ = FunctionWithRequirements.from_str(
            '''
invaliddef add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
        )


@pytest.mark.parametrize("cls", classes_to_test)
def test_cant_run_broken_str_function_with_reqs(cls) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        func = FunctionWithRequirements.from_str(
            '''
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
        )

        executor = cls(work_dir=temp_dir, functions=[func])
        code = f"""from {executor.functions_module} import add_two_numbers
print(add_two_numbers(object(), False))"""

        result = executor.execute_code_blocks(
            code_blocks=[
                CodeBlock(language="python", code=code),
            ]
        )
        assert "TypeError: unsupported operand type(s) for +:" in result.output
        assert result.exit_code == 1
