# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from a2ui.core import STRICT_VALIDATION, MessageProcessor, MessageProcessorOptions
from a2ui.core.basic_catalog import BasicCatalog

import ag2.a2ui.beta
from ag2.a2ui.beta._types import A2UIVersion
from ag2.a2ui.beta.schema_manager import A2UISchemaManager

BETA_DIR = Path(ag2.a2ui.beta.__file__).parent

_OLD_PACKAGE_IMPORT = re.compile(r"^\s*(?:from|import)\s+ag2\.a2ui(?!\.beta)\b", re.MULTILINE)
_PARENT_RELATIVE_IMPORT = re.compile(r"^\s*from\s+\.\.\s*(?:import|\w)", re.MULTILINE)


def test_beta_imports_nothing_from_the_old_a2ui_package() -> None:
    offenders = [
        path.relative_to(BETA_DIR).as_posix()
        for path in BETA_DIR.rglob("*.py")
        if _OLD_PACKAGE_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_beta_relative_imports_stay_inside_beta() -> None:
    # `from ..x import y` in a top-level beta module would reach into `ag2.a2ui`;
    # inside `beta/a2a` and `beta/transports` it resolves back to `beta`.
    offenders = [
        path.relative_to(BETA_DIR).as_posix()
        for path in BETA_DIR.glob("*.py")
        if _PARENT_RELATIVE_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_beta_vendors_no_schemas() -> None:
    assert not [p for p in BETA_DIR.rglob("*.json") if "spec" in p.parts]
    assert not any((BETA_DIR / name).exists() for name in ("v0_9", "v0_9_1", "v1_0"))


def test_missing_extra_gives_the_install_hint() -> None:
    code = (
        "import sys\n"
        "sys.modules['a2ui'] = None\n"
        "import ag2.a2ui.beta as beta\n"
        "try:\n"
        "    beta.A2UIServer()\n"
        "except ImportError as e:\n"
        "    print(e)\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert 'pip install "ag2[a2ui-beta]"' in result.stdout


@pytest.mark.parametrize("version", ["v0.9", "v0.9.1", "v1.0"])
def test_prompt_example_is_valid_under_core(version: A2UIVersion) -> None:
    manager = A2UISchemaManager(protocol_version=version)
    manager.generate_prompt_section()  # loads the example
    example = json.loads(manager._build_prompt_example())  # noqa: SLF001 - the shipped prompt data is under test
    processor = MessageProcessor(
        [BasicCatalog(version)], options=MessageProcessorOptions(validation_config=STRICT_VALIDATION)
    )
    processor.process_messages(example)  # type: ignore[arg-type]
