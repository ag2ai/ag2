# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Type-checks for MemorySkill's overloaded ``resource`` / ``script`` decorators.

mypy validates this module (it is in ``[tool.mypy].files``); it is not executed
as a test. The decorators must return the wrapped callable unchanged — so the
decorated function stays callable with its original signature and return type.
A wrong overload (e.g. returning the decorator factory, or ``Any``) would make
these ``assert_type`` calls or the ``[arg-type]`` ignores fail type-checking.
"""

from typing_extensions import assert_type

from autogen.beta.tools import MemorySkill

skill = MemorySkill(name="s", description="d")


# Bare form: @skill.resource(func) -> the same callable.
@skill.resource
def res_bare(x: int) -> str:
    return str(x)


assert_type(res_bare(1), str)
res_bare("not-an-int")  # type: ignore[arg-type]  # signature preserved


# Parameterized form: @skill.resource(...) -> decorator -> the same callable.
@skill.resource(name="r", description="d")
def res_param(x: int) -> str:
    return str(x)


assert_type(res_param(1), str)
res_param("not-an-int")  # type: ignore[arg-type]


# Bare form for scripts.
@skill.script
def script_bare(value: float) -> str:
    return str(value)


assert_type(script_bare(1.0), str)
script_bare("not-a-float")  # type: ignore[arg-type]


# Parameterized form for scripts.
@skill.script(name="s2", description="d")
def script_param(value: float) -> str:
    return str(value)


assert_type(script_param(1.0), str)
script_param("not-a-float")  # type: ignore[arg-type]
