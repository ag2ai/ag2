# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/https://github.com/Lancetnik/FastDepends are under the MIT License.
# SPDX-License-Identifier: MIT

from autogen.fast_depends import Depends
from autogen.fast_depends.core import build_call_model
from autogen.fast_depends.library import CustomField


def test_params():
    def func1(m): ...

    def func2(c, b=Depends(func1), d=CustomField()):  # noqa: B008
        ...

    def func3(b): ...

    def main(a, b, m=Depends(func2), k=Depends(func3)): ...

    def extra_func(n): ...

    model = build_call_model(main, extra_dependencies=(Depends(extra_func),))

    assert set(model.params.keys()) == {"a", "b"}
    assert set(model.flat_params.keys()) == {"a", "b", "c", "m", "n"}


def test_args_kwargs_params():
    def func1(m): ...

    def func2(c, b=Depends(func1), d=CustomField()):  # noqa: B008
        ...

    def func3(b): ...

    def default_var_names(a, *args, b, m=Depends(func2), k=Depends(func3), **kwargs):
        return a, args, b, kwargs

    def custom_var_names(a, *args_, b, m=Depends(func2), k=Depends(func3), **kwargs_):
        return a, args_, b, kwargs_

    def extra_func(n): ...

    model1 = build_call_model(default_var_names, extra_dependencies=(Depends(extra_func),))

    assert set(model1.params.keys()) == {"a", "args", "b", "kwargs"}
    assert set(model1.flat_params.keys()) == {"a", "args", "b", "kwargs", "c", "m", "n"}

    model2 = build_call_model(custom_var_names, extra_dependencies=(Depends(extra_func),))

    assert set(model2.params.keys()) == {"a", "args_", "b", "kwargs_"}
    assert set(model2.flat_params.keys()) == {"a", "args_", "b", "kwargs_", "c", "m", "n"}

    assert default_var_names(1, *("a"), b=2, **{"kw": "kw"}) == (1, ("a",), 2, {"kw": "kw"})
    assert custom_var_names(1, *("a"), b=2, **{"kw": "kw"}) == (1, ("a",), 2, {"kw": "kw"})
