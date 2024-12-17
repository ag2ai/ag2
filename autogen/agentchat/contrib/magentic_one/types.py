# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Literal, TypedDict, Union


class DOMRectangle(TypedDict):
    x: Union[int, float]
    y: Union[int, float]
    width: Union[int, float]
    height: Union[int, float]
    top: Union[int, float]
    right: Union[int, float]
    bottom: Union[int, float]
    left: Union[int, float]


class VisualViewport(TypedDict):
    height: Union[int, float]
    width: Union[int, float]
    offsetLeft: Union[int, float]
    offsetTop: Union[int, float]
    pageLeft: Union[int, float]
    pageTop: Union[int, float]
    scale: Union[int, float]
    clientWidth: Union[int, float]
    clientHeight: Union[int, float]
    scrollWidth: Union[int, float]
    scrollHeight: Union[int, float]


class InteractiveRegion(TypedDict):
    tag_name: str
    role: str
    aria_name: str
    v_scrollable: bool
    rects: List[DOMRectangle]


# Helper functions for dealing with JSON. Not sure there's a better way?


def _get_str(d: Any, k: str) -> str:
    val = d[k]
    assert isinstance(val, str)
    return val


def _get_number(d: Any, k: str) -> Union[int, float]:
    val = d[k]
    assert isinstance(val, int) or isinstance(val, float)
    return val


def _get_bool(d: Any, k: str) -> bool:
    val = d[k]
    assert isinstance(val, bool)
    return val


def domrectangle_from_dict(rect: Dict[str, Any]) -> DOMRectangle:
    return DOMRectangle(
        x=_get_number(rect, "x"),
        y=_get_number(rect, "y"),
        width=_get_number(rect, "width"),
        height=_get_number(rect, "height"),
        top=_get_number(rect, "top"),
        right=_get_number(rect, "right"),
        bottom=_get_number(rect, "bottom"),
        left=_get_number(rect, "left"),
    )


def interactiveregion_from_dict(region: Dict[str, Any]) -> InteractiveRegion:
    typed_rects: List[DOMRectangle] = []
    for rect in region["rects"]:
        typed_rects.append(domrectangle_from_dict(rect))

    return InteractiveRegion(
        tag_name=_get_str(region, "tag_name"),
        role=_get_str(region, "role"),
        aria_name=_get_str(region, "aria-name"),
        v_scrollable=_get_bool(region, "v-scrollable"),
        rects=typed_rects,
    )


def visualviewport_from_dict(viewport: Dict[str, Any]) -> VisualViewport:
    return VisualViewport(
        height=_get_number(viewport, "height"),
        width=_get_number(viewport, "width"),
        offsetLeft=_get_number(viewport, "offsetLeft"),
        offsetTop=_get_number(viewport, "offsetTop"),
        pageLeft=_get_number(viewport, "pageLeft"),
        pageTop=_get_number(viewport, "pageTop"),
        scale=_get_number(viewport, "scale"),
        clientWidth=_get_number(viewport, "clientWidth"),
        clientHeight=_get_number(viewport, "clientHeight"),
        scrollWidth=_get_number(viewport, "scrollWidth"),
        scrollHeight=_get_number(viewport, "scrollHeight"),
    )


class SentinelMeta(type):
    """
    A baseclass for sentinels that plays well with type hints.
    Define new sentinels like this:

    ```
    class MY_DEFAULT(metaclass=SentinelMeta):
        pass


    foo: list[str] | None | type[MY_DEFAULT] = MY_DEFAULT
    ```

    Reference: https://stackoverflow.com/questions/69239403/type-hinting-parameters-with-a-sentinel-value-as-the-default
    """

    def __repr__(cls) -> str:
        return f"<{cls.__name__}>"

    def __bool__(cls) -> Literal[False]:
        return False
