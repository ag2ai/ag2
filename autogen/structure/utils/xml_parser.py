# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/lion-agi/lion-core are under the Apache-2.0 License
# SPDX-License-Identifier: Apache-2.0

import re
import xml.etree.ElementTree as ET
from typing import Any


def xml_to_dict(
    xml_string: str,
    /,
    suppress=False,
    remove_root: bool = True,
    root_tag: str = None,
) -> dict[str, Any]:
    """
    Parse an XML string into a nested dictionary structure.

    This function converts an XML string into a dictionary where:
    - Element tags become dictionary keys
    - Text content is assigned directly to the tag key if there are no children
    - Attributes are stored in a '@attributes' key
    - Multiple child elements with the same tag are stored as lists

    Args:
        xml_string: The XML string to parse.

    Returns:
        A dictionary representation of the XML structure.

    Raises:
        ValueError: If the XML is malformed or parsing fails.
    """
    try:
        a = XMLParser(xml_string).parse()
        if remove_root and (root_tag or "root") in a:
            a = a[root_tag or "root"]
        return a
    except ValueError as e:
        if not suppress:
            raise e


def dict_to_xml(data: dict, /, root_tag: str = "root") -> str:

    root = ET.Element(root_tag)

    def convert(dict_obj: dict, parent: Any) -> None:
        for key, val in dict_obj.items():
            if isinstance(val, dict):
                element = ET.SubElement(parent, key)
                convert(dict_obj=val, parent=element)
            else:
                element = ET.SubElement(parent, key)
                element.text = str(object=val)

    convert(dict_obj=data, parent=root)
    return ET.tostring(root, encoding="unicode")


class XMLParser:
    def __init__(self, xml_string: str):
        self.xml_string = xml_string.strip()
        self.index = 0

    def parse(self) -> dict[str, Any]:
        """Parse the XML string and return the root element as a dictionary."""
        return self._parse_element()

    def _parse_element(self) -> dict[str, Any]:
        """Parse a single XML element and its children."""
        self._skip_whitespace()
        if self.xml_string[self.index] != "<":
            raise ValueError(f"Expected '<', found '{self.xml_string[self.index]}'")

        tag, attributes = self._parse_opening_tag()
        children: dict[str, str | list | dict] = {}
        text = ""

        while self.index < len(self.xml_string):
            self._skip_whitespace()
            if self.xml_string.startswith("</", self.index):
                closing_tag = self._parse_closing_tag()
                if closing_tag != tag:
                    raise ValueError(f"Mismatched tags: '{tag}' and '{closing_tag}'")
                break
            elif self.xml_string.startswith("<", self.index):
                child = self._parse_element()
                child_tag, child_data = next(iter(child.items()))
                if child_tag in children:
                    if not isinstance(children[child_tag], list):
                        children[child_tag] = [children[child_tag]]
                    children[child_tag].append(child_data)
                else:
                    children[child_tag] = child_data
            else:
                text += self._parse_text()

        result: dict[str, Any] = {}
        if attributes:
            result["@attributes"] = attributes
        if children:
            result.update(children)
        elif text.strip():
            result = text.strip()

        return {tag: result}

    def _parse_opening_tag(self) -> tuple[str, dict[str, str]]:
        """Parse an opening XML tag and its attributes."""
        match = re.match(
            r'<(\w+)((?:\s+\w+="[^"]*")*)\s*/?>',
            self.xml_string[self.index :],  # noqa
        )
        if not match:
            raise ValueError("Invalid opening tag")
        self.index += match.end()
        tag = match.group(1)
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', match.group(2)))
        return tag, attributes

    def _parse_closing_tag(self) -> str:
        """Parse a closing XML tag."""
        match = re.match(r"</(\w+)>", self.xml_string[self.index :])  # noqa
        if not match:
            raise ValueError("Invalid closing tag")
        self.index += match.end()
        return match.group(1)

    def _parse_text(self) -> str:
        """Parse text content between XML tags."""
        start = self.index
        while self.index < len(self.xml_string) and self.xml_string[self.index] != "<":
            self.index += 1
        return self.xml_string[start : self.index]  # noqa

    def _skip_whitespace(self) -> None:
        """Skip any whitespace characters at the current parsing position."""
        p_ = len(self.xml_string[self.index :])  # noqa
        m_ = len(self.xml_string[self.index :].lstrip())  # noqa

        self.index += p_ - m_
