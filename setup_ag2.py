# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

# Get the code version
version = {}
with open(os.path.join(here, "autogen/version.py")) as fp:
    exec(fp.read(), version)
__version__ = version["__version__"]

setuptools.setup(
    name="ag2",
    version=__version__,
    description="Alias package for pyautogen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["pyautogen==" + __version__],
    url="https://github.com/ag2ai/ag2",
    author="Chi Wang & Qingyun Wu",
    author_email="support@ag2.ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache Software License 2.0",
    python_requires=">=3.8,<3.14",
)
