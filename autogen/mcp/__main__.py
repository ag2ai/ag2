# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Annotated, Optional

import typer

from .. import __version__
from .mcp_proxy import MCPProxy

app = typer.Typer(rich_markup_mode="rich")

logger = logging.getLogger(__name__)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", help="Show the version and exit.", callback=version_callback),
    ] = None,
) -> None:
    """AG2 mcp proxy CLI - The [bold]mcp proxy[/bold] command line app. 😎

    Generate mcp proxy for your [bold]AG2[/bold] projects.

    Read more in the docs: ...
    """  # noqa: D415


@app.command()
def create(
    openapi_json: Annotated[
        Optional[str],
        "JSON specification of the OpenAPI to use for the proxy generation.",
    ] = None,
    openapi_url: Annotated[
        Optional[str],
        "URL to the OpenAPI specification to use for the proxy generation.",
    ] = None,
    client_source_path: Annotated[
        Optional[str],
        "Path to the generated proxy client source code.",
    ] = None,
    server_url: Annotated[
        Optional[str],
        "Comma-separated list of server URLs to use for the proxy generation.",
    ] = None,
) -> None:
    """Generate mcp proxy for your AG2 projects."""
    MCPProxy.create(
        openapi_json=openapi_json,
        openapi_url=openapi_url,
        client_source_path=client_source_path,
        servers=[{"url": server_url}],
    )


if __name__ == "__main__":
    app(prog_name="mcp_proxy")
