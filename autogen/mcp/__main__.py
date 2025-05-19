# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Annotated, Literal, Optional

from .. import __version__
from ..import_utils import optional_import_block, require_optional_import
from .mcp_proxy import MCPProxy

logger = logging.getLogger(__name__)

with optional_import_block():
    import typer


@require_optional_import(["typer"], "mcp-proxy-gen")
def create_typer_app() -> "typer.Typer":
    """Create a Typer app for the mcp proxy CLI."""
    app = typer.Typer(rich_markup_mode="rich")

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
        openapi_specification: Annotated[
            Optional[str],
            "Specification of the OpenAPI to use for the proxy generation.",
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
        configuration_type: Annotated[
            Literal["json", "yaml"],
            "Configuration type of the specification. Can be 'json' or 'yaml'.",
        ] = "json",
    ) -> None:
        """Generate mcp proxy for your AG2 projects."""
        MCPProxy.create(
            openapi_specification=openapi_specification,
            openapi_url=openapi_url,
            client_source_path=client_source_path,
            servers=[{"url": server_url}],
            configuration_type=configuration_type,
        )

    return app


if __name__ == "__main__":
    app = create_typer_app()
    app(prog_name="mcp_proxy")
