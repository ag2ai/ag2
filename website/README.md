# Website

This website is built using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), a modern website generator.

## Building Documentation Locally

You can build and serve the documentation locally by following these steps:


### Prerequisites

1.  Install Quarto:
    - Visit the Quarto download <a href="https://quarto.org/docs/download/" target="_blank">page</a>.
    - Navigate to the Pre-release tab and download the latest version
    - Ensure you install version `1.5.23` or higher.

### Installation

From the project root directory, install the necessary Python packages:

```console
pip install -e ".[docs]"
```

### Building the Documentation

To build the documentation locally, run the following command from the project root directory:

```console
./scripts/docs_build_mkdocs.sh
```

Optionally, you can pass the `--force` flag to clean up all temporary files and generate the documentation from scratch:

```console
./scripts/docs_build_mkdocs.sh --force
```

### Serving the documentation

Once the build is complete, please run the following command to serve the docs:

```console
./scripts/docs_serve_mkdocs.sh
```

This will spin up a server at port 8000, which you can access by visiting `http://localhost:8000` in your browser.

## Build with Dev Containers

If you prefer to use a containerized development environment, you can build and test the documentation using Dev Containers.

### Setting up Dev Containers

- Install <a href="https://code.visualstudio.com" target="_blank">VSCode</a> if you haven't already.
- Open the project in VSCode.
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select `Dev Containers: Reopen in Container`.

This will open the project in a Dev Container with all the required dependencies pre-installed.

### Building and serving in the container

Once your project is open in the Dev Container:

- Open a terminal in VSCode and install the project with docs dependencies:

    ```console
    pip install -e ".[docs]"
    ```

- Build the documentation:

    ```console
    ./scripts/docs_build_mkdocs.sh
    ```

- Serve the documentation:

    ```console
    ./scripts/docs_serve_mkdocs.sh
    ```

The documentation will be accessible at `http://localhost:8000` in your browser.

## Handling updates or changes

For any changes to be reflected in the documentation, you will need to:

- Stop the running server
- Run the build command again
- Start the server again


When switching branches or making major changes to the documentation structure, you might occasionally notice deleted files still appearing or changes not showing up properly. This happens due to cached build files. In such cases, running the commands with the `--force` flag will clear the cache and rebuild everything from scratch:

```console
./scripts/docs_build_mkdocs.sh --force
./scripts/docs_serve_mkdocs.sh
```


## Adding Notebooks to the Website

When you want to add a new Jupyter notebook and have it rendered in the documentation, you need to follow specific guidelines to ensure proper integration with the website.

Please refer to <a href="https://github.com/ag2ai/ag2/blob/main/notebook/contributing.md#how-to-get-a-notebook-displayed-on-the-website" target="_blank">this</a> guideline for more details.
