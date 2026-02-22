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

### Building and Serving the Documentation

**Note:** The scripts (`./scripts/docs_build_mkdocs.sh`, `./scripts/docs_serve_mkdocs.sh`, and any related automation) previously used for building and serving documentation have been removed from the codebase. Please refer to the main project README or contact project maintainers for updated instructions on how to build and serve documentation locally.

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

- Build and serve the documentation using the updated workflow. (The commands referenced previously may no longer be available; check with the most recent project documentation or contact maintainers.)

The documentation will be accessible at `http://localhost:8000` in your browser if you follow the appropriate serving instructions.

## Handling updates or changes

For any changes to be reflected in the documentation, you will need to:

- Stop any running doc server
- Follow the updated build/serve process as provided in the main project documentation


When switching branches or making major changes to the documentation structure, you might occasionally notice deleted files still appearing or changes not showing up properly. This happens due to cached build files. In previous workflows, running with the `--force` flag would clear the cache and rebuild from scratch; as the related scripts and automation have been removed, please check the latest project documentation or ask a maintainer for the current process to rebuild documentation and handle cache issues.


## Adding Notebooks to the Website

When you want to add a new Jupyter notebook and have it rendered in the documentation, you need to follow specific guidelines to ensure proper integration with the website.

Please refer to <a href="https://github.com/ag2ai/ag2/blob/main/notebook/contributing.md#how-to-get-a-notebook-displayed-on-the-website" target="_blank">this</a> guideline for more details.
 
  