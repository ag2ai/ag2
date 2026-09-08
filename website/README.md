# Website

This website is built using [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), a modern website generator.

## Building Documentation Locally

You can build and serve the documentation locally by following these steps:


### Installation

From the project root directory, install the necessary Python packages:

```console
uv pip install --group docs -e .
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
