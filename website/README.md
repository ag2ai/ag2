# Website

This website is built using [Mintlify](https://mintlify.com/docs/quickstart), a modern website generator.

## Prerequisites

To build and test documentation locally, begin by downloading and installing [Node.js](https://nodejs.org/en/download/) and [Mintlify CLI](https://www.npmjs.com/package/mintlify)

## Installation

```console
pip install pydoc-markdown pyyaml colored
```

### Install Quarto

`quarto` is used to render notebooks.

Install it [here](https://github.com/quarto-dev/quarto-cli/releases).

> Note: Ensure that your `quarto` version is `1.5.23` or higher.

## Local Development

Navigate to the `website` folder and run:

```console
pydoc-markdown
python ./process_notebooks.py render
npm install
```

Run the following command at the root of your documentation (where mint.json is)

```console
npm run mintlify:dev
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.
