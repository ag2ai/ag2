/*
 * Purpose of This File:
 * This file is required for Quarto to render both HTML and MDX files in the `website` folder.
 * These rendered files are essential for Mintlify to function correctly.
 *
 * Context:
 * - The `quarto render` command behaves differently when the `docusaurus.config.js` file is
 *   present in the `website` folder.
 * - Without this file, Quarto does not render MDX files, leading to failures
 *   when running the `process_notebook.py` script.
 * - As a result, this file is retained temporarily until a better solution is identified.
 *
 * Options Tried (and Issues Faced):
 * 1. Adding a `_quarto.yml` file in the `website` folder with the following configurations:
 *
 *    **Minimal Configuration:**
 *    ```
 *    project:
 *      type: docusaurus
 *
 *    format:
 *      html: default
 *      docusaurus-md: default
 *    ```
 *
 *    **Full Configuration from Quarto Repository:**
 *    The configuration from [Quarto's docusaurus extension](https://github.com/quarto-dev/quarto-cli/blob/main/src/resources/extensions/quarto/docusaurus/_extension.yml).
 *
 *    - Both configurations instruct Quarto to render HTML and MDX files.
 *    - However, in both cases, Mintlify fails to generate docs, raising the following error:
 *      > Warning: data for page "/[[...slug]]" (path "/docs/Home") is 56.1 MB which exceeds
 *      > the threshold of 21 MB. This amount of data can reduce performance.
 *    - More info: https://nextjs.org/docs/messages/large-page-data
 */
