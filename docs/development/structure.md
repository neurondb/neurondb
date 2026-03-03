# Documentation Structure

This file explains how docs are organized and how to extend them.

## Top-level layout

- `Docs/getting-started/`: fastest path to a working setup; minimal prerequisites
- `Docs/internals/`: deeper dives (internals, optimization, production)
- `Docs/reference/`: stable reference pages (glossary, summaries, doc maps)

## Style guidelines

- **Prefer short sections** with clear headings.
- **Link to source** by referencing file paths like `NeuronDB/src/...` rather than duplicating code.
- **Separate concerns**:
  - getting-started docs: “do this”
  - internals docs: “why it works / how it’s built”
  - reference docs: “what is X / where is Y”

## Adding a new doc

1. Pick the right section (`getting-started`, `internals`, or `reference`)
2. Add the file
3. Link it from:
   - the relevant section `README.md`
   - `documentation.md` if it's a core entry point


