```markdown
# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 — 2025-09-26

First public docs + examples release.

- Add `.github/copilot-instructions.md` for AI agent guidance (architecture, data flow, conventions, LLM integration, workflows)
- Add `examples/`:
  - `transactions.sample.csv` (minimal dataset)
  - `generate_ofx.py` (end-to-end demo generating `transactions.sample.ofx`)
  - `rules.example.yaml` (template for custom TRNTYPE rule overrides)
- Improve `README.md`:
  - Quick start and end-to-end snippet
  - "Try it" section pointing to examples
  - Rules customization section referencing the example template
- Add `.vscode/tasks.json` with a "Run tests" task (pytest)
- Minor docstrings/notes in `utils/`

Tag: `v0.1.0` (commit b2921e3)


## v0.1.1 — 2025-09-26

Internal refactor and docs update.

- Moved trntype inference out of `utils.cleaning` into `utils.trntype` to
  separate concerns between data cleaning and rule-driven inference. `utils.cleaning`
  continues to re-export `infer_trntype`/`infer_trntype_series` for backwards
  compatibility; prefer `utils.trntype` for new imports.

Tag: `v0.1.1` (internal)

```
# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 — 2025-09-26

First public docs + examples release.

- Add `.github/copilot-instructions.md` for AI agent guidance (architecture, data flow, conventions, LLM integration, workflows)
- Add `examples/`:
  - `transactions.sample.csv` (minimal dataset)
  - `generate_ofx.py` (end-to-end demo generating `transactions.sample.ofx`)
  - `rules.example.yaml` (template for custom TRNTYPE rule overrides)
- Improve `README.md`:
  - Quick start and end-to-end snippet
  - "Try it" section pointing to examples
  - Rules customization section referencing the example template
- Add `.vscode/tasks.json` with a "Run tests" task (pytest)
- Minor docstrings/notes in `utils/`

Tag: `v0.1.0` (commit b2921e3)
