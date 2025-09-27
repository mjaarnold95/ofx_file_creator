# Copilot instructions: OFX File Creator

Purpose: turn bank exports (CSV/Excel) into OFX. Pipeline = load → normalize → enrich (optional LLM) → infer types → validate → render OFX.

## Big picture and data flow
- Entry points
  - CLI example in `main.py` shows end-to-end usage around two files (checking/savings) and optional mlx-lm enrichment.
  - Library flow: `utils.etl.load_and_prepare(path, llm_client=None)` → cleaned DataFrame → `utils.build_ofx.build_ofx(df, accttype, acctid, ...)` → OFX text.
- Stages and contracts
  - I/O: `utils.io.load_transactions(Path)` handles CSV and Excel (picks sheet via `utils.sheet.find_best_sheet`). All columns loaded as object dtype.
  - Normalization: `utils.sheet.normalize_columns` lowercases and snake_cases; `utils.sheet.detect_columns` maps many vendor column names to canonical ones: `acctnum, acctname, date, time, amount, description, trntype, fitid, checknum, memo, name`.
  - ETL: `utils.etl.load_and_prepare` builds key fields:
    - `raw_desc` and `payee_display` via coalescing multiple description-like columns (avoids bleeding `acctname`).
    - `date_parsed` (UTC) from `date` or any “date”-like column; `time` parsed into timedelta (supports Excel fractional days and HH:MM[:SS]).
    - `amount_clean` vectorized via `utils.cleaning.clean_amount_series` (handles commas, $ signs, parenthesis negatives).
    - `cleaned_desc` uppercased, whitespace collapsed.
    - Optional LLM enrichment adds `payee_llm, category_llm, description_llm`.
    - `trntype_norm` inferred by `utils.cleaning.infer_trntype_series` using `utils.rules.DEFAULT_RULES`.
    - `fitid_norm` (cleaned original id) if present.
  - Validation: `utils.validate.assert_ofx_ready(df)` requires column `amount_clean` and at least one usable timestamp (prefers `date_parsed`, falls back to statement dates).
  - Rendering: `utils.build_ofx.build_ofx`
    - Computes `<DTSTART>/<DTEND>` from `date_parsed` range; when dates missing, uses fallback from validation.
    - `<DTPOSTED>` per transaction = `ofx_datetime(date_parsed)` else fallback (statement end or now), never the string "None".
    - Prioritizes display fields: `NAME` from `payee_llm` → `payee_display` → `posting_memo` → `cleaned_desc` → `raw_desc`; `MEMO` from `description_llm` → memo/desc fallbacks.
    - Ensures `TRNTYPE` via `trntype_norm` or inference from amount sign; generates stable `FITID` via `utils.id.make_fitid` when absent.

## Conventions and patterns to follow
- Always normalize columns early and rely on the canonical names above; add to `detect_columns` instead of scattering custom renames.
- Keep pandas operations vectorized (see time parsing and coalescing patterns in `etl.py`).
- OFX formatting uses `utils.date_time.ofx_datetime(ts)` which expects tz-aware UTC timestamps and returns `YYYYMMDDHHMMSS.000[0:UTC]`.
- When adding new transaction-type heuristics, extend `utils.rules` (regex and keyword rules) rather than hardcoding elsewhere. Support JSON/YAML overrides via `utils.rules.load_rules`.

## LLM enrichment integration
- Optional step. `utils.llm_enrichment.enrich_transactions_with_llm(df, client, batch_size)` expects columns: `raw_desc`, `payee_display`, `cleaned_desc`.
- Client shapes supported:
  - Preferred: object with `generate_batch(prompts) -> list[str]`.
  - Fallbacks: object with `generate(prompt)`, a callable, or an `mlx_lm` config dict.
- mlx-lm: pass `{"llm_model": "mlx_lm", "model": model, "tokenizer": tokenizer, "generation_kwargs": {...}, "use_batch": True/False}`. See tests for parsing and batching behavior.

## Key files to study and extend
- `utils/etl.py`: the orchestrated loader; extend here to support new vendor columns or enrichment fields.
- `utils/build_ofx.py`: OFX assembly and field precedence; careful with escaping and max lengths (`NAME` sliced to 32 chars).
- `utils/cleaning.py`: amount parsing, description cleaning, TRNTYPE inference.
- `utils/rules.py`: default rules and override/merge logic; supports YAML/JSON with `extend`/`replace` keys.
- `utils/date_time.py`: UTC coercion and OFX timestamp formatting; robust time parsing (`parse_time_to_timedelta`).
- `utils/io.py` and `utils/sheet.py`: input dispatch and sheet/column detection.
- `utils/validate.py`: single source of truth for pre-render checks and timestamp fallback selection.

## Developer workflows
- Install deps (macOS, zsh):
  ```bash
  python3 -m pip install -r requirements.txt
  # Optional for tests/LLM demos
  python3 -m pip install pytest pyyaml mlx mlx-lm
  ```
- Run tests (pytest-based; see `tests/test_utils.py` for examples covering dates, rules, LLM, OFX):
  ```bash
  python3 -m pytest -q
  ```
- Run end-to-end (edit paths and LLM block in `main.py` or call library API):
  ```python
  from pathlib import Path
  from utils.etl import load_and_prepare
  from utils.build_ofx import build_ofx
   - Custom TRNTYPE rules quick start: see `examples/rules.example.yaml` as a template for overrides.

  df = load_and_prepare(Path("transactions.csv"))
  ofx = build_ofx(df, accttype="checking", acctid="12345")
  Path("transactions.ofx").write_text(ofx, encoding="utf-8")
  ```
 - Quick smoke test (uses included example data):
   ```bash
   python3 examples/generate_ofx.py
   ```
   Output: `examples/transactions.sample.ofx`

## Gotchas and edge cases captured in code/tests
- Excel time fractions (0..1) are supported; strings like "1:30 PM" parsed too.
- Amount strings with parentheses mean negative values; empty/"nan" handled.
- If both debit/credit columns exist, net amount = credit - debit.
- `derive_acctid_from_path` pulls digits from filename; else prefixes hash with stub (e.g., `CHK`/`SAV`).
- Tests monkeypatch `mlx_lm`; installing mlx is optional for non-LLM flows.

When adding features, preserve the DataFrame contract used by `build_ofx` and expand detection/cleaning rules rather than introducing one-off branches.
