# OFX File Creator

![CI](https://github.com/mjaarnold95/ofx_file_creator/actions/workflows/ci.yml/badge.svg)

Turn bank exports (CSV/Excel) into OFX files for import into personal finance tools.
The `utils/` package normalizes columns, cleans text, infers transaction types, and
renders OFX-compliant output. Optional LLM enrichment can provide nicer payee names and
categories.

## Quick start

Install dependencies (macOS):

```bash
python3 -m pip install -r requirements.txt
# Optional for tests or local LLM demos
python3 -m pip install pytest pyyaml mlx mlx-lm
```

Create an OFX from a CSV:

```python
from pathlib import Path
from utils.etl import load_and_prepare
from utils.build_ofx import build_ofx

df = load_and_prepare(Path("transactions.csv"))
ofx = build_ofx(df, accttype="checking", acctid="12345")
Path("transactions.ofx").write_text(ofx, encoding="utf-8")
```

Or run the example CLI in `main.py` after editing the file paths.

### Try it with the included example

- Input: `examples/transactions.sample.csv`
- Run:
   ```bash
   python3 examples/generate_ofx.py
   ```
- Output: `examples/transactions.sample.ofx`

## How it works (pipeline)

1) Load via `utils.io.load_transactions` (CSV/Excel; sheet chosen by
   `utils.sheet.find_best_sheet`). Columns are read as `object` dtype.
2) Normalize with `utils.sheet.normalize_columns` and map vendor columns to canonical
   names (`acctnum, date, time, amount, description, trntype, fitid, ...`) via
   `utils.sheet.detect_columns`.
3) ETL (`utils.etl.load_and_prepare`) builds:
   - `raw_desc`, `payee_display` (coalesced text); `cleaned_desc` (uppercased, squashed).
   - `date_parsed` (UTC) + optional `time` parsing (supports Excel fractions and HH:MM[:SS]).
   - `amount_clean` (handles `$`, commas, parentheses negatives, empty/NaN).
   - Optional LLM columns: `payee_llm`, `category_llm`, `description_llm`.
   - `trntype_norm` via rule-driven inference; `fitid_norm` cleanup if provided.
4) Validate with `utils.validate.assert_ofx_ready` (`amount_clean` required, timestamp fallback allowed).
5) Render with `utils.build_ofx.build_ofx`:
   - `<DTSTART>/<DTEND>` from `date_parsed` range or fallback.
   - `<DTPOSTED>` always set (never the string "None").
   - Name/memo precedence: `payee_llm` → `payee_display` → `posting_memo` → `cleaned_desc` → `raw_desc`.
   - `TRNTYPE` from `trntype_norm` or inferred from amount; `FITID` generated if missing.

## Optional LLM enrichment

Enable by passing `llm_client` to `load_and_prepare`:

```python
client = ...  # your LLM SDK, callable, or mlx-lm config dict
df = load_and_prepare(Path("transactions.csv"), llm_client=client, llm_batch_size=10)
```

Client shapes supported: an object with `generate_batch(prompts)`, a callable, or an
`mlx-lm` configuration dict. See `tests/test_utils.py` for concrete examples.

### Using mlx-lm locally (Apple silicon)

```python
from mlx_lm import load as mlx_load
from utils.etl import load_and_prepare

model, tokenizer = mlx_load("mlx-community/Llama-3.2-3B-Instruct-4bit")
client = {
    "llm_model": "mlx_lm",
    "model": model,
    "tokenizer": tokenizer,
    "generation_kwargs": {"max_tokens": 128},
}
df = load_and_prepare(Path("transactions.csv"), llm_client=client, llm_batch_size=10)
```

Set `use_batch=False` to call `mlx_lm.generate` per prompt. LLM usage is optional; the
pipeline works fully offline when `llm_client=None`.

## Customizing transaction type rules (template: `examples/rules.example.yaml`)

Rules live in `utils/rules.py` and support JSON/YAML overrides. See
`examples/rules.example.yaml` for a ready-to-tweak template showing `extend`/`replace`:

```python
from utils.rules import load_rules
from utils.cleaning import infer_trntype_series

rules = load_rules("examples/rules.example.yaml")
df = load_and_prepare(Path("transactions.csv"))
df["trntype_norm"] = infer_trntype_series(
    df["amount_clean"], df.get("trntype"), df.get("cleaned_desc"), rules=rules
)
```

## Testing

```bash
python3 -m pytest -q
```

Tests cover time parsing (incl. Excel fractions), amount cleaning, rule overrides,
LLM batching/parsing, FITID generation, and OFX field precedence.

## Gotchas

- Parentheses amounts are negatives; empty/"nan" handled.
- If both debit/credit columns exist, net amount = credit - debit.
- `derive_acctid_from_path` extracts digits from filenames; otherwise uses a stub + hash.
- `NAME` is escaped and trimmed to 32 chars for OFX.
