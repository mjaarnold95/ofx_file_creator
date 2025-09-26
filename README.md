# OFX File Creator

This project helps transform raw bank or credit card exports into OFX files that can be
imported into personal finance tools. The utilities under `utils/` normalize common CSV
fields, clean descriptions, infer transaction types, and assemble OFX-compliant output.

## LLM-powered enrichment (optional)

Some workflows benefit from model-generated payee names or categories. The
`utils.llm_enrichment.enrich_transactions_with_llm` helper can be enabled via the
`load_and_prepare` function:

```python
from utils.etl import load_and_prepare
from utils.llm_enrichment import enrich_transactions_with_llm

client = ...  # your LLM SDK client
prepared = load_and_prepare(path_to_csv, llm_client=client, llm_batch_size=10)
```

When an `llm_client` is supplied, the loader will batch transactions and send prompts
containing the raw description, the selected payee display text, and the cleaned
description. The model is asked to emit minified JSON with `payee`, `category`, and
`description` fields. Parsed results are added to the dataframe as `payee_llm`,
`category_llm`, and `description_llm` columns.

### Configuring clients

* **API keys** – Follow the vendor instructions for your chosen SDK (e.g., OpenAI,
  Anthropic). Export any required tokens (such as `OPENAI_API_KEY`) before running the
  enrichment step.
* **Batch sizing** – Adjust `llm_batch_size` to match the provider's rate limits. The
  default of 20 rows keeps requests small and avoids long prompts.
* **Cost awareness** – LLM calls can be expensive. Estimate token usage from the
  prompt template and size your batches accordingly. For exploratory runs, point the
  client at a lower-cost model or reduce the batch size.

### Using mlx-lm locally

Apple silicon users can rely on [`mlx-lm`](https://github.com/ml-explore/mlx-examples)
for local inference. Load the model with `mlx_lm.utils.load` (or `mlx_lm.load`) and
pass a configuration dictionary to `load_and_prepare`:

```python
from mlx_lm import load as mlx_load

model, tokenizer = mlx_load("mlx-community/Llama-3.2-3B-Instruct-4bit")

client = {
    "framework": "mlx_lm",  # tells the helper to use the mlx integration
    "model": model,
    "tokenizer": tokenizer,
    "generation_kwargs": {"max_tokens": 128},  # forwarded to mlx_lm.generate
}

prepared = load_and_prepare(path_to_csv, llm_client=client, llm_batch_size=10)
```

Set `use_batch=False` in the configuration if you prefer to call `mlx_lm.generate`
per prompt instead of batching through `mlx_lm.batch_generate`. Remember that mlx-lm
requires the base [`mlx`](https://pypi.org/project/mlx/) package, which is only
available on recent macOS releases.

### Offline or air-gapped usage

Leave `llm_client=None` (the default) to skip enrichment entirely. The OFX generation
pipeline remains fully functional without model calls, making it safe to run offline or
in environments without API access.
