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

### Offline or air-gapped usage

Leave `llm_client=None` (the default) to skip enrichment entirely. The OFX generation
pipeline remains fully functional without model calls, making it safe to run offline or
in environments without API access.
