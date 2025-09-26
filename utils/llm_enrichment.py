"""Utilities for enriching transactions with large language models."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "You are a financial data enrichment assistant.\n"
    "Given details about a bank transaction, infer the best-guess payee name,\n"
    "the spending category, and a concise user-facing description.\n"
    "Respond ONLY with minified JSON following this schema:\n"
    '{{"payee": string | null, "category": string | null, "description": string | null}}.\n'
    "If you are unsure, use null for that field.\n\n"
    "Transaction data:\n"
    "- Raw description: {raw_desc}\n"
    "- Payee display: {payee_display}\n"
    "- Cleaned description: {cleaned_desc}\n\n"
    "Return only the JSON object with double quotes."
)


@dataclass
class _ClientInvoker:
    client: object

    def invoke(self, prompts: Sequence[str]) -> List[str]:
        """Call the provided LLM client and return a list of response strings."""

        if hasattr(self.client, "generate_batch"):
            return list(self.client.generate_batch(list(prompts)))

        if callable(self.client):
            return [str(self.client(prompt)) for prompt in prompts]

        if hasattr(self.client, "generate"):
            return [str(getattr(self.client, "generate")(prompt)) for prompt in prompts]

        raise AttributeError(
            "LLM client must provide a 'generate_batch', 'generate', or be callable"
        )


def _coerce_str(value: object):
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else pd.NA


def _safe_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value)
    return text if text else ""


def _parse_response(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw

    text = str(raw).strip()
    if not text:
        raise ValueError("empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        return json.loads(snippet)

    raise ValueError("Could not parse JSON from response")


def enrich_transactions_with_llm(
    df: pd.DataFrame,
    client,
    *,
    batch_size: int = 20,
) -> pd.DataFrame:
    """Use an LLM to enrich transaction data.

    Parameters
    ----------
    df: pandas.DataFrame
        Must contain ``raw_desc``, ``payee_display`` and ``cleaned_desc`` columns.
    client: object
        LLM client supporting ``generate_batch`` (preferred), ``generate``, or
        being directly callable with a prompt string.
    batch_size: int, default 20
        Number of rows per batch request.
    """

    required_cols = {"raw_desc", "payee_display", "cleaned_desc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "DataFrame must contain the columns: " + ", ".join(sorted(missing))
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    invoker = _ClientInvoker(client)

    index = list(df.index)
    prompts = [
        _PROMPT_TEMPLATE.format(
            raw_desc=_safe_value(df.at[idx, "raw_desc"]),
            payee_display=_safe_value(df.at[idx, "payee_display"]),
            cleaned_desc=_safe_value(df.at[idx, "cleaned_desc"]),
        )
        for idx in index
    ]

    payee_series = pd.Series(pd.NA, index=index, dtype="string")
    category_series = pd.Series(pd.NA, index=index, dtype="string")
    description_series = pd.Series(pd.NA, index=index, dtype="string")

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_indices = index[start : start + len(batch_prompts)]
        try:
            responses = invoker.invoke(batch_prompts)
        except Exception as exc:  # pragma: no cover - logging side-effect
            logger.warning("LLM enrichment request failed: %s", exc)
            continue

        if len(responses) != len(batch_prompts):
            logger.warning(
                "LLM client returned %s responses for %s prompts",
                len(responses),
                len(batch_prompts),
            )
            continue

        for idx, raw_response in zip(batch_indices, responses):
            try:
                data = _parse_response(raw_response)
            except Exception as exc:  # pragma: no cover - logging side-effect
                logger.debug("Could not parse LLM response for index %s: %s", idx, exc)
                continue

            payee_series.at[idx] = _coerce_str(data.get("payee"))
            category_series.at[idx] = _coerce_str(data.get("category"))
            description_series.at[idx] = _coerce_str(data.get("description"))

    return pd.DataFrame(
        {
            "payee_llm": payee_series,
            "category_llm": category_series,
            "description_llm": description_series,
        }
    )
