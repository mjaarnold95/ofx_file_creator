"""Utilities for enriching transactions with large language models."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence

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

        mlx_result = _try_mlx_lm_invocation(self.client, prompts)
        if mlx_result is not None:
            return mlx_result
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

def _try_mlx_lm_invocation(client: object, prompts: Sequence[str]) -> Optional[List[str]]:
    """Attempt to route prompts through an ``mlx-lm`` client configuration."""

    config = _resolve_mlx_client_config(client)
    if config is None:
        return None

    try:
        # Import lazily so downstream projects without mlx-lm installed still work.
        import importlib

        mlx_module = importlib.import_module("mlx_lm")
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "mlx_lm is required for this client configuration but could not be imported"
        ) from exc

    batch_fn = config.get("batch_generate") or getattr(mlx_module, "batch_generate", None)
    generate_fn = config.get("generate") or getattr(mlx_module, "generate", None)

    model = config["model"]
    tokenizer = config["tokenizer"]
    tokenize_kwargs = config.get("tokenize_kwargs", {})
    generation_kwargs = dict(config.get("generation_kwargs", {}))
    max_tokens = generation_kwargs.pop("max_tokens", None)
    sampling_params = config.get("sampling_params")
    if sampling_params is not None:
        generation_kwargs.setdefault("sampling_params", sampling_params)

    if batch_fn is not None and config.get("use_batch", True):
        batch_kwargs = dict(generation_kwargs)
        if max_tokens is not None:
            batch_kwargs.setdefault("max_tokens", max_tokens)
        if tokenize_kwargs:
            batch_kwargs.setdefault("tokenize_kwargs", tokenize_kwargs)
        responses = batch_fn(model, tokenizer, prompts, **batch_kwargs)
        texts = _extract_mlx_batch_texts(responses)
        return [str(text) for text in texts]

    if generate_fn is None:
        raise AttributeError(
            "mlx_lm client must supply a 'generate' function when batch generation is disabled"
        )

    single_kwargs = dict(generation_kwargs)
    if max_tokens is not None:
        single_kwargs.setdefault("max_tokens", max_tokens)
    if tokenize_kwargs:
        single_kwargs.setdefault("tokenize_kwargs", tokenize_kwargs)

    outputs = []
    for prompt in prompts:
        response = generate_fn(model, tokenizer, prompt=prompt, **single_kwargs)
        outputs.append(_extract_mlx_single_text(response))
    return outputs


def _resolve_mlx_client_config(client: object) -> Optional[Dict[str, Any]]:
    """Extract mlx-lm configuration from supported client shapes."""

    data: Optional[Dict[str, Any]] = None
    if isinstance(client, Mapping):
        data = dict(client)
    elif isinstance(client, tuple) and len(client) in (2, 3):
        data = {"model": client[0], "tokenizer": client[1]}
        if len(client) == 3 and isinstance(client[2], Mapping):
            data.update(client[2])

    if not data:
        return None

    framework = None
    for key in ("framework", "provider", "type"):
        if key in data:
            framework = str(data[key])
            break
    explicit_flag = any(
        key in data and bool(data[key]) for key in ("mlx_lm", "use_mlx_lm")
    )
    if framework:
        if "mlx" not in framework.lower():
            return None
    elif not explicit_flag:
        return None

    model = data.get("model")
    tokenizer = data.get("tokenizer")
    if model is None or tokenizer is None:
        raise ValueError(
            "mlx_lm configuration must include both 'model' and 'tokenizer' references"
        )

    generation_kwargs = dict(data.get("generation_kwargs", {}))
    if "max_tokens" in data and "max_tokens" not in generation_kwargs:
        generation_kwargs["max_tokens"] = data["max_tokens"]

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "generation_kwargs": generation_kwargs,
        "use_batch": data.get("use_batch", True),
    }
    if "batch_generate" in data:
        config["batch_generate"] = data["batch_generate"]
    if "generate" in data:
        config["generate"] = data["generate"]
    if "tokenize_kwargs" in data:
        config["tokenize_kwargs"] = dict(data["tokenize_kwargs"])
    if "sampling_params" in data:
        config["sampling_params"] = data["sampling_params"]

    return config

def _extract_mlx_batch_texts(responses: Any) -> List[str]:
    """Normalize batch generation outputs into a list of strings."""

    texts = None
    if hasattr(responses, "texts"):
        texts = responses.texts
    elif isinstance(responses, Mapping) and "texts" in responses:
        texts = responses["texts"]
    else:
        texts = responses

    if hasattr(texts, "tolist"):
        texts = texts.tolist()

    if isinstance(texts, Mapping):
        if "text" in texts:
            texts = texts["text"]
        elif "completions" in texts:
            texts = texts["completions"]

    if isinstance(texts, str):
        return [texts]

    if isinstance(texts, Sequence):
        return [str(item) for item in texts]

    return [str(texts)]


def _extract_mlx_single_text(response: Any) -> str:
    """Normalize a single generation output into a string."""

    if isinstance(response, str):
        return response

    if hasattr(response, "text"):
        return str(response.text)

    if hasattr(response, "generated_text"):
        return str(response.generated_text)

    if hasattr(response, "completion"):
        return str(response.completion)

    if isinstance(response, Mapping):
        for key in ("text", "generated_text", "completion"):
            if key in response:
                value = response[key]
                if isinstance(value, Sequence) and not isinstance(value, str):
                    if value:
                        return str(value[0])
                return str(value)
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if isinstance(choice, Mapping):
                if "text" in choice:
                    return str(choice["text"])
                if "message" in choice and isinstance(choice["message"], Mapping):
                    content = choice["message"].get("content")
                    if content is not None:
                        return str(content)

    if isinstance(response, Sequence) and not isinstance(response, str):
        first = response[0]
        if isinstance(first, Mapping):
            if "text" in first:
                return str(first["text"])
            if "generated_text" in first:
                return str(first["generated_text"])
        return str(first)

    return str(response)
