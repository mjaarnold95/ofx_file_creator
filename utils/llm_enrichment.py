"""Utilities for enriching transactions with large language models."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Sequence as Seq

import pandas as pd

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "You are a financial data enrichment assistant.\n"
    "Given details about a bank transaction, or by researching the transaction details online if unknown, generate the best-guess payee name,\n"
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


def _normalize_responses(obj: Any) -> List[Any]:
    """Return *obj* as a list while preserving common response containers."""

    if obj is None:
        return []

    if isinstance(obj, str):
        return [obj]

    if isinstance(obj, (bytes, bytearray)):
        return [obj.decode("utf-8", "replace")]

    if isinstance(obj, Mapping):  # e.g. {"text": "..."} or {"responses": [...]}
        # Try to find common container keys
        for key in ("responses", "texts", "choices", "text"):
            if key in obj and obj[key] is not None:
                val = obj[key]
                # If text is a single value, wrap; if sequence, normalize to list
                if isinstance(val, str) or not isinstance(val, Sequence):
                    return [val]
                return list(val)
        # Fallback: treat mapping itself as one response
        return [obj]

    if isinstance(obj, Sequence):
        return list(obj)

    return [obj]


def _coerce_response_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return value


@dataclass
class _ClientInvoker:
    client: object

    def _resolve_callable(self, name: str) -> Optional[Callable[[Seq[str]], Any]]:
        # If client has attribute that is callable, return a wrapper
        attr = getattr(self.client, name, None)
        if callable(attr):
            return lambda prompts: attr(prompts)
        # If client is a mapping and contains a callable under the name, return wrapper
        if isinstance(self.client, Mapping):
            fn = self.client.get(name)
            if callable(fn):
                return lambda prompts: fn(prompts)
        return None

    def invoke(self, prompts: Seq[str]) -> List[Any]:
        # Prefer specialized mlx-lm route
        try:
            mlx_candidates = _try_mlx_lm_invocation(self.client, prompts)
            if mlx_candidates is not None:
                return mlx_candidates
        except Exception:
            # Log and continue to other invocation methods
            logger.debug(
                "mlx-lm invocation failed; falling back to generic client",
                exc_info=True,
            )

        # Try generate_batch style
        batch_fn = self._resolve_callable("generate_batch") or self._resolve_callable(
            "batch_generate"
        )
        if batch_fn is not None:
            res = batch_fn(prompts)
            return _normalize_responses(res)

        # Try single-call generate function (will be invoked per prompt)
        gen_fn = self._resolve_callable("generate")
        if gen_fn is not None:
            outputs = []
            for p in prompts:
                try:
                    out = gen_fn([p]) if False else gen_fn(p)  # prefer direct call
                except TypeError:
                    # Some generate APIs expect keyword args or single-prompt lists; attempt both patterns
                    try:
                        out = gen_fn(prompt=p)
                    except Exception as exc:
                        raise
                outputs.append(out)
            return _normalize_responses(outputs)

        # If client is directly callable, try calling with list first
        if callable(self.client):
            try:
                out = self.client(prompts)
                normalized = _normalize_responses(out)
                if len(normalized) == len(prompts):
                    return normalized
                # Not a batch response -> fall back to per-item calls
            except TypeError:
                pass
            # Call per-prompt
            outputs = [self.client(p) for p in prompts]
            return _normalize_responses(outputs)

        # As last resort, attempt to treat client as mapping with 'model'/'tokenizer' (handled in mlx function)
        raise RuntimeError("LLM client does not expose a supported invocation method")


def _coerce_str(value: Any) -> object:
    """Coerce arbitrary input into a trimmed string or pandas.NA.

    We accept Any here and carefully call pd.isna inside a try/except to
    avoid mypy overload issues when the argument is a generic object.
    """
    if value is None:
        return pd.NA
    # Guard usage of pd.isna: it has several overloads and some don't accept
    # bare 'object' typed values. Use try/except to preserve runtime behavior
    # while keeping mypy satisfied.
    try:
        if pd.isna(value):
            return pd.NA
    except Exception:
        pass
    text = str(value).strip()
    return text if text else pd.NA


def _safe_value(value: Any) -> str:
    """Return a safe string for display; empty string for None/NA values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
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

    # Try to extract JSON substring
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
        except Exception as exc:
            logger.exception("LLM invocation failed for batch starting at %s", start)
            # Leave these entries as NA and continue
            continue

        if len(responses) != len(batch_prompts):
            # Try to normalize: if single string for whole batch, attempt to split into per-prompt responses
            logger.debug(
                "Unexpected response length from LLM: %s (expected %s). Normalizing.",
                len(responses),
                len(batch_prompts),
            )
            # If single combined response, attempt to parse it as one JSON and replicate (best-effort)
            if len(responses) == 1:
                responses = [responses[0]] * len(batch_prompts)
            else:
                # Fall back: skip this batch
                logger.warning("Skipping LLM batch due to mismatched response count")
                continue

        for idx, raw_response in zip(batch_indices, responses):
            try:
                parsed = _parse_response(raw_response)
            except Exception:
                # If parsing fails, try to coerce raw string into something useful
                logger.debug("Failed to parse LLM response; storing NA", exc_info=True)
                continue

            # Extract and coerce values
            payee_series.at[idx] = _coerce_str(parsed.get("payee"))
            category_series.at[idx] = _coerce_str(parsed.get("category"))
            description_series.at[idx] = _coerce_str(parsed.get("description"))

    return pd.DataFrame(
        {
            "payee_llm": payee_series,
            "category_llm": category_series,
            "description_llm": description_series,
        }
    )


def _try_mlx_lm_invocation(client: object, prompts: Seq[str]) -> Optional[List[str]]:
    """Attempt to route prompts through an ``mlx-lm`` client configuration."""

    config = _resolve_mlx_client_config(client)
    if config is None:
        return None

    try:
        mlx_module = importlib.import_module("mlx_lm")
    except Exception as exc:
        raise RuntimeError(
            "mlx_lm is required for this client configuration but could not be imported"
        ) from exc

    batch_fn = config.get("batch_generate") or getattr(
        mlx_module, "batch_generate", None
    )
    generate_fn = config.get("generate") or getattr(mlx_module, "generate", None)

    model = config["model"]
    tokenizer = config["tokenizer"]
    generation_kwargs = dict(config.get("generation_kwargs", {}))
    use_batch = config.get("use_batch", True)

    if batch_fn is not None and use_batch:
        # Call batch function and extract texts
        responses = batch_fn(model, tokenizer, prompts, **generation_kwargs)
        texts = _extract_mlx_batch_texts(responses)
        return [str(t) for t in texts]

    if generate_fn is None:
        raise AttributeError(
            "mlx_lm client must supply a 'generate' function when batch generation is disabled"
        )

    outputs = []
    for prompt in prompts:
        response = generate_fn(model, tokenizer, prompt=prompt, **generation_kwargs)
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

    llm_model_value = str(data.setdefault("llm_model", "client") or "client")
    llm_model_lower = llm_model_value.lower()

    framework = None
    if llm_model_lower != "client":
        framework = llm_model_value
    else:
        for key in ("framework", "provider", "type"):
            if key in data:
                framework = str(data[key])
                break

    explicit_flag = any(
        key in data and bool(data[key]) for key in ("mlx_lm", "use_mlx_lm")
    )
    if llm_model_lower != "client":
        explicit_flag = True

    if framework and "mlx" not in framework.lower() and not explicit_flag:
        return None
    elif not framework and not explicit_flag:
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
        try:
            texts = texts.tolist()
        except Exception:
            pass

    if isinstance(texts, Mapping):
        # Common mapping shapes
        if "text" in texts:
            return [str(texts["text"])]
        if "completions" in texts:
            comps = texts["completions"]
            if isinstance(comps, Sequence):
                out = []
                for c in comps:
                    if isinstance(c, Mapping):
                        out.append(
                            str(
                                c.get("text")
                                or c.get("generated_text")
                                or c.get("completion")
                                or c
                            )
                        )
                    else:
                        out.append(str(c))
                return out
            return [str(comps)]
        # Fall back to stringifying mapping
        return [json.dumps(texts)]

    if isinstance(texts, str):
        return [texts]

    if isinstance(texts, Sequence):
        normalized = []
        for item in texts:
            try:
                normalized.append(_extract_mlx_single_text(item))
            except Exception:
                normalized.append(str(item))
        return normalized

    return [str(texts)]


def _extract_mlx_single_text(response: Any) -> str:
    """Normalize a single generation output into a string."""

    if isinstance(response, str):
        return response

    if isinstance(response, Mapping):
        # Try common keys
        for key in ("text", "generated_text", "completion"):
            if key in response:
                return str(response[key])
        if "choices" in response and response["choices"]:
            first = response["choices"][0]
            if isinstance(first, Mapping):
                for key in ("text", "generated_text", "completion"):
                    if key in first:
                        return str(first[key])
                return str(first)
            return str(first)
        return json.dumps(response)

    if hasattr(response, "text"):
        return str(response.text)

    if hasattr(response, "generated_text"):
        return str(response.generated_text)

    if hasattr(response, "completion"):
        return str(response.completion)

    if isinstance(response, Sequence) and not isinstance(response, str):
        first = response[0]
        if isinstance(first, Mapping):
            for key in ("text", "generated_text", "completion"):
                if key in first:
                    return str(first[key])
            return str(first)
        return str(first)

    return str(response)
