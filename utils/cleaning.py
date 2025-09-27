import re
from typing import Optional

import numpy as np
import pandas as pd

from utils.rules import DEFAULT_RULES, RuleSet


# ---------- cleaning ----------
def clean_amount_series(values: pd.Series) -> pd.Series:
    """Vectorized parse of currency-like strings into floats."""
    if values.empty:
        return pd.Series([], index=values.index, dtype="float64")

    result = pd.Series(np.nan, index=values.index, dtype="float64")
    str_vals = values.astype("string").str.strip()
    str_vals = str_vals.replace({"": pd.NA})
    cleaned = str_vals.str.replace(",", "")

    paren_mask = cleaned.str.startswith("(") & cleaned.str.endswith(")")
    if paren_mask.any():
        paren_vals = cleaned.loc[paren_mask].str[1:-1].str.replace("$", "").str.strip()
        result.loc[paren_mask] = -pd.to_numeric(paren_vals, errors="coerce")

    remaining_mask = ~paren_mask & cleaned.notna()
    if remaining_mask.any():
        remaining_vals = cleaned.loc[remaining_mask].str.replace("$", "")
        result.loc[remaining_mask] = pd.to_numeric(remaining_vals, errors="coerce")

    return result


def clean_amount(x):
    series = pd.Series([x])
    val = clean_amount_series(series).iloc[0]
    return val


def clean_description(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip()).upper()


# trntype inference has been moved to `utils.trntype` to separate concerns.
from utils.trntype import infer_trntype, infer_trntype_series  # noqa: E402


__all__ = [
    "clean_amount",
    "clean_amount_series",
    "clean_description",
    # re-exported for compatibility; prefer utils.trntype
    "infer_trntype",
    "infer_trntype_series",
]
