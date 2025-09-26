import re
from typing import Optional

import numpy as np
import pandas as pd

from utils.rules import DEFAULT_RULES, RuleSet

# ---------- cleaning & trntype ----------
def clean_amount_series(values: pd.Series) -> pd.Series:
    """Vectorized parse of currency-like strings into floats."""
    if values.empty:
        return pd.Series([], index=values.index, dtype="float64")
    
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    str_vals = values.astype("string").str.strip()
    str_vals = str_vals.replace({"":pd.NA})
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

_OFX_TYPE_WHITELIST = {
    "CASH",
    "INT",
    "DIV",
    "FEE",
    "SRVCHG",
    "DEP",
    "ATM",
    "POS",
    "XFER",
    "CHECK",
    "PAYMENT",
    "DIRECTDEP",
    "DIRECTDEBIT",
    "REPEATPMT",
    "OTHER",
    "CREDIT",
    "DEBIT",
    }

# noinspection PyUnresolvedReferences
def infer_trntype_series(
    amount: pd.Series,
    trntype_text: Optional[pd.Series],
    cleaned_desc: Optional[pd.Series] = None,
    rules: Optional[RuleSet] = None,
    ) -> pd.Series:
    """Infer OFX transaction type values for a series of transactions.

    Parameters
    ----------
    amount:
        Transaction amount values.
    trntype_text:
        Raw trntype column values which may contain aliases.
    cleaned_desc:
        Optional cleaned description column to use for pattern matching.
    rules:
        Optional :class:`~utils.rules.RuleSet` instance describing the
        normalization aliases and matching rules to use. When not provided the
        :data:`utils.rules.DEFAULT_RULES` collection is used.
    """

    rules = rules or DEFAULT_RULES
    idx = amount.index
    trn_series = (
        trntype_text if trntype_text is not None else pd.Series(pd.NA, index=idx)
    )
    desc_series = (
        cleaned_desc if cleaned_desc is not None else pd.Series(pd.NA, index=idx)
    )

    trn_text = trn_series.astype("string").str.strip().str.upper()
    normalized = trn_text.replace(rules.source_aliases)

    result = pd.Series(pd.NA, index=idx, dtype="string")
    exact_mask = normalized.isin(_OFX_TYPE_WHITELIST)
    result.loc[exact_mask] = normalized.loc[exact_mask]

    haystack = (
        trn_text.fillna("") + " " + desc_series.astype("string").fillna("")
    ).str.upper()
    haystack = haystack.str.strip()

    pending = result.isna()
    for pattern, output in rules.rules_regex:
        if not pending.any():
            break
        mask = pending & haystack.str.contains(pattern, regex=True, na=False)
        result.loc[mask] = output
        pending = result.isna()

    for pattern, output in rules.keyword_rules:
        if not pending.any():
            break
        mask = pending & haystack.str.contains(pattern, regex=True, na=False)
        result.loc[mask] = output
        pending = result.isna()

    if pending.any():
        numeric_amounts = pd.to_numeric(amount, errors="coerce")
        other_mask = pending & numeric_amounts.isna()
        result.loc[other_mask] = "OTHER"
        pending = result.isna()
        if pending.any():
            amt_values = numeric_amounts.loc[pending]
            result.loc[pending] = np.where(amt_values < 0, "DEBIT", "CREDIT")

    return result.fillna("OTHER")


def infer_trntype(
    amount,
    trntype_text: Optional[str],
    cleaned_desc: Optional[str] = None,
    rules: Optional[RuleSet] = None,
    ) -> str:
    series = infer_trntype_series(
        pd.Series([amount]),
        pd.Series([trntype_text]),
        pd.Series([cleaned_desc]),
        rules=rules,
    )
    val = series.iloc[0]
    return "OTHER" if pd.isna(val) else str(val)
