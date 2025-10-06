"""Transaction-type inference helpers.

This module contains functions that apply the rule-driven transaction-type
inference logic. Rules are configured via `utils.rules.RuleSet` and can be
loaded/overridden from JSON/YAML using `utils.rules.load_rules`.

Prefer importing inference functions from `utils.trntype` in new code. The
legacy module `utils.cleaning` re-exports these functions to preserve
backwards-compatibility.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd

from utils.rules import DEFAULT_RULES, RuleSet


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


def infer_trntype_series(
    amount: pd.Series,
    trntype_text: Optional[pd.Series],
    cleaned_desc: Optional[pd.Series] = None,
    rules: Optional[RuleSet] = None,
) -> pd.Series:
    """Infer OFX transaction type values for a series of transactions.

    This function was previously defined in `utils.cleaning`. It has been
    moved here to separate cleaning utilities from trntype inference logic.
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
    for regex_pattern, output in rules.rules_regex:
        if not pending.any():
            break
        mask = pending & haystack.str.contains(regex_pattern, regex=True, na=False)
        result.loc[mask] = output
        pending = result.isna()

    for keyword_pattern, output in rules.keyword_rules:
        if not pending.any():
            break
        mask = pending & haystack.str.contains(keyword_pattern, regex=True, na=False)
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
