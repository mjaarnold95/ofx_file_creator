import re
from typing import Optional

import numpy as np
import pandas as pd

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

_SOURCE_ALIASES = {
    "D":"DEBIT",
    "DR":"DEBIT",
    "DBT":"DEBIT",
    "WITHDRAWAL":"DEBIT",
    "W/D":"DEBIT",
    "WD":"DEBIT",
    "DEPOSIT":"DEP",
    "DEP":"DEP",
    "C":"CREDIT",
    "CR":"CREDIT",
    "XFR":"XFER",
    "TRANSFER":"XFER",
    "PAY":"PAYMENT",
    "PMT":"PAYMENT",
    }

_RULES_REGEX = [
    (re.compile(r"\b(?:VENMO|CASH\s+APP|ZELLE|APPLE\s+CASH|P2P)\b", re.I), "PAYMENT"),
    (re.compile(r"\bMOBILE\s+DEPOSIT\b", re.I), "DEP"),
    (re.compile(r"\bCHECK\b", re.I), "CHECK"),
    (re.compile(r"\bATM\b", re.I), "ATM"),
    (re.compile(r"\bCASH(?:\s+(?:WITHDRAWAL|DEPOSIT))?\b", re.I), "CASH"),
    (re.compile(r"\bPOS(?:\s+PURCHASE)?\b", re.I), "POS"),
    (
        re.compile(
            r"\b(?:UTIL(?:ITY)?|ACH|PPD|CCD|(?:AUTO|AUTOP|BILL|DIRECT|ONLINE|WEB|E(?:LECTRONIC)?)[-_/.\s]*(?:P("
            r"?:AY)?(?:MENT|MNT|MT)?|PMT|PMNT|PYMT|PYMNT))\b",
            re.I,
            ),
        "DIRECTDEBIT",
        ),
    (
        re.compile(
            r"\b(?:PAYROLL|IRS(?:\s*REFUND)?|SSA|SOCIAL\s+SECURITY|(?:STATE\s*)?TREAS)\b",
            re.I,
            ),
        "DIRECTDEP",
        ),
    (re.compile(r"\b(?:TRANSFER|(?:EXT-|EXTERNAL\s*)?XFER|XFR)\b", re.I), "XFER"),
    (re.compile(r"\b(?:INT(?:EREST)?|FINANCE\s*CHARGE|APR)\b", re.I), "INT"),
    (
        re.compile(
            r"\b(?:SERVICE\s*CHARGE|MONTHLY\s*SERVICE|MAINT(?:ENANCE)?\s*FEE)\b",
            re.I,
            ),
        "SRVCHG",
        ),
    (
        re.compile(
            r"\b(?:OVERDRAFT|NSF|WIRE\s*FEE|RTN\s*ITEM(?:\s*FEE)?|STOP\s*PAY(?:MENT)?\s*FEE|FEE)\b",
            re.I,
            ),
        "FEE",
        ),
    (re.compile(r"\bDIV(?:IDEND)?\b", re.I), "DIV"),
    (re.compile(r"\bREV(?:ERSAL)?\b", re.I), "CREDIT"),
    (re.compile(r"\bRETURN(?!ED\s+ITEM\s+FEE)\b", re.I), "CREDIT"),
    (re.compile(r"\bE-?PAY(?:MENT)?\b", re.I), "DIRECTDEBIT"),
    (re.compile(r"\bPAYMNT\b", re.I), "DIRECTDEBIT"),
    (re.compile(r"\bACH\s*PAY(?:MENT)?\b", re.I), "DIRECTDEBIT"),
    (re.compile(r"\bDISCOVER\s*E-?PAYMENT\b", re.I), "DIRECTDEBIT"),
    ]

_KEYWORD_RULES = [
    (r"\bDEPOSIT\b", "DEP"),
    (r"\bINTEREST\b", "INT"),
    (r"\bINT\b", "INT"),
    (r"\bDIVIDEND\b", "DIV"),
    (r"\bDIV\b", "DIV"),
    (r"\bSRVCHG\b", "FEE"),
    (r"\bFEE\b", "FEE"),
    (r"\bCHECK\b", "CHECK"),
    (r"\bATM\b", "ATM"),
    (r"\bPURCHASE\b", "POS"),
    (r"\bPOS\b", "POS"),
    (r"\bTRANSFER\b", "XFER"),
    (r"\bXFER\b", "XFER"),
    (r"\bXFR\b", "XFER"),
    (r"\bWITHDRAW\b", "DEBIT"),
    (r"\bWD\b", "DEBIT"),
    (r"\bPAYMENT\b", "PAYMENT"),
    (r"\bPMT\b", "PAYMENT"),
    (r"\bREFUND\b", "CREDIT"),
    (r"\bPAYROLL\b", "DIRECTDEP"),
    (r"\bCASH\b", "CASH"),
    ]

# noinspection PyUnresolvedReferences
def infer_trntype_series(
    amount: pd.Series,
    trntype_text: Optional[pd.Series],
    cleaned_desc: Optional[pd.Series] = None,
    ) -> pd.Series:
    idx = amount.index
    trn_series = (
        trntype_text if trntype_text is not None else pd.Series(pd.NA, index=idx)
    )
    desc_series = (
        cleaned_desc if cleaned_desc is not None else pd.Series(pd.NA, index=idx)
    )
    
    trn_text = trn_series.astype("string").str.strip().str.upper()
    normalized = trn_text.replace(_SOURCE_ALIASES)
    
    result = pd.Series(pd.NA, index=idx, dtype="string")
    exact_mask = normalized.isin(_OFX_TYPE_WHITELIST)
    result.loc[exact_mask] = normalized.loc[exact_mask]
    
    haystack = (trn_text.fillna("") + " " + desc_series.astype("string").fillna("")).str.upper()
    haystack = haystack.str.strip()
    
    pending = result.isna()
    for pattern, output in _RULES_REGEX:
        if not pending.any():
            break
        mask = pending & haystack.str.contains(pattern, regex=True, na=False)
        result.loc[mask] = output
        pending = result.isna()
    
    for pattern, output in _KEYWORD_RULES:
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
    amount, trntype_text: Optional[str], cleaned_desc: Optional[str] = None
    ) -> str:
    series = infer_trntype_series(
        pd.Series([amount]),
        pd.Series([trntype_text]),
        pd.Series([cleaned_desc]),
        )
    val = series.iloc[0]
    return "OTHER" if pd.isna(val) else str(val)