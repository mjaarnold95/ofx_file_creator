import hashlib
import re
from pathlib import Path

import pandas as pd

from utils.date_time import ofx_datetime


# ---------- ids ----------
def _normalize_fitid(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value).strip()
    return "" if not text or text.lower() == "nan" else text


def make_fitid(row: pd.Series, idx: int) -> str:
    fitid = _normalize_fitid(row.get("fitid")) or _normalize_fitid(
        row.get("fitid_norm")
    )
    if fitid:
        return fitid[:32]

    parts = [
        ofx_datetime(row.get("date_parsed")) or "",
        f"{row.get('amount_clean', '')}",
        str(row.get("cleaned_desc", ""))[:64],
        str(row.get("memo", ""))[:64],
        str(row.get("name", ""))[:64],
        str(idx),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest().upper()

def derive_acctid_from_path(path: Path, default_stub: str) -> str:
    digits = re.sub(r"[^0-9]", "", path.stem)
    if digits:
        return digits
    h = hashlib.md5(path.stem.encode()).hexdigest()[:10]
    return f"{default_stub}{h}".upper()
