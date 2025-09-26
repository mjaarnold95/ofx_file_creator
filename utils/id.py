import hashlib
import re
from pathlib import Path

import pandas as pd

from utils.date_time import ofx_datetime

# ---------- ids ----------
def make_fitid(row: pd.DataFrame, idx: int) -> str:
    if "fitid" not in row.any() or pd.notna(row["fitid"]).empty:
        parts = [
            ofx_datetime(row.get("date_parsed")),
            f"{row.get('amount_clean', '')}",
            str(row.get("cleaned_desc", ""))[:64],
            str(row.get("memo", ""))[:64],
            str(row.get("name", ""))[:64],
            str(idx),
            ]
        return hashlib.md5("|".join(parts).encode()).hexdigest().upper()
    return str(row["fitid"])[:32]

def derive_acctid_from_path(path: Path, default_stub: str) -> str:
    digits = re.sub(r"[^0-9]", "", path.stem)
    if digits:
        return digits
    h = hashlib.md5(path.stem.encode()).hexdigest()[:10]
    return f"{default_stub}{h}".upper()