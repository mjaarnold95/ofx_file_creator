from pathlib import Path

# ---------- acct type ----------
def detect_account_type(path: Path) -> str:
    stem = path.stem.lower()
    if "checking" in stem or "chk" in stem:
        return "CHECKING"
    if "savings" in stem or "hys" in stem:
        return "SAVINGS"
    return "CHECKING"