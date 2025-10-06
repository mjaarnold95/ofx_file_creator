from pathlib import Path


_ACCOUNT_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CHECKING", ("checking", "chk")),
    ("SAVINGS", ("savings", "hys")),
    (
        "CREDITCARD",
        (
            "credit",
            "cc",
            "creditcard",
            "credit-card",
            "visa",
            "mastercard",
            "discover",
            "amex",
            "americanexpress",
            "american express",
        ),
    ),
    ("BROKERAGE", ("brokerage", "investment", "invest", "espp", "trust")),
    ("LOAN", ("loan", "mortgage", "mtg")),
    ("MONEYMRKT", ("cash", "money market", "mm")),
    ("HSA", ("hsa", "health savings")),
    ("FSA", ("fsa", "flexible spending")),
    ("_401K", ("401k", "401-k", "401 k")),
    ("_403B", ("403b", "403-b", "403 b")),
    ("_457B", ("457b", "457-b", "457 b")),
    ("ANNUITY", ("annuity",)),
    (
        "BROKERAGE",
        (
            "ira",
            "roth",
            "sep",
            "simple",
            "keogh",
            "401a",
            "401-a",
            "401 a",
            "457f",
            "457-f",
            "457 f",
            "pension",
            "529",
            "college savings",
        ),
    ),
)


def detect_account_type(path: Path) -> str:
    stem = path.stem.lower()
    for acct_type, keywords in _ACCOUNT_TYPE_KEYWORDS:
        if any(keyword in stem for keyword in keywords):
            return acct_type
    return "CHECKING"
