import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.build_ofx import build_ofx
from utils.date_time import ofx_datetime, parse_time_to_timedelta
from utils.id import make_fitid


def test_ofx_datetime_formats_timestamp():
    ts = pd.Timestamp("2023-05-01 12:34:56", tz="UTC")
    assert ofx_datetime(ts) == "20230501123456.000[0:UTC]"


def test_ofx_datetime_handles_none():
    assert ofx_datetime(None) is None
    assert ofx_datetime(pd.NaT) is None


def test_parse_time_to_timedelta_parses_strings():
    td = parse_time_to_timedelta("1:30 PM")
    assert td == pd.to_timedelta(13, unit="h") + pd.to_timedelta(30, unit="m")


def test_make_fitid_prefers_existing_value():
    row = pd.Series({"fitid": " existing ", "amount_clean": 10})
    assert make_fitid(row, 0) == "existing"


def test_make_fitid_generates_deterministic_hash():
    row = pd.Series(
        {
            "fitid": None,
            "fitid_norm": None,
            "date_parsed": pd.Timestamp("2023-05-01", tz="UTC"),
            "amount_clean": 12.34,
            "cleaned_desc": "Test Transaction",
            "memo": "memo",
            "name": "name",
        }
    )
    expected = hashlib.md5(
        "20230501000000.000[0:UTC]|12.34|Test Transaction|memo|name|0".encode()
    ).hexdigest().upper()
    assert make_fitid(row, 0) == expected


def test_build_ofx_uses_transaction_date_range():
    df = pd.DataFrame(
        {
            "date_parsed": [
                pd.Timestamp("2023-01-01", tz="UTC"),
                pd.Timestamp("2023-01-03", tz="UTC"),
            ],
            "amount_clean": [10.0, -5.0],
            "trntype_norm": ["CREDIT", "DEBIT"],
            "fitid_norm": ["ABC123", None],
            "cleaned_desc": ["Deposit", "Withdrawal"],
        }
    )

    ofx_text = build_ofx(df, accttype="checking", acctid="12345")

    assert "<DTSTART>20230101000000.000[0:UTC]</DTSTART>" in ofx_text
    assert "<DTEND>20230103000000.000[0:UTC]</DTEND>" in ofx_text
    assert "<FITID>ABC123</FITID>" in ofx_text
