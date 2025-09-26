import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.build_ofx import build_ofx
from utils.date_time import ofx_datetime, parse_time_to_timedelta
from utils.id import make_fitid
from utils.validate import assert_ofx_ready


@pytest.fixture
def df_without_dates():
    return pd.DataFrame(
        {
            "amount_clean": [25.0],
            "cleaned_desc": ["No Date Transaction"],
            "trntype_norm": ["CREDIT"],
            "statement_end_date": [pd.Timestamp("2023-04-30", tz="UTC")],
        }
    )


def test_assert_ofx_ready_accepts_valid_dataframe():
    df = pd.DataFrame(
        {
            "amount_clean": [10.0, 5.0],
            "date_parsed": [
                pd.Timestamp("2023-01-01", tz="UTC"),
                pd.Timestamp("2023-01-05", tz="UTC"),
            ],
        }
    )

    result = assert_ofx_ready(df)

    assert result == pd.Timestamp("2023-01-05", tz="UTC")


def test_assert_ofx_ready_rejects_missing_amount_column():
    df = pd.DataFrame({"date_parsed": [pd.Timestamp("2023-01-01", tz="UTC")]})

    with pytest.raises(ValueError, match="amount_clean"):
        assert_ofx_ready(df)


def test_assert_ofx_ready_requires_non_null_amounts():
    df = pd.DataFrame(
        {
            "amount_clean": [pd.NA, pd.NA],
            "statement_end_date": [pd.Timestamp("2023-01-31", tz="UTC"), pd.NaT],
        }
    )

    with pytest.raises(ValueError, match="non-null 'amount_clean'"):
        assert_ofx_ready(df)


def test_assert_ofx_ready_accepts_fallback_timestamp():
    fallback = pd.Timestamp("2023-02-28", tz="UTC")
    df = pd.DataFrame(
        {
            "amount_clean": [100.0],
            "statement_end_date": [fallback],
        }
    )

    result = assert_ofx_ready(df)

    assert result == fallback


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


def test_build_ofx_defaults_missing_dtposted(df_without_dates):
    ofx_text = build_ofx(df_without_dates, accttype="checking", acctid="12345")
    dtposted_match = re.search(r"<DTPOSTED>([^<]+)</DTPOSTED>", ofx_text)

    assert dtposted_match is not None
    dtposted_value = dtposted_match.group(1)

    assert dtposted_value != "None"
    assert re.fullmatch(r"\d{14}\.\d{3}\[0:UTC\]", dtposted_value)


def test_build_ofx_uses_fallback_timestamp_for_ranges():
    fallback = pd.Timestamp("2023-03-31", tz="UTC")
    df = pd.DataFrame(
        {
            "amount_clean": [12.0],
            "trntype_norm": ["DEBIT"],
            "fitid_norm": ["ID123"],
            "cleaned_desc": ["Fallback transaction"],
            "statement_end_date": [fallback],
        }
    )

    ofx_text = build_ofx(df, accttype="checking", acctid="12345")
    fallback_str = ofx_datetime(fallback)

    assert f"<DTSTART>{fallback_str}</DTSTART>" in ofx_text
    assert f"<DTEND>{fallback_str}</DTEND>" in ofx_text
