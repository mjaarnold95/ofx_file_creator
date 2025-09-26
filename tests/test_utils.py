import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.build_ofx import build_ofx
from utils.cleaning import infer_trntype_series
from utils.date_time import ofx_datetime, parse_time_to_timedelta
from utils.id import make_fitid

from utils.rules import load_rules
from utils.etl import load_and_prepare


@pytest.fixture
def df_without_dates():
    return pd.DataFrame(
        {
            "amount_clean": [25.0],
            "cleaned_desc": ["No Date Transaction"],
            "trntype_norm": ["CREDIT"],
        }
    )


@pytest.fixture
def sample_transaction_csv(tmp_path):
    df = pd.DataFrame(
        {
            "Date": ["2023-01-02"],
            "Amount": ["123.45"],
            "Description": ["Sample transaction"],
        }
    )
    csv_path = tmp_path / "transactions.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


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


def test_infer_trntype_series_uses_custom_rules(tmp_path):
    config_path = tmp_path / "rules.json"
    config_path.write_text(
        json.dumps(
            {
                "source_aliases": {"XYZ": "PAYMENT"},
                "keyword_rules": {
                    "extend": [
                        {
                            "pattern": r"\bESPRESSO\b",
                            "trntype": "CASH",
                        }
                    ]
                },
            }
        )
    )

    custom_rules = load_rules(config_path)

    amounts = pd.Series([10.0, -12.5])
    trntype_text = pd.Series(["xyz", None])
    cleaned_desc = pd.Series([None, "Morning espresso run"])

    result = infer_trntype_series(
        amounts,
        trntype_text,
        cleaned_desc,
        rules=custom_rules,
    )

    assert list(result) == ["PAYMENT", "CASH"]
    
def test_load_and_prepare_handles_csv(sample_transaction_csv):
    df = load_and_prepare(sample_transaction_csv)

    assert "amount_clean" in df.columns
    assert df.loc[0, "amount_clean"] == pytest.approx(123.45)

    assert "date_parsed" in df.columns
    assert pd.Timestamp("2023-01-02", tz="UTC") == df.loc[0, "date_parsed"]
    
