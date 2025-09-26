import hashlib
import json
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.build_ofx import build_ofx
from utils.cleaning import infer_trntype_series
from utils.date_time import ofx_datetime, parse_time_to_timedelta
from utils.id import make_fitid
from utils.validate import assert_ofx_ready

from utils.rules import load_rules
from utils.etl import load_and_prepare
from utils.llm_enrichment import enrich_transactions_with_llm, _resolve_mlx_client_config


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


def test_build_ofx_prefers_llm_enriched_fields():
    df = pd.DataFrame(
        {
            "date_parsed": [pd.Timestamp("2024-05-01", tz="UTC")],
            "amount_clean": [-4.75],
            "trntype_norm": ["DEBIT"],
            "cleaned_desc": ["Coffee Shop"],
            "payee_display": ["Coffee Shop"],
            "payee_llm": ["Daily Grind Coffee"],
            "description_llm": ["Coffee purchase"],
        }
    )

    ofx_text = build_ofx(df, accttype="checking", acctid="12345")

    assert "<NAME>Daily Grind Coffee</NAME>" in ofx_text
    assert "<MEMO>Coffee purchase (_DEBIT_)</MEMO>" in ofx_text


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


def test_build_ofx_requires_amount_column():
    df = pd.DataFrame(
        {
            "date_parsed": [pd.Timestamp("2024-01-01", tz="UTC")],
        }
    )

    with pytest.raises(ValueError, match="amount_clean"):
        build_ofx(df, accttype="checking", acctid="12345")


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


class FakeLLMClient:
    def __init__(self):
        self.calls = []

    def generate_batch(self, prompts):
        self.calls.append(list(prompts))
        responses = []
        for prompt in prompts:
            if "Coffee" in prompt:
                responses.append(
                    json.dumps(
                        {
                            "payee": "Daily Grind Coffee",
                            "category": "Food & Drink",
                            "description": "Coffee purchase",
                        }
                    )
                )
            elif "Hardware" in prompt:
                responses.append(
                    "Here you go: {\"payee\": \"Ace Hardware\","
                    " \"category\": \"Home Improvement\","
                    " \"description\": \"Hardware run\"}"
                )
            else:
                responses.append("No idea")
        return responses


def test_resolve_mlx_client_config_defaults_to_client():
    cfg = {"llm_model": "client", "model": object(), "tokenizer": object()}
    assert _resolve_mlx_client_config(cfg) is None

    cfg_no_hint = {"model": object(), "tokenizer": object()}
    assert _resolve_mlx_client_config(cfg_no_hint) is None


def test_enrich_transactions_with_llm_batches_and_failures():
    df = pd.DataFrame(
        {
            "raw_desc": ["Coffee Shop", "Weekend Hardware", "Unknown"],
            "payee_display": ["Coffee Shop", "Hardware Store", "Mystery"],
            "cleaned_desc": ["coffee", "hardware", ""],
        }
    )

    client = FakeLLMClient()
    enriched = enrich_transactions_with_llm(df, client, batch_size=2)

    assert list(enriched.columns) == [
        "payee_llm",
        "category_llm",
        "description_llm",
    ]
    assert enriched.loc[0, "payee_llm"] == "Daily Grind Coffee"
    assert enriched.loc[1, "category_llm"] == "Home Improvement"
    assert pd.isna(enriched.loc[2, "payee_llm"])
    assert len(client.calls) == 2


def test_enrich_transactions_with_llm_supports_mlx(monkeypatch):
    df = pd.DataFrame(
        {
            "raw_desc": ["Coffee Shop", "Weekend Hardware", "Unknown"],
            "payee_display": ["Coffee Shop", "Hardware Store", "Mystery"],
            "cleaned_desc": ["coffee", "hardware", ""],
        }
    )

    fake_module = types.ModuleType("mlx_lm")
    fake_module.calls = []

    class FakeTokenizer:
        eos_token_ids = [0]

    fake_tokenizer = FakeTokenizer()

    def response_for_prompt(text):
        if "Coffee" in text:
            return json.dumps(
                {
                    "payee": "Daily Grind Coffee",
                    "category": "Food & Drink",
                    "description": "Coffee purchase",
                }
            )
        if "Hardware" in text:
            return json.dumps(
                {
                    "payee": "Ace Hardware",
                    "category": "Home Improvement",
                    "description": "Hardware run",
                }
            )
        return "No idea"

    def fake_batch_generate(model, tokenizer, prompts, **kwargs):
        fake_module.calls.append(("batch", prompts, kwargs))
        texts = [response_for_prompt(p) for p in prompts]
        return types.SimpleNamespace(texts=texts)

    def fake_generate(model, tokenizer, prompt, **kwargs):
        fake_module.calls.append(("single", prompt, kwargs))
        return response_for_prompt(prompt)

    fake_module.batch_generate = fake_batch_generate
    fake_module.generate = fake_generate

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    client = {
        "model": object(),
        "tokenizer": fake_tokenizer,
        "llm_model": "mlx_lm",
        "generation_kwargs": {"max_tokens": 64},
    }

    enriched_batch = enrich_transactions_with_llm(df, client, batch_size=3)

    assert enriched_batch.loc[0, "payee_llm"] == "Daily Grind Coffee"
    assert enriched_batch.loc[1, "category_llm"] == "Home Improvement"
    assert pd.isna(enriched_batch.loc[2, "payee_llm"])
    assert fake_module.calls and fake_module.calls[0][0] == "batch"

    fake_module.calls = []
    client_single = {**client, "use_batch": False}
    enriched_single = enrich_transactions_with_llm(df, client_single, batch_size=2)

    pd.testing.assert_frame_equal(enriched_single, enriched_batch)
    assert len(fake_module.calls) == len(df)
    assert all(call[0] == "single" for call in fake_module.calls)

def test_load_and_prepare_enriches_when_llm_client_provided(tmp_path):
    csv_path = tmp_path / "transactions.csv"
    pd.DataFrame(
        {
            "Date": ["2023-05-01", "2023-05-02"],
            "Amount": ["10.00", "25.00"],
            "Description": ["Coffee Shop", "Weekend Hardware"],
            "Payee": ["Coffee Shop", "Hardware Store"],
        }
    ).to_csv(csv_path, index=False)

    client = FakeLLMClient()
    df = load_and_prepare(csv_path, llm_client=client, llm_batch_size=1)

    assert "payee_llm" in df.columns
    assert df.loc[0, "payee_llm"] == "Daily Grind Coffee"
    assert df.loc[0, "category_llm"] == "Food & Drink"
    assert df.loc[1, "payee_llm"] == "Ace Hardware"

