from pathlib import Path

from utils.etl import load_and_prepare
from utils.build_ofx import build_ofx


def test_smoke_load_and_build_ofx(tmp_path: Path):
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    csv_path = examples_dir / "transactions.sample.csv"

    # exercise the main ETL path on the tiny sample CSV
    df = load_and_prepare(csv_path)

    # basic invariants
    assert "amount_clean" in df.columns
    assert not df.empty

    # build an OFX and check it contains the account id and at least one transaction tag
    ofx = build_ofx(df, accttype="checking", acctid="12345")
    assert "12345" in ofx
    assert "<STMTTRN>" in ofx
