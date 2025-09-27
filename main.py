"""
main.py

Process Excel files containing bank transaction data, normalize and clean the data,
infer transaction types, and generate OFX files for financial software import.

Usage:
    pip3 install -r requirements.txt  # installs pandas, numpy, openpyxl, python-dateutil
    python main.py

Edit the paths in the __main__ section to point to your checking and savings account Excel files.
"""

from pathlib import Path

from utils.build_ofx import build_ofx
from utils.detect_account_type import detect_account_type
from utils.etl import load_and_prepare
from utils.id import derive_acctid_from_path
from mlx_lm.utils import load as mlx_load

# ---------- main ----------
# noinspection PyShadowingNames
def main(checking_path: Path, savings_path: Path, llm_client: object):
    # Process checking account
    if checking_path.exists():
        try:
            df = load_and_prepare(checking_path, llm_client=llm_client)
            accttype = detect_account_type(checking_path)
            acctid = derive_acctid_from_path(checking_path, "CHK")
            check_ofx_text = build_ofx(df, accttype=accttype, acctid=acctid)
            out_path = checking_path.with_suffix(".ofx")
            out_path.write_text(check_ofx_text, encoding="utf-8")
            print(f"Checking OFX written to {out_path}")
        except Exception as e:
            print(f"Error processing checking account: {e}")

    # Process savings account
    if savings_path.exists():
        try:
            df = load_and_prepare(savings_path, llm_client=llm_client)
            accttype = detect_account_type(savings_path)
            acctid = derive_acctid_from_path(savings_path, "SAV")
            sav_ofx_text = build_ofx(df, accttype=accttype, acctid=acctid)
            out_path = savings_path.with_suffix(".ofx")
            out_path.write_text(sav_ofx_text, encoding="utf-8")
            print(f"Savings OFX written to {out_path}")
        except Exception as e:
            print(f"Error processing savings account: {e}")


if __name__ == "__main__":
    checking_path: Path = Path(
        "/Users/marnold8/Desktop/Finances/LendingClub/LendingClub_checking.xlsx"
    )
    savings_path: Path = Path(
        "/Users/marnold8/Desktop/Finances/LendingClub/LendingClub_savings.xlsx"
    )

    model, tokenizer = mlx_load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    client = {
        "llm_model": "mlx_lm",
        "model": model,
        "tokenizer": tokenizer,
        "generation_kwargs": {"max_tokens": 128},
    }
    llm_client = client

    main(checking_path=checking_path, savings_path=savings_path, llm_client=llm_client)
