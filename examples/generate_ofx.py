from pathlib import Path
import sys

# Ensure the repository root (with the 'utils' package) is on PYTHONPATH when run from examples/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.etl import load_and_prepare
from utils.build_ofx import build_ofx


def main():
    examples_dir = Path(__file__).resolve().parent
    csv_path = examples_dir / "transactions.sample.csv"
    out_path = examples_dir / "transactions.sample.ofx"

    df = load_and_prepare(csv_path)
    ofx_text = build_ofx(df, accttype="checking", acctid="12345")
    out_path.write_text(ofx_text, encoding="utf-8")
    print(f"Wrote OFX to {out_path}")


if __name__ == "__main__":
    main()
