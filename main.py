"""
main.py

Process Excel files containing bank transaction data, normalize and clean the data,
infer transaction types, and generate OFX files for financial software import.

Usage:
    pip3 install -r requirements.txt  # installs pandas, numpy, openpyxl, python-dateutil
    python main.py

Edit the paths in the __main__ section to point to your checking and savings account Excel files.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

try:  # optional dependency; fallback to non-LLM mode when unavailable
    from mlx_lm.utils import load as mlx_load
except ImportError:  # pragma: no cover - dependency not installed
    mlx_load = None  # type: ignore[assignment]

from utils.build_ofx import build_ofx
from utils.detect_account_type import detect_account_type
from utils.etl import load_and_prepare
from utils.id import derive_acctid_from_path


def _normalise_timestamp(value: object) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        try:
            ts = pd.to_datetime(value, errors='coerce', utc=True)
        except Exception:
            return None
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0] if not ts.empty else pd.NaT
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts


def _statement_suffix(
    begin: Optional[pd.Timestamp], end: Optional[pd.Timestamp], fallback_index: int
) -> str:
    def fmt(val: Optional[pd.Timestamp]) -> Optional[str]:
        if val is None:
            return None
        return val.strftime('%Y%m%d')

    parts = []
    begin_str = fmt(begin)
    end_str = fmt(end)
    if begin_str:
        parts.append(begin_str)
    if end_str and end_str != begin_str:
        parts.append(end_str)
    if parts:
        return '-'.join(parts)
    return f'statement{fallback_index}'


def _process_account(
    path: Path, stub: str, label: str, llm_client: Optional[object]
) -> None:
    if not path.exists():
        print(f"Skipping {label.lower()} account; missing source file at {path}")
        return

    try:
        df = load_and_prepare(path, llm_client=llm_client, llm_batch_size=50)

        begin_col = next(
            (col for col in ('statement_begin_date', 'statement_begin') if col in df.columns),
            None,
        )
        end_col = next(
            (col for col in ('statement_end_date', 'statement_end') if col in df.columns),
            None,
        )

        for col in (begin_col, end_col):
            if col is not None:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

        group_cols = [col for col in (begin_col, end_col) if col is not None]
        if group_cols and df[group_cols].notna().any(axis=None):
            grouped = list(df.groupby(group_cols, dropna=False, sort=False))
        else:
            grouped = [((None,) * len(group_cols), df)]

        accttype = detect_account_type(path)
        acctid = derive_acctid_from_path(path, stub)

        emitted_paths: set[Path] = set()

        for idx, (_, group_df) in enumerate(grouped, start=1):
            subset = group_df.copy()

            begin_val = None
            if begin_col:
                begin_values = subset[begin_col].dropna()
                if not begin_values.empty:
                    begin_val = _normalise_timestamp(begin_values.min())

            end_val = None
            if end_col:
                end_values = subset[end_col].dropna()
                if not end_values.empty:
                    end_val = _normalise_timestamp(end_values.max())

            ofx_text = build_ofx(
                subset,
                accttype=accttype,
                acctid=acctid,
                statement_begin=begin_val,
                statement_end=end_val,
            )

            if len(grouped) == 1:
                out_path = path.with_suffix('.ofx')
                suffix_label = None
            else:
                suffix = _statement_suffix(begin_val, end_val, idx)
                out_path = path.with_name(f"{path.stem}_{suffix}.ofx")
                suffix_label = suffix

                if out_path in emitted_paths:
                    out_path = path.with_name(f"{path.stem}_{suffix}_{idx}.ofx")

            emitted_paths.add(out_path)

            out_path.write_text(ofx_text, encoding='utf-8')
            if suffix_label:
                print(f"{label} OFX ({suffix_label}) written to {out_path}")
            else:
                print(f"{label} OFX written to {out_path}")
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Error processing {label.lower()} account: {exc}")


# noinspection PyShadowingNames
def main(
    checking_path: Path,
    savings_path: Path,
    llm_client: Optional[object] = None,
) -> None:
    _process_account(checking_path, 'CHK', 'Checking', llm_client)
    _process_account(savings_path, 'SAV', 'Savings', llm_client)


if __name__ == '__main__':
    base_dir = (
        Path.home()
        / 'Library'
        / 'CloudStorage'
        / 'OneDrive-Personal'
        / 'Documents'
    )

    checking_path: Path = base_dir / 'LendingClub2_checking.xlsx'
    savings_path: Path = base_dir / 'LendingClub2_savings.xlsx'

    use_llm = os.environ.get('OFX_USE_LLM', '1') not in {'0', '', 'false', 'False'}

    if use_llm:
        if mlx_load is None:
            print('mlx_lm is not installed; continuing without LLM enrichment')
            client = None
        else:
            model, tokenizer = mlx_load('mlx-community/Llama-3.2-3B-Instruct-4bit')
            client = {
                'llm_model': 'mlx_lm',
                'model': model,
                'tokenizer': tokenizer,
                'generation_kwargs': {'max_tokens': 1024},
                'use_batch': True,
            }
    else:
        client = None
        if mlx_load is None:
            print('LLM enrichment disabled; install mlx_lm and set OFX_USE_LLM=1 to enable')

    main(checking_path=checking_path, savings_path=savings_path, llm_client=client)
