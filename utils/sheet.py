import re
from typing import Dict, Optional

import pandas as pd

_PREFERRED_SHEETS: tuple[str, ...] = (
    'transactions',
    'sheet1',
    'sheet',
    'chk',
    'checking',
    'sav',
    'hys',
    'savings',
)

_COLUMN_CANDIDATES: Dict[str, tuple[str, ...]] = {
    'acctnum': (
        'account_number',
        'acct_num',
        'account_no',
        'acct_no',
        'acct',
    ),
    'acctname': (
        'account_name',
        'acct_nickname',
        'account_nick',
        'acct_nick',
        'acct_nname',
    ),
    'date': (
        'transaction_date',
        'posting_date',
        'statement_begin_date',
        'statement_end_date',
        'date',
        'posted_date',
        'post_date',
        'dtposted',
    ),
    'time': ('transaction_time', 'time', 'posting_time', 'posted_time', 'post_time'),
    'amount': (
        'amount_signed',
        'signed_amount',
        'amount',
        'amt',
        'transaction_amount',
        'credit_debit',
        'value',
    ),
    'balance': (
        'balance',
        'bal',
        'running_balance',
        'available_balance',
        'ending_balance',
    ),
    'description': (
        'description',
        'transaction_description',
        'trans_description',
        'desc',
        'memo',
        'narrative',
        'detail',
        'details',
        'posting_memo',
        'payment_details',
    ),
    'trntype': (
        'type',
        'trntype',
        'transaction_type',
        'debit_credit',
    ),
    'fitid': (
        'fitid',
        'id',
        'transaction_id',
        'reference',
        'ref',
        'unique_id',
    ),
    'checknum': ('checknum', 'check_no', 'check_number', 'chknum'),
    'name': ('payee', 'payee_name', 'merchant', 'merchant_name'),
    'memo': ('memo', 'notes', 'extra', 'details', 'posting_memo', 'payment_details'),
}


def find_best_sheet(xl: pd.ExcelFile) -> str:
    sheet_names = [str(name) for name in xl.sheet_names]
    lowered = [name.lower() for name in sheet_names]

    for needle in _PREFERRED_SHEETS:
        for original, lower in zip(sheet_names, lowered):
            if needle in lower:
                return original
    return sheet_names[0]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = [
        re.sub(r"[^a-z0-9]+", '_', str(col).strip().lower()).strip('_')
        for col in df.columns
    ]

    copy = df.copy()
    copy.columns = normalized
    return copy


def _pick_column(
    original: tuple[str, ...],
    lowered: tuple[str, ...],
    candidates: tuple[str, ...],
    used: set[str],
) -> Optional[str]:
    candidate_lower = tuple(item.lower() for item in candidates)

    for idx, column in enumerate(lowered):
        if column in candidate_lower and original[idx] not in used:
            return original[idx]

    for idx, column in enumerate(lowered):
        if original[idx] in used:
            continue
        if any(token in column for token in candidate_lower):
            return original[idx]
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    original = tuple(str(col) for col in df.columns)
    lowered = tuple(col.lower() for col in original)

    used: set[str] = set()
    result: Dict[str, Optional[str]] = {}
    for key, candidates in _COLUMN_CANDIDATES.items():
        chosen = _pick_column(original, lowered, candidates, used)
        result[key] = chosen
        if chosen is not None:
            used.add(chosen)

    if result['balance'] is None:
        result['balance'] = result['amount']

    if result['trntype'] is None:
        result['trntype'] = result['description']

    if result['name'] is None:
        result['name'] = result['description']

    if result['memo'] is None:
        result['memo'] = result['description']

    return result
