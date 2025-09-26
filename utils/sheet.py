import re
from typing import Optional, Dict

import pandas as pd

# ---------- sheet & columns ----------
def find_best_sheet(xl: pd.ExcelFile) -> str:
    sheet_names = [str(s).lower() for s in xl.sheet_names]
    preferred = [
        "transactions",
        "sheet1",
        "sheet",
        "chk",
        "checking",
        "sav",
        "hys",
        "savings",
        ]
    for p in preferred:
        for idx, s in enumerate(sheet_names):
            if p in s:
                return str(xl.sheet_names[idx])
    return str(xl.sheet_names[0])

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()) for c in df.columns
        ]
    return df

def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = df.columns.tolist()
    
    def pick(*cands):
        # exact match first
        for c in cands:
            if c in cols:
                return c
        # partial contains next
        for c in cols:
            for cand in cands:
                if cand in c:
                    return c
        return None
    
    # account metadata
    acct_num_cols = ["account_number", "acct_num", "account_no", "acct_no", "acct"]
    acct_nname_cols = [
        "account_name",
        "acct_nickname",
        "account_nick",
        "acct_nick",
        "acct_nname",
        ]
    
    # dates/times
    date_cols = [
        "transaction_date",
        "posting_date",
        "statement_begin_date",
        "statement_end_date",
        "date",
        "posted_date",
        "post_date",
        "dtposted",
        ]
    time_cols = ["transaction_time", "time", "posting_time"]
    
    # description-like fields (do NOT include generic 'account_name')
    desc_cols = [
        "description",
        "transaction_description",
        "trans_description",
        "desc",
        "memo",
        "narrative",
        "detail",
        "details",
        "payee",
        "payee_name",
        "merchant",
        "merchant_name",
        "posting_memo",
        "payment_details",
        ]
    
    # amounts
    amt_cols = [
        "amount",
        "amt",
        "transaction_amount",
        "credit_debit",
        "value",
        "withdrawal",
        "deposit",
        ]
    bal_cols = [
        "balance",
        "bal",
        "running_balance",
        "available_balance",
        "ending_balance",
        ]
    
    # types/ids
    type_cols = ["type", "trntype", "transaction_type", "debit_credit"]
    fitid_cols = ["fitid", "id", "transaction_id", "reference", "ref", "unique_id"]
    checknum_cols = ["checknum", "check_no", "check_number", "chknum"]
    
    # explicit payee-ish (do NOT include generic 'name' here to avoid acct name bleed)
    name_cols = ["payee", "payee_name", "merchant", "merchant_name"]
    memo_cols = ["memo", "notes", "extra", "details", "posting_memo", "payment_details"]
    
    return dict(
        acctnum=pick(*acct_num_cols),
        acctname=pick(*acct_nname_cols),
        date=pick(*date_cols),
        time=pick(*time_cols),
        amount=pick(*amt_cols),
        balance=pick(*bal_cols) or pick(*amt_cols),
        description=pick(*desc_cols),
        trntype=pick(*type_cols) or pick(*desc_cols),
        fitid=pick(*fitid_cols),
        checknum=pick(*checknum_cols),
        name=pick(*name_cols) or pick(*desc_cols),  # payee-ish first, then desc-like
        memo=pick(*memo_cols) or pick(*desc_cols),
        )