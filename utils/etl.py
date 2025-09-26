from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils.cleaning import (
    clean_amount_series,
    clean_description,
    infer_trntype_series,
    )
from utils.date_time import parse_time_to_timedelta
from utils.io import load_transactions
from utils.llm_enrichment import enrich_transactions_with_llm
from utils.sheet import normalize_columns, detect_columns

# ---------- ETL ----------
# noinspection PyTypeChecker
def load_and_prepare(
    path: Path,
    *,
    llm_client: Optional[object] = None,
    llm_batch_size: int = 20,
) -> pd.DataFrame:
    df = load_transactions(path)
    df = normalize_columns(df)
    
    cols = detect_columns(df)
    ren: Dict[str, str] = {}
    for k in (
            "acctnum",
            "acctname",
            "date",
            "time",
            "amount",
            "description",
            "trntype",
            "fitid",
            "checknum",
            "memo",
            "name",
            ):
        if cols[k]:
            ren[cols[k]] = k
    df = df.rename(columns=ren)
    
    # ---------- Build robust text fields ----------
    # 1) raw_desc: coalesce many description-like fields, excluding 'acctname'
    desc_candidates = [
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
        "name",  # allow generic 'name' but only if it varies per row
        ]
    existing = [c for c in desc_candidates if c in df.columns and c != "acctname"]
    varying = [c for c in existing if df[c].nunique(dropna=False) > 1]
    df["raw_desc"] = df[varying].bfill(axis=1).iloc[:, 0] if varying else ""
    
    # 2) payee_display: prefer explicit payee/merchant/name that varies; fallback to raw_desc
    payee_candidates = ["payee", "payee_name", "merchant", "merchant_name", "name"]
    payee_existing = [c for c in payee_candidates if c in df.columns]
    payee_varying = [c for c in payee_existing if df[c].nunique(dropna=False) > 1]
    df["payee_display"] = (
        df[payee_varying].bfill(axis=1).iloc[:, 0] if payee_varying else df["raw_desc"]
    )
    
    # Date (UTC, normalized)
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        # search other columns containing "date"
        alt_date_cols = [c for c in df.columns if "date" in c and c != "date"]
        df["date_parsed"] = (
            pd.to_datetime(
                df[alt_date_cols].bfill(axis=1).iloc[:, 0], errors="coerce", utc=True
                )
            if alt_date_cols
            else pd.NaT
        )
    
    # TIME -> Timedelta (vectorized where possible, fallback per-cell)
    if "time" in df.columns:
        t = df["time"]
        
        # numeric fractions (0..1) -> seconds
        t_num = pd.to_numeric(t, errors="coerce")
        frac_mask = t_num.between(0, 1)
        td = pd.Series(pd.NaT, index=t.index, dtype="timedelta64[ns]")
        if frac_mask.any():
            secs = (t_num[frac_mask] * 86400).round().astype("Int64")
            td.loc[frac_mask] = pd.to_timedelta(secs, unit="s")
        
        # fast string path HH:MM[:SS]
        str_mask = t.astype(str).str.contains(":", regex=False)
        if str_mask.any():
            parsed = pd.to_datetime(t[str_mask], format="%H:%M:%S", errors="coerce")
            # try HH:MM for those that failed
            need = parsed.isna()
            if need.any():
                parsed2 = pd.to_datetime(
                    t[str_mask][need], format="%H:%M", errors="coerce"
                    )
                parsed = parsed.fillna(parsed2)
            
            good = parsed.notna()
            if good.any():
                td.loc[parsed.index[good]] = (
                        pd.to_timedelta(parsed[good].dt.hour, unit="h")
                        + pd.to_timedelta(parsed[good].dt.minute, unit="m")
                        + pd.to_timedelta(parsed[good].dt.second, unit="s")
                )
        
        # fallback: robust per-cell parser
        remaining = td.isna()
        if remaining.any():
            td.loc[remaining] = t[remaining].map(parse_time_to_timedelta)
        
        df["time_delta"] = td
    else:
        df["time_delta"] = pd.NaT
    
    # Combine date + time (vectorized)
    dp = df["date_parsed"]
    # normalize to UTC midnight
    dp_utc = pd.Series(pd.NaT, index=dp.index, dtype="datetime64[ns, UTC]")
    notna = dp.notna()
    if notna.any():
        dp_local = dp[notna].copy()
        dp_local = (
            dp_local.dt.tz_convert("UTC")
            if dp_local.dt.tz is not None
            else dp_local.dt.tz_localize("UTC")
        )
        dp_utc.loc[notna] = dp_local.dt.normalize()
    
    td = df["time_delta"]
    has_time = td.notna()
    # start with base dates
    combined = dp_utc.copy()
    # add time when both present
    both = notna & has_time
    if both.any():
        combined.loc[both] = dp_utc[both] + td[both]
    # if only time present, use today's UTC date + time
    only_time = (~notna) & has_time
    if only_time.any():
        today = pd.Timestamp.utcnow().normalize().tz_localize("UTC")
        combined.loc[only_time] = today + td[only_time]
    
    df["date_parsed"] = combined
    
    # Drop temp time column
    df.drop(columns=["time_delta"], inplace=True)
    
    # Amounts
    if "amount" in df.columns:
        df["amount_clean"] = clean_amount_series(df["amount"])
    else:
        poss = [
            c
            for c in df.columns
            if "amount" in c
               or "debit" in c
               or "credit" in c
               or c in ("withdrawal", "deposit")
            ]
        if (
                len(poss) >= 2
                and any("debit" in c for c in poss)
                and any("credit" in c for c in poss)
        ):
            debit_col = next(c for c in poss if "debit" in c)
            credit_col = next(c for c in poss if "credit" in c)
            debit = clean_amount_series(df[debit_col]).fillna(0)
            credit = clean_amount_series(df[credit_col]).fillna(0)
            df["amount_clean"] = credit - debit
        elif poss:
            df["amount_clean"] = clean_amount_series(df[poss[0]])
        else:
            df["amount_clean"] = np.nan
    
    # Cleaned description (from coalesced raw_desc)
    df["cleaned_desc"] = df["raw_desc"].apply(clean_description)

    if llm_client is not None:
        enriched = enrich_transactions_with_llm(
            df, llm_client, batch_size=llm_batch_size
        )
        df = df.join(enriched)
    
    # TRNTYPE normalization/inference
    trntype_source = (
        df["trntype"] if "trntype" in df.columns else pd.Series(pd.NA, index=df.index)
    )
    df["trntype_norm"] = infer_trntype_series(
        df["amount_clean"], trntype_source, df["cleaned_desc"]
        )
    
    # FITID cleanup
    if "fitid" in df.columns:
        df["fitid_norm"] = (
            df["fitid"].astype(str).str.strip().replace(r"(?i)^nan$", pd.NA, regex=True)
        )
    else:
        df["fitid_norm"] = None
    
    # Keep non-null amount rows; sort by date if present
    df = df[df["amount_clean"].notna()].copy()
    if df["date_parsed"].notna().any():
        df.sort_values("date_parsed", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
