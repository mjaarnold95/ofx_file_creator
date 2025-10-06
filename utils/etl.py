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
        col_name = cols[k]
        if col_name:
            # cols maps canonical keys to detected column names (or None).
            # Narrow the Optional[str] to str for mypy before using as dict key.
            ren[col_name] = k
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

    # Date (UTC, preserving time when available)
    date_priority = (
        "posting_datetime",
        "posted_datetime",
        "transaction_datetime",
        "date_time",
        "datetime",
        "posted_date",
        "posting_date",
        "post_date",
        "transaction_date",
        "date",
    )
    date_candidates = [col for col in date_priority if col in df.columns]

    if date_candidates:
        parsed_frames = []
        for col in date_candidates:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            parsed_frames.append(parsed.rename(col))
        stacked = pd.concat(parsed_frames, axis=1)
        base_dates = stacked.bfill(axis=1).iloc[:, 0]
    else:
        base_dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    df["date_parsed"] = base_dates

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
    dp_utc = pd.Series(pd.NaT, index=dp.index, dtype="datetime64[ns, UTC]")
    notna = dp.notna()
    if notna.any():
        dp_local = dp[notna].copy()
        dp_local = (
            dp_local.dt.tz_convert("UTC")
            if dp_local.dt.tz is not None
            else dp_local.dt.tz_localize("UTC")
        )
        dp_utc.loc[notna] = dp_local
        dp_base = dp_local.dt.normalize()
    else:
        dp_base = pd.Series(pd.NaT, index=dp.index, dtype="datetime64[ns, UTC]")

    td = df["time_delta"]
    has_time = td.notna()
    combined = dp_utc.copy()
    both = notna & has_time
    if both.any():
        combined.loc[both] = dp_base[both] + td[both]
    only_time = (~notna) & has_time
    if only_time.any():
        today = pd.Timestamp.utcnow().normalize().tz_localize("UTC")
        combined.loc[only_time] = today + td[only_time]

    df["date_parsed"] = combined

    df.drop(columns=["time_delta"], inplace=True)

    # Amounts
    amount_series = None

    if "amount" in df.columns:
        amt = clean_amount_series(df["amount"])
        if amt.notna().any():
            amount_series = amt

    if amount_series is None:
        signed_col = next(
            (
                c
                for c in df.columns
                if ("signed" in c and "amount" in c) or "amount_signed" in c or "signed_amount" in c
            ),
            None,
        )
        if signed_col is not None:
            amt = clean_amount_series(df[signed_col])
            if amt.notna().any():
                amount_series = amt

    debit_cols = [
        c
        for c in df.columns
        if "debit" in c or "withdraw" in c or "withdrawal" in c
    ]
    credit_cols = [
        c
        for c in df.columns
        if "credit" in c or "deposit" in c
    ]

    if amount_series is None and (debit_cols or credit_cols):
        debit_col = debit_cols[0] if debit_cols else None
        credit_col = credit_cols[0] if credit_cols else None
        debit = (
            clean_amount_series(df[debit_col]).fillna(0)
            if debit_col is not None
            else pd.Series(0, index=df.index, dtype="float64")
        )
        credit = (
            clean_amount_series(df[credit_col]).fillna(0)
            if credit_col is not None
            else pd.Series(0, index=df.index, dtype="float64")
        )
        amount_series = credit - debit

    if amount_series is None:
        poss = [c for c in df.columns if "amount" in c]
        if poss:
            amount_series = clean_amount_series(df[poss[0]])

    if amount_series is None:
        amount_series = pd.Series(np.nan, index=df.index, dtype="float64")

    df["amount_clean"] = amount_series

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
