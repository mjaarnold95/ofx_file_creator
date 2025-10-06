"""Utilities for transforming cleaned transaction data into OFX documents.

When transaction rows are missing a ``date_parsed`` value, the OFX specification
still requires a posting timestamp.  To keep the generated files valid, the
``build_ofx`` helper falls back to the statement end date (or the current UTC
time when no statement range is available) so that each transaction includes a
fully formatted ``<DTPOSTED>`` element instead of the string ``"None"``.
"""

import textwrap
from uuid import uuid4

import pandas as pd
from typing import Optional

from utils.cleaning import infer_trntype, infer_trntype_series
from utils.date_time import ofx_datetime
from utils.id import make_fitid
from utils.validate import assert_ofx_ready


def _normalize_timestamp(value: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = value
    if not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.to_datetime(ts, errors="coerce", utc=True)
        except Exception:
            return None
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0] if not ts.empty else pd.NaT
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


# ---------- OFX ----------
def build_ofx(
    df_txn: pd.DataFrame,
    accttype: str,
    acctid: str,
    bankid: str = "211075086",
    org: str = "LendingClub",
    fid: str = "68354",
    currency: str = "USD",
    starting_balance: float = 0.0,
    *,
    statement_begin: Optional[pd.Timestamp] = None,
    statement_end: Optional[pd.Timestamp] = None,
) -> str:
    validation_timestamp = assert_ofx_ready(df_txn)

    statement_begin = _normalize_timestamp(statement_begin)
    statement_end = _normalize_timestamp(statement_end)

    fallback_timestamp = statement_end or statement_begin or validation_timestamp

    now = pd.Timestamp.utcnow()
    dtserver = ofx_datetime(now)

    # Prefer acct info from the file if present
    if "acctnum" in df_txn.columns and df_txn["acctnum"].notna().any():
        acctid = str(df_txn["acctnum"].dropna().iloc[0])[:32]

    accttype = accttype.upper()
    if accttype not in {
        "CHECKING",
        "SAVINGS",
        "CREDITLINE",
        "CREDITCARD",
        "MONEYMRKT",
        "LOAN",
        "_401K",
        "BROKERAGE",
        "_403B",
        "HSA",
        "_457B",
        "ANNUITY",
        "CD",
    }:
        accttype = "CHECKING"

    if "date_parsed" in df_txn.columns:
        normalized_dates = pd.to_datetime(df_txn["date_parsed"], errors="coerce", utc=True)
    else:
        normalized_dates = pd.Series(pd.NaT, index=df_txn.index, dtype="datetime64[ns, UTC]")

    if normalized_dates.notna().any():
        df_txn["date_parsed"] = normalized_dates
    else:
        for candidate in ("posting_date", "posted_date", "transaction_date", "date"):
            if candidate in df_txn.columns:
                candidate_series = pd.to_datetime(df_txn[candidate], errors="coerce", utc=True)
                if candidate_series.notna().any():
                    df_txn["date_parsed"] = candidate_series
                    normalized_dates = candidate_series
                    break
        else:
            df_txn["date_parsed"] = normalized_dates

    has_dates = normalized_dates.notna().any()
    if statement_begin is not None:
        dtstart_ts = statement_begin
    elif has_dates:
        dtstart_ts = normalized_dates.min()
    else:
        dtstart_ts = fallback_timestamp

    if statement_end is not None:
        dtend_ts = statement_end
    elif has_dates:
        dtend_ts = normalized_dates.max()
    else:
        dtend_ts = fallback_timestamp

    dtstart = ofx_datetime(dtstart_ts)
    dtend = ofx_datetime(dtend_ts)

    if "amount_clean" in df_txn.columns:
        df_txn = df_txn[df_txn["amount_clean"].notna()].copy()
    else:
        df_txn = df_txn.copy()
        df_txn["amount_clean"] = 0.0

    amounts = df_txn["amount_clean"].astype(float)

    trntype_series = (
        df_txn["trntype_norm"].astype("string")
        if "trntype_norm" in df_txn.columns
        else pd.Series(pd.NA, index=df_txn.index, dtype="string")
    )
    missing_trntype = trntype_series.isna()
    if missing_trntype.any():
        trntype_source = (
            df_txn["trntype"]
            if "trntype" in df_txn.columns
            else pd.Series(pd.NA, index=df_txn.index)
        )
        cleaned_desc = (
            df_txn["cleaned_desc"]
            if "cleaned_desc" in df_txn.columns
            else pd.Series(pd.NA, index=df_txn.index)
        )
        inferred = infer_trntype_series(amounts, trntype_source, cleaned_desc)
        trntype_series = trntype_series.fillna(inferred)

    fitid_series = (
        df_txn["fitid_norm"].astype("string")
        if "fitid_norm" in df_txn.columns
        else pd.Series(pd.NA, index=df_txn.index, dtype="string")
    )
    missing_fitid = fitid_series.isna()
    if missing_fitid.any():
        generated = [
            make_fitid(df_txn.loc[idx], pos)
            for pos, idx in enumerate(df_txn.index[missing_fitid])
        ]
        # Assign a Series (not a plain list) when using boolean .loc selection
        gen_series = pd.Series(
            generated, index=df_txn.index[missing_fitid], dtype="string"
        )
        fitid_series.loc[missing_fitid] = gen_series

    # Ensure FITIDs are unique (OFX requires unique FITIDs per account)
    if fitid_series.duplicated().any():
        fitid_series = fitid_series.astype(str) + fitid_series.groupby(
            fitid_series
        ).cumcount().astype(str)

    def _coalesce_columns(columns: tuple[str, ...]) -> pd.Series:
        """Return the first non-null value across the provided columns."""

        selected = [
            df_txn[col].astype("string") for col in columns if col in df_txn.columns
        ]
        if not selected:
            return pd.Series("", index=df_txn.index, dtype="string")

        stacked = pd.concat(selected, axis=1)
        return stacked.bfill(axis=1).iloc[:, 0].fillna("")

    raw_name = _coalesce_columns(
        (
            "payee_llm",
            "payee_display",
            "posting_memo",
            "cleaned_desc",
            "raw_desc",
        )
    )

    raw_memo = _coalesce_columns(
        (
            "description_llm",
            "cleaned_desc",
            "posting_memo",
            "raw_desc",
            "payee_llm",
            "payee_display",
        )
    )

    def _escape_series(series: pd.Series) -> pd.Series:
        """Normalise OFX string fields while preserving XML entities."""

        normalised = (
            series.astype("string")
            .fillna("")
            .str.replace(r"\r?\n", " ", regex=True)
            .str.strip()
        )

        return (
            normalised.str.replace("&", "&amp;")
            .str.replace("<", "&lt;")
            .str.replace(">", "&gt;")
        )

    name_series = _escape_series(raw_name).str.slice(0, 32)
    memo_series = _escape_series(raw_memo)

    fallback_dtposted = ofx_datetime(fallback_timestamp) or dtend or ofx_datetime(now)
    if "date_parsed" in df_txn.columns:
        date_series = pd.to_datetime(df_txn["date_parsed"], errors="coerce", utc=True)
    else:
        date_series = pd.Series(pd.NaT, index=df_txn.index, dtype="datetime64[ns, UTC]")

    df_txn["date_parsed"] = date_series
    if date_series.notna().any():
        dtposted_series = date_series.map(ofx_datetime).fillna(fallback_dtposted)
    else:
        dtposted_series = pd.Series(fallback_dtposted, index=df_txn.index)

    checknum_series = (
        df_txn["checknum"]
        if "checknum" in df_txn.columns
        else pd.Series(index=df_txn.index)
    )

    render_df = pd.DataFrame(
        {
            "trnamt": amounts,
            "trntype": trntype_series.fillna("OTHER"),
            "fitid": fitid_series.fillna("").astype(str),
            "checknum": checknum_series,
            "dtposted": dtposted_series,
            "name": name_series.fillna(""),
            "memo": memo_series.fillna(""),
        },
        index=df_txn.index,
    )

    stmt_lines = []
    for idx, row in enumerate(render_df.itertuples(index=False, name="Txn")):
        trntype = infer_trntype(row.trnamt, None) if not row.trntype else row.trntype
        trntype = str(trntype)
        fitid = row.fitid or make_fitid(df_txn.loc[render_df.index[idx]], idx)
        # Normalize checknum/name/memo to str, decoding bytes if necessary
        raw_checknum = row.checknum
        checknum: Optional[str] = (
            (
                raw_checknum.decode("utf-8", "replace")
                if isinstance(raw_checknum, (bytes, bytearray))
                else str(raw_checknum)
            )
            if raw_checknum is not None and not pd.isna(raw_checknum)
            else None
        )

        # Ensure trnamt is a float-compatible value; decode bytes defensively
        trnamt_raw = row.trnamt
        if isinstance(trnamt_raw, (bytes, bytearray)):
            trnamt_val: object = trnamt_raw.decode("utf-8", "replace")
        else:
            trnamt_val = trnamt_raw
        trnamt_f: float = float(str(trnamt_val))

        # Prepare safe string fields that may be bytes
        def _safe_str(x: object) -> str:
            if isinstance(x, (bytes, bytearray)):
                return x.decode("utf-8", "replace")
            return str(x) if x is not None else ""

        xml = [
            "<STMTTRN>",
            f"  <TRNTYPE>{trntype}</TRNTYPE>",
            f"  <DTPOSTED>{str(_safe_str(row.dtposted))}</DTPOSTED>",
            f"  <TRNAMT>{float(str(trnamt_f)):.2f}</TRNAMT>",
            f"  <FITID>{_safe_str(fitid)}</FITID>",
        ]
        if checknum:
            xml.append(f"  <CHECKNUM>{checknum}</CHECKNUM>")
        if row.name:
            xml.append(f"  <NAME>{_safe_str(row.name)}</NAME>")
        if row.memo:
            memo_text = f"{_safe_str(row.memo)}"
            xml.append(f"  <MEMO>{memo_text}</MEMO>")
        xml.append("</STMTTRN>")
        stmt_lines.append("\n".join(xml))

    trns_str = "\n".join(stmt_lines)

    running = starting_balance + float(amounts.sum())

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<?OFX OFXHEADER="200" VERSION="220" SECURITY="NONE" OLDFILEUID="NONE" NEWFILEUID="NONE"?>
<OFX>
  <SIGNONMSGSRSV1>
    <SONRS>
      <STATUS>
        <CODE>0</CODE>
        <SEVERITY>INFO</SEVERITY>
      </STATUS>
      <DTSERVER>{dtserver}</DTSERVER>
      <LANGUAGE>ENG</LANGUAGE>
      <FI>
        <ORG>{org}</ORG>
        <FID>{fid}</FID>
      </FI>
    </SONRS>
  </SIGNONMSGSRSV1>
  <BANKMSGSRSV1>
    <STMTTRNRS>
      <TRNUID>{uuid4()}</TRNUID>
      <STATUS>
        <CODE>0</CODE>
        <SEVERITY>INFO</SEVERITY>
      </STATUS>
      <STMTRS>
        <CURDEF>{currency}</CURDEF>
        <BANKACCTFROM>
          <BANKID>{bankid}</BANKID>
          <ACCTID>{acctid}</ACCTID>
          <ACCTTYPE>{accttype}</ACCTTYPE>
        </BANKACCTFROM>
        <BANKTRANLIST>
          <DTSTART>{dtstart}</DTSTART>
          <DTEND>{dtend}</DTEND>
{textwrap.indent(trns_str, '          ')}
        </BANKTRANLIST>
        <LEDGERBAL>
          <BALAMT>{running:.2f}</BALAMT>
          <DTASOF>{dtend}</DTASOF>
        </LEDGERBAL>
      </STMTRS>
    </STMTTRNRS>
  </BANKMSGSRSV1>
</OFX>
"""
