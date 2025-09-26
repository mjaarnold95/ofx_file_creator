import textwrap
from uuid import uuid4

import pandas as pd

from utils.cleaning import infer_trntype, infer_trntype_series
from utils.date_time import ofx_datetime
from utils.id import make_fitid

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
    ) -> str:
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
    
    if "date_parsed" not in df_txn.columns or not df_txn["date_parsed"].notna().any():
        dtstart_ts = dtend_ts = now
    else:
        dtstart_ts = df_txn["date_parsed"].min()
        dtend_ts = df_txn["date_parsed"].max()
    
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
        fitid_series.loc[missing_fitid] = generated
    
    name_candidates = []
    for col in ("payee_display", "posting_memo", "cleaned_desc", "raw_desc"):
        if col in df_txn.columns:
            name_candidates.append(df_txn[col].astype("string"))
    if name_candidates:
        stacked = pd.concat(name_candidates, axis=1)
        raw_name = stacked.bfill(axis=1).iloc[:, 0].fillna("")
    else:
        raw_name = pd.Series("", index=df_txn.index, dtype="string")
    
    def _escape_series(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.replace("&", "&amp;")
            .str.replace("<", "&lt;")
            .str.replace(">", "&gt;")
            .str.replace("\n", "")
            .str.strip()
        )
    
    escaped = _escape_series(raw_name)
    name_series = escaped.str.slice(0, 32)
    memo_series = escaped
    
    dtposted_series = (
        df_txn["date_parsed"].apply(ofx_datetime)
        if "date_parsed" in df_txn.columns
        else pd.Series("", index=df_txn.index)
    )
    
    checknum_series = (
        df_txn["checknum"] if "checknum" in df_txn.columns else pd.Series(index=df_txn.index)
    )
    
    render_df = pd.DataFrame(
        {
            "trnamt":amounts,
            "trntype":trntype_series.fillna("OTHER"),
            "fitid":fitid_series.fillna("").astype(str),
            "checknum":checknum_series,
            "dtposted":dtposted_series,
            "name":name_series.fillna(""),
            "memo":memo_series.fillna(""),
            },
        index=df_txn.index,
        )
    
    stmt_lines = []
    for idx, row in enumerate(render_df.itertuples(index=False, name="Txn")):
        trntype = infer_trntype(row.trnamt, None) if not row.trntype else row.trntype
        fitid = row.fitid or make_fitid(df_txn.loc[render_df.index[idx]], idx)
        checknum = (
            str(row.checknum)
            if row.checknum is not None and not pd.isna(row.checknum)
            else None
        )
        xml = [
            "<STMTTRN>",
            f"  <TRNTYPE>{trntype}</TRNTYPE>",
            f"  <DTPOSTED>{row.dtposted}</DTPOSTED>",
            f"  <TRNAMT>{float(row.trnamt):.2f}</TRNAMT>",
            f"  <FITID>{fitid}</FITID>",
            ]
        if checknum:
            xml.append(f"  <CHECKNUM>{checknum}</CHECKNUM>")
        if row.name:
            xml.append(f"  <NAME>{row.name.upper()}</NAME>")
        if row.memo:
            xml.append(f"  <MEMO>{row.memo.upper()} (_{trntype}_)</MEMO>")
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
