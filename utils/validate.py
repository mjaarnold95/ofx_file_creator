"""Validation helpers for OFX generation."""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

# Columns that must be present before attempting to render OFX output.
REQUIRED_COLUMNS = {"amount_clean"}

# Timestamp-like columns that can act as a fallback when ``date_parsed`` is
# missing or empty.  The ordering reflects how closely each column maps to the
# posting timestamp required by OFX files.
FALLBACK_TIMESTAMP_COLUMNS: Iterable[str] = (
    "statement_end_date",
    "statement_end",
    "statement_begin_date",
    "statement_begin",
    "date",
)


def _coerce_series_to_utc(series: pd.Series) -> pd.Series:
    """Return the series converted to UTC ``Timestamp`` values."""

    if series.empty:
        return series

    converted = pd.to_datetime(series, errors="coerce", utc=True)
    return converted.dropna()


def _first_available_timestamp(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Return the first usable timestamp from known fallback columns."""

    for column in FALLBACK_TIMESTAMP_COLUMNS:
        if column not in df.columns:
            continue
        timestamps = _coerce_series_to_utc(df[column])
        if not timestamps.empty:
            return timestamps.max()
    return None


def assert_ofx_ready(df: pd.DataFrame) -> pd.Timestamp:
    """Validate that the DataFrame contains the fields required for OFX."""

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"OFX generation requires the following columns: {missing_list}"
        )

    amounts = df["amount_clean"]
    if not amounts.notna().any():
        raise ValueError(
            "OFX generation requires at least one non-null 'amount_clean' value."
        )

    timestamp_series = None
    if "date_parsed" in df.columns:
        timestamp_series = _coerce_series_to_utc(df["date_parsed"])

    if timestamp_series is not None and not timestamp_series.empty:
        return timestamp_series.max()

    fallback = _first_available_timestamp(df)
    if fallback is None:
        raise ValueError(
            "OFX generation requires at least one timestamp column such as "
            "'date_parsed' or 'statement_end_date'."
        )

    return fallback

