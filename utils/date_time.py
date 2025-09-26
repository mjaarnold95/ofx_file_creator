from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------- datetime helpers ----------
def _coerce_to_timestamp(
    dt: Union[pd.Timestamp, datetime, str, float, int, None]
) -> Optional[pd.Timestamp]:
    """Return a timezone-aware ``Timestamp`` in UTC or ``None``.

    The real application code often passes in ``pd.Timestamp`` objects, but the
    helpers are occasionally called with plain ``datetime`` objects or strings
    (for example when generating FITIDs).  The previous implementation attempted
    to handle every case inline which made the control-flow difficult to follow
    and, worse, converted ``None`` inputs into the literal string ``"None"``.

    Having a tiny coercion helper keeps ``ofx_datetime`` focused on the one job
    the tests care about – producing correctly formatted OFX timestamps.
    """

    if dt is None:
        return None

    if isinstance(dt, pd.Timestamp):
        ts = dt
    else:
        try:
            ts = pd.to_datetime(dt, errors="coerce")
        except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
            return None

    if ts is pd.NaT or pd.isna(ts):
        return None

    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts


def ofx_datetime(dt: Optional[pd.Timestamp]) -> Optional[str]:
    """Format a datetime-like value as an OFX timestamp.

    ``None`` or ``NaT`` inputs simply return ``None`` which allows callers to
    decide on sensible fallbacks.  Valid datetimes are normalised to UTC and
    formatted according to the ``YYYYMMDDHHMMSS.000[0:UTC]`` convention used by
    OFX files.
    """

    ts = _coerce_to_timestamp(dt)
    if ts is None:
        return None

    py_dt = ts.to_pydatetime()
    return f"{py_dt.strftime('%Y%m%d%H%M%S')}.000[0:UTC]"

def parse_date(val) -> pd.Timestamp:
    # robust parse -> UTC
    return pd.to_datetime(val, errors="coerce", utc=True)

def parse_time_to_timedelta(val) -> Optional[pd.Timedelta]:
    """Return a pure time-of-day as Timedelta (hh:mm:ss). None if not parseable."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    
    # Excel time fraction (0..1)
    if isinstance(val, (int, float)) and 0 <= float(val) < 1:
        total = int(round(float(val) * 86400))
        return pd.to_timedelta(total, unit="s")
    
    s = str(val).strip()
    if not s:
        return None
    
    fmts = ["%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p", "%I%p", "%H%M%S", "%H%M"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return (
                    pd.to_timedelta(dt.hour, unit="h")
                    + pd.to_timedelta(dt.minute, unit="m")
                    + pd.to_timedelta(dt.second, unit="s")
            )
        except ValueError:
            continue
    
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.notna(ts):
            return (
                pd.to_timedelta(ts.hour, unit="h")
                + pd.to_timedelta(ts.minute, unit="m")
                + pd.to_timedelta(ts.second, unit="s")
            )
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
        pass
    return None
