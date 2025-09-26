from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ---------- datetime helpers ----------
def ofx_datetime(dt: Optional[pd.Timestamp]) -> str | None:
    if dt is None or (isinstance(dt, pd.Timestamp) and pd.isna(dt)):
        if isinstance(dt, pd.Timestamp):
            dt = (
                dt.tz_localize(timezone.utc)
                if dt.tzinfo is None
                else dt.tz_convert(timezone.utc)
            )
            py_dt = dt.to_pydatetime()
        else:
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=timezone.utc)
            py_dt = dt.astimezone(timezone.utc)
        return f"{py_dt.strftime('%Y%m%d%H%M%S')}.000[0:UTC]"
    return None

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