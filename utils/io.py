from pathlib import Path
from typing import Iterable

import pandas as pd

from utils.sheet import find_best_sheet


_EXCEL_SUFFIXES: Iterable[str] = (".xls", ".xlsx", ".xlsm", ".xlsb")


def load_transactions(path: Path) -> pd.DataFrame:
    """Load a transaction file into a DataFrame with consistent dtypes.

    The loader inspects the file suffix and dispatches to :func:`pandas.read_excel`
    or :func:`pandas.read_csv` while ensuring data is kept as ``object`` dtype so
    downstream normalization operates identically for each format.
    """

    suffix = path.suffix.lower()
    if suffix in _EXCEL_SUFFIXES:
        xl = pd.ExcelFile(path)
        sheet = find_best_sheet(xl)
        return xl.parse(sheet_name=sheet, dtype=object)

    if suffix == ".csv":
        return pd.read_csv(path, dtype=object)

    raise ValueError(f"Unsupported transaction file type: {suffix or path}")
