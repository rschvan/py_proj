# pypf/readers.py: reading proximity data files.
# Works with file names and contents (bytes), so callers can pass files from disk, from an
# upload widget, or from anywhere else. Nothing here knows about Streamlit or Collection.
import io
import os

import numpy as np
import pandas as pd

from pypf.legacy_files import legacy_txt_files
from pypf.proximity import Proximity, ProximityError

SPREADSHEET_EXTENSIONS = (".xlsx", ".xls", ".csv")


def proximity_name(filename: str) -> str:
    """The proximity name for a file: its base name up to the first period."""
    return os.path.basename(filename).split(".")[0]


def read_table(source, filename: str) -> pd.DataFrame:
    """Reads a spreadsheet (.xlsx/.xls/.csv) with no header row. `source` is a path, a
    file-like object, or bytes; `filename` decides the format."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(source, header=None, engine="openpyxl")
        return pd.read_csv(source, header=None)
    except Exception as e:
        raise ProximityError(f"could not read the file ({e})") from e


def table_to_matrix(df: pd.DataFrame, filename: str = "") -> tuple[list, np.ndarray]:
    """Splits a spreadsheet table into terms and a distance matrix.

    Accepted layouts, with n terms (empty rows and columns are ignored):
      * n+1 rows x n+1 columns: terms in the first row and in the first column.
      * n rows x n+1 columns: terms in the first column only.
    Cells that are not numbers become infinite distances.
    """
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    nrows, ncols = df.shape
    if nrows == ncols:
        terms = df.iloc[1:, 0].tolist()
        values = df.iloc[1:, 1:]
    elif nrows == ncols - 1:
        terms = df.iloc[:, 0].tolist()
        values = df.iloc[:, 1:]
    else:
        raise ProximityError(
            f"the sheet has {nrows} rows and {ncols} columns; with n terms it needs "
            "n+1 rows and n+1 columns (terms in the first row and column) "
            "or n rows and n+1 columns (terms in the first column)")
    if any(pd.isna(t) or str(t).strip() == "" for t in terms):
        raise ProximityError("a term name is missing (blank cell in the first column)")
    terms = [_term_text(t) for t in terms]
    dismat = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return terms, dismat


def _term_text(term) -> str:
    """Term names as text; numbers read from a sheet (e.g. years) lose a trailing .0."""
    if isinstance(term, float) and term.is_integer():
        return str(int(term))
    return term if isinstance(term, str) else str(term)


def read_spreadsheet(filename: str, data) -> Proximity:
    """Reads a proximity spreadsheet. `data` is the file's bytes, a file-like object, or a path.
    Raises ProximityError with the reason if it is not a valid proximity."""
    terms, dismat = table_to_matrix(read_table(data, filename), filename)
    prx = Proximity(terms=terms, dismat=dismat, name=proximity_name(filename))
    prx.filename = os.path.basename(filename)
    return prx


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def is_terms_file(filename: str) -> bool:
    """Legacy terms files: name.trm.txt (for one data file) or terms.txt (for all)."""
    lower = os.path.basename(filename).lower()
    return lower.endswith(".trm.txt") or lower == "terms.txt"


def _find_terms(name: str, n: int, terms_files: dict[str, list[str]]) -> list[str]:
    """Terms for legacy data file `name`: name.trm.txt first, then terms.txt.
    A file is used only if it has exactly n terms; otherwise terms are numbered 0..n-1."""
    for candidate in (f"{name}.trm.txt".lower(), "terms.txt"):
        terms = terms_files.get(candidate)
        if terms is not None and len(terms) == n:
            return terms
    return [str(i) for i in range(n)]


def read_legacy_files(files: dict[str, bytes]) -> tuple[list[Proximity], list[str]]:
    """Reads legacy text proximity files together with their terms files.

    `files` maps file names to contents. Terms files (see is_terms_file) are matched to data
    files by name. Returns the proximities read and an error message for each data file that
    could not be read, in file-name order.
    """
    terms_files = {}
    for fname, data in files.items():
        if is_terms_file(fname):
            lines = _decode(data).splitlines()
            terms_files[os.path.basename(fname).lower()] = [s for line in lines if (s := line.strip())]

    proximities, errors = [], []
    for fname in sorted(files):
        if is_terms_file(fname):
            continue
        base = os.path.basename(fname)
        try:
            pdict = legacy_txt_files(prx_file=base, raw_lines=_decode(files[fname]).splitlines())
            if pdict["error"]:
                errors.append(f"{base}: {pdict['error'].removeprefix('Error: ')}")
                continue
            terms = _find_terms(pdict["name"], pdict["nnodes"], terms_files)
            prx = Proximity(terms=terms, dismat=pdict["dismat"], name=pdict["name"])
            prx.filename = base
            proximities.append(prx)
        except Exception as e:  # ProximityError, or a malformed file the parser trips on
            errors.append(f"{base}: {e}")
    return proximities, errors


def read_files(files: dict[str, bytes], kind: str = "spreadsheet") -> tuple[list[Proximity], list[str]]:
    """Reads proximity files of one kind ("spreadsheet" or "legacy").
    Returns the proximities read and an error message for each file that failed."""
    if kind == "legacy":
        return read_legacy_files(files)
    if kind != "spreadsheet":
        raise ValueError(f"kind must be 'spreadsheet' or 'legacy', not {kind!r}")
    proximities, errors = [], []
    for fname in sorted(files):
        base = os.path.basename(fname)
        try:
            proximities.append(read_spreadsheet(base, files[fname]))
        except ProximityError as e:
            errors.append(f"{base}: {e}")
    return proximities, errors
