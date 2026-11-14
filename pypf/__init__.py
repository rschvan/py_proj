# pypf/__init__.py
"""
PyPathfinder (pypf) is a Python package for creating and analyzing Pathfinder Networks.

It provides tools to:
- Read proximity data files (pypf.readers) into Proximity objects.
- Derive Pathfinder and nearest-neighbor networks (PFnet) and compute layouts.
- Manage sets of proximities and networks, and save them as projects (Collection).

Sample data ships with the package:
    >>> import pypf
    >>> prx = pypf.load_sample("psy")
    >>> net = pypf.PFnet(prx, q=2)
"""
from importlib.resources import files
from pathlib import Path

__version__ = "0.1.0"

from pypf.proximity import Proximity, ProximityError
from pypf.pfnet import PFnet
from pypf.collection import Collection


def sample_path(filename: str = "") -> Path:
    """Path of a sample data file shipped with pypf (or of the data folder itself)."""
    return Path(str(files("pypf") / "data")) / filename


def load_sample(name: str) -> Proximity:
    """Loads a sample proximity by name, e.g. "psy", "bio", "cities" or "bank6"."""
    from pypf.readers import read_spreadsheet

    for ext in (".prx.xlsx", ".prx.csv"):
        path = sample_path(name + ext)
        if path.is_file():
            return read_spreadsheet(path.name, path.read_bytes())
    raise FileNotFoundError(f"no sample proximity named {name!r} in {sample_path()}")
