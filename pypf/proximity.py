#   pypf/proximity.py
import os

import numpy as np

from pypf.utility import coherence, get_off_diagonal, get_lower, get_termsid, sig_figs_flat


class ProximityError(ValueError):
    """Raised when data cannot make a valid Proximity. The message says why."""


class Proximity:
    """
    Represents a proximity object with items (terms) and their pairwise distances.

    Build one from a file with Proximity(filepath) (or the readers in pypf.readers), or from
    data with Proximity(terms=..., dismat=..., name=...). Invalid data raises ProximityError.
    The spreadsheet layouts accepted are described in pypf.readers.table_to_matrix.
    """
    def __init__(self, filepath=None, terms=None, dismat=None, name=None):
        """
        Creates a Proximity object.

        Args:
            filepath: Path (or file-like object with a .name) of a spreadsheet (.xlsx, .xls or
                .csv) with terms and distances. Its name, up to the first period, names the
                proximity. Used only when dismat is None.
            terms: Item names, when not reading a file.
            dismat: Square matrix of pairwise distances, when not reading a file.
            name: Name for the proximity (overrides the file name).
        With no arguments, an empty Proximity is returned.
        """
        self.terms = []
        self.termsid = ""
        self.nterms = 0
        self.dismat = np.array([])
        self.mean = np.nan
        self.sd = np.nan
        self.min = np.nan
        self.max = np.nan
        self.coh = np.nan
        self.name = None
        self.filename = None
        self.filepath = filepath
        self.issymmetric = False
        self.sources: list[str] = []

        if dismat is None:
            if filepath is None:  # empty Proximity
                return
            from pypf.readers import proximity_name, read_table, table_to_matrix
            self.filename = filepath.name if hasattr(filepath, "name") else os.path.basename(filepath)
            self.name = proximity_name(self.filename)
            terms, dismat = table_to_matrix(read_table(filepath, self.filename), self.filename)
        elif terms is None:
            raise ProximityError("terms are needed with a distance matrix")

        dismat = np.array(dismat, dtype=float)
        if dismat.ndim != 2 or dismat.shape[0] != dismat.shape[1]:
            raise ProximityError(f"the distance matrix must be square, not {dismat.shape}")
        terms = list(terms)  # kept exactly as given: termsid (used in project files) depends on them
        if dismat.shape[0] != len(terms):
            raise ProximityError(f"{len(terms)} terms but a {dismat.shape[0]} x {dismat.shape[1]} "
                                 "distance matrix")
        if len(terms) < 2:
            raise ProximityError("at least 2 terms are needed")
        duplicates = sorted({t for t in terms if terms.count(t) > 1})
        if duplicates:
            raise ProximityError(f"terms must be unique; repeated: {', '.join(duplicates)}")

        self.terms = terms
        self.nterms = len(terms)
        self.termsid = get_termsid(self.terms)
        if name is not None:
            self.name = name

        np.fill_diagonal(dismat, 0.0)
        dismat[np.isnan(dismat)] = np.inf
        off_diagonal = ~np.eye(self.nterms, dtype=bool)
        if not np.isfinite(dismat[off_diagonal]).any():
            raise ProximityError("there are no finite distances between different terms")
        self.dismat = dismat
        self.issymmetric = np.array_equal(self.dismat, self.dismat.T)
        self.calculate_stats()
        self.coh = coherence(self.dismat, self.max)

    def symmetrized(self) -> "Proximity":
        """A symmetric copy using the smaller of dij and dji for each pair, named with "_fs".
        Returns self if the distances are already symmetric."""
        if self.issymmetric:
            return self
        prx = Proximity(terms=list(self.terms), dismat=np.minimum(self.dismat, self.dismat.T),
                        name=f"{self.name}_fs")
        prx.filename, prx.filepath, prx.sources = self.filename, self.filepath, list(self.sources)
        return prx

    def calculate_stats(self):
        """
        Calculates descriptive statistics for the off-diagonal distances, excluding infinity and NaN.
        """
        if self.issymmetric:
            datapoints = get_lower(self.dismat)
        else:
            datapoints = get_off_diagonal(self.dismat)
        datapoints = datapoints.astype(float).flatten()
        datapoints = datapoints[np.isfinite(datapoints)]
        self.mean = np.mean(datapoints)
        self.sd = np.std(datapoints) if len(datapoints) > 1 else np.nan
        self.min = np.min(datapoints)
        self.max = np.max(datapoints)

    def get_info(self):
        info = {}
        info["name"] = self.name
        info["nterms"] = self.nterms
        info["symmetric?"] = self.issymmetric
        info["coherence"] = self.coh
        info["mean"] = sig_figs_flat(x=self.mean, n=3)
        info["sd"] = sig_figs_flat(x=self.sd, n=3)
        info["min"] = sig_figs_flat(x=self.min, n=3)
        info["max"] = sig_figs_flat(x=self.max, n=3)
        info["termsid"] = self.termsid
        return info

    def prxprint(self, note=None):
        if note is None:
            note = self.name
        print("\nProximity:", note)
        print(f"filename: {self.filename}")
        print(f"filepath: {self.filepath}")
        print(f"name: {self.name}")
        print(f"nterms: {self.nterms}")
        print(f"terms[0]: {self.terms[0]}")
        print(f"termsid: {self.termsid}")
        print(f"dismat[0,1]: {self.dismat[0,1]}")
        print(f"issymmetric: {self.issymmetric}")
        print(f"mean: {self.mean:.6g}")
        print(f"sd: {self.sd:.6g}")
        print(f"min: {self.min:.6g}")
        print(f"max: {self.max:.6g}")
        print(f"coh: {self.coh}")
