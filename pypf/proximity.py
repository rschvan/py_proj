#   pypf/proximity.py
from pypf.utility import coherence, get_off_diagonal, get_lower, get_termsid
import pandas as pd
#from tkinter import Tk
#from tkinter.filedialog import askopenfilename
import numpy as np
import os  #   To work with file paths

class Proximity:
    """
    Represents a proximity object with items and their pairwise distances.
    Assumes the data file is either:
    1.  A square matrix where the first row is a header (terms), and the first column (from the second row onwards) contains the same terms.
    2.  A matrix with one more column than rows, where the first column contains the terms, and the following columns form the square distance matrix (no header row).
    """
    def __init__(self, filepath=None, terms=None, dismat=None, name=None):
        """
        creates a Proximity object.

        Args:
            filepath (str, optional): Path to a spreadsheet file (.xlsx or .csv) with terms and distances. If provided, loads data from the file and extracts a name. If None, triggers a file dialog. Defaults to None.
            terms (list, optional): A list of item names (used if data is not loaded from a file). Defaults to None.
            dismat (numpy.ndarray, optional): A square matrix of pairwise distances (used if data is not loaded from a file). Defaults to None.
            name (str, optional): A name for the proximity object. If None, uses the filename (if available) or the first term in the list. Defaults to None.
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

        if dismat is None:  # must read a file
            if filepath is None: # must prompt for the file
                filepath = askopenfilename(
                    title="Select a spreadsheet file",
                    filetypes=[("Spreadsheet files", "*.xlsx *.csv")]
                )
                if not filepath:
                    print("No file selected.")
                    return
            self.filepath = filepath
            self.filename = os.path.basename(filepath)
            self.name, _ = os.path.splitext(self.filename) # returns filename and ext
            parts = self.name.split(".")
            self.name = parts[0] if len(parts) > 1 else self.name
            try:
                df = self._read_data_file(filepath)
                nr, nc = df.shape
                if nr == nc: # has terms on the first row
                    terms = df.iloc[1:,0].tolist()
                    dismat = df.iloc[1:,1:].apply(pd.to_numeric, errors='coerce', downcast='float') # dismat from [1:,1:]
                elif nr == nc-1:
                    terms = df.iloc[:,0].tolist()
                    dismat = df.iloc[:,1:].apply(pd.to_numeric, errors='coerce', downcast='float') # dismat from [:,1:]
            except Exception as e:
                print(f"Error reading file: {e}")
                return
        else: # create proximity from inputs
            dismat = np.array(dismat)
            nr, nc = dismat.shape
            if terms is None:
                print(f"Error expected terms: )")
                return
            if nr != nc:
                print(f"Variable inputs are not valid: ")
                return
        self.terms = terms
        self.nterms = len(self.terms)
        self.termsid = get_termsid(self.terms)

        if name is not None:
            self.name = name
        self.dismat = np.copy(dismat)
        if self.dismat.shape != (self.nterms, self.nterms):
            print(f"error: terms and distance matrix are not congruent")
            return
        np.fill_diagonal(self.dismat, 0.0)
        self.dismat[np.isnan(self.dismat)] = np.inf
        self.issymmetric = np.array_equal(self.dismat, self.dismat.T)
        if self.dismat.size > 0:
            self.calculate_stats()
            self.coh = coherence(self.dismat, self.max if hasattr(self, 'max') else np.nan)

    def _read_data_file(self, filepath):
        """Reads the data file (xlsx or csv) into a Pandas DataFrame."""
        if filepath.endswith(('.xlsx', '.xls')):
            return pd.read_excel(filepath, header=None)
        else:
            return pd.read_csv(filepath, header=None)

    def calculate_stats(self):
        """
        Calculates descriptive statistics for the off-diagonal distances, excluding infinity and NaN.
        """
        datapoints = []
        n = self.dismat.shape[0]
        if self.issymmetric:
            datapoints = get_lower(self.dismat)
        else:
            datapoints = get_off_diagonal(self.dismat)

        datapoints.flatten()
        datapoints = datapoints[~np.isinf(datapoints)]
        datapoints = datapoints[~np.isnan(datapoints)]
        self.mean = np.mean(datapoints).round(2)
        self.sd = np.std(datapoints).round(2)
        self.min = np.min(datapoints)
        self.max = np.max(datapoints)

    def get_info(self):
        info = {}
        info["name"] = self.name
        info["nterms"] = self.nterms
        info["coherence"] = self.coh
        info["mean"] = self.mean
        info["sd"] = self.sd
        info["min"] = self.min
        info["max"] = self.max
        info["termsid"] = self.termsid
        return info

    def prxprint(self, note=None):
        from pypf.proximity import Proximity  #   Import Proximity here
        if note is None:
            note = self.name
        print("\nProximity:", note)
        print(f"filename: {self.filename}")
        print(f"filepath: {self.filepath}")
        print(f"name: {self.name}")
        print(f"nterms: {self.nterms}")  #   Changed from nterms to n
        print(f"terms[0]: {self.terms[0]}")
        print(f"termsid: {self.termsid}")
        print(f"dismat[0,1]: {self.dismat[0,1]}")
        print(f"issymmetric: {self.issymmetric}")
        print(f"mean: {self.mean}")
        print(f"sd: {self.sd}")
        print(f"min: {self.min}")
        print(f"max: {self.max}")
        print(f"coh: {self.coh}")

if __name__ == "__main__":
    p = Proximity("data\psy.prx.xlsx")
    p.prxprint()
    print(p.get_info())