import numpy as np
import pandas as pd
import os
import re
from typing import List, Dict, Any, Union, Tuple
from scipy.spatial.distance import euclidean, cityblock  # For norm 2 and 1


# --- Helper Functions (Replicating MATLAB's internal functions) ---

def _getnum(line: str) -> Union[float, None]:
    """
    MATLAB's getnum function equivalent.
    Attempts to extract the first numeric value from a line.
    """
    # Split the line by whitespace, then try to convert each part to float
    parts = line.split()
    for part in parts:
        try:
            return float(part)
        except ValueError:
            continue
    return None


def _cosang(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    MATLAB's cosang function equivalent.
    Calculates the cosine of the angle between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0  # Handle zero vectors gracefully
    return dot_product / (norm_v1 * norm_v2)


def _get_term_file(fpath: str, name: str) -> str:
    """
    MATLAB's get_term_file function equivalent.
    Finds a matching terms file based on naming conventions.
    """
    potential_files = [
        os.path.join(fpath, f"{name}.trm.txt"),
        os.path.join(fpath, f"{name}.trm"),
        os.path.join(fpath, "terms.txt")
    ]
    for tfile in potential_files:
        if os.path.exists(tfile):
            return tfile
    return ""  # Return empty string if no file is found


def _readterms(file: str) -> List[str]:
    """
    MATLAB's readterms function equivalent.
    Attempts to read terms (lines) from a file, skipping blank lines.
    """
    terms = []
    if os.path.exists(file):
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                term = line.strip()
                if term:  # Only add non-empty, non-whitespace lines
                    terms.append(term)
        except Exception:
            # On failure, return empty list
            return []
    return terms


def _get_terms(fpath: str, name: str, n: int) -> List[str]:
    """
    MATLAB's get_terms function equivalent.
    Tries to locate n terms for nodes; returns node numbers if file not found or count mismatch.
    """
    tfile = _get_term_file(fpath, name)
    terms = _readterms(tfile)
    #print(f"file: {tfile} terms: {terms} type: {type(terms[0])}")
    if len(terms) == n:
        return terms
    else:
        # Returns string representations of 1 to n (node numbers)
        return [str(i) for i in range(1, n + 1)]


def _write_excel_file(terms: List[str], dismat: np.ndarray, file: str) -> None:
    """
    MATLAB's write_excel_file function equivalent.
    Writes the distance matrix to an Excel file using pandas.
    NaN is used for infinite distances.
    """
    # Replace numpy inf with NaN for proper Excel writing
    dismat_clean = np.where(np.isinf(dismat), np.nan, dismat)

    # Create a pandas DataFrame
    df = pd.DataFrame(dismat_clean, index=terms, columns=terms)

    # Write to Excel
    df.to_excel(file, index=True, header=True)


# --- Main Function ---

def legacy_prx_files(path: str = None, files: List[str] = None,
                     write_excel: bool = False) -> Dict[str, Any]:
    """
    Replicates the functionality of legacy_prx_files.m.
    Reads proximity data from legacy text files and generates a proximity matrix.

    :param path: The directory path containing the files.
    :param files: A list of proximity file names (e.g., ['file1.prx.txt', 'file2.prx.txt']).
    :param write_excel: Boolean to indicate whether to write the result to a file with spreadheet format.
    :return: A dictionary where keys are dataset names and values are data structures (dictionaries).
    """

    col: Dict[str, Dict[str, Any]] = {}

    def _error(data: Dict[str, Any], message: str) -> None:
        """Internal error handler (replicating MATLAB's nested function)."""
        data['error'] = message
        col[data['name']] = data
        # print(message)
        # In a real application, you'd log or raise,
        # but here we just replicate the MATLAB structure.

    # if files is None, user selects multiple .txt files, filling both path and files
    if files is None:
        from tkinter.filedialog import askopenfilenames
        from tkinter import Tk
        root = Tk()
        root.withdraw()
        full_paths = askopenfilenames(title="Select Proximity Text Files",
                                  filetypes=[("Text Files", "*.txt")])
        #full_paths = str(full_paths[0])
        root.destroy()
        if not full_paths:
            # User canceled the selection
            return {}  # Return empty dictionary or handle as needed

        # Extract the common directory (path) and list of filenames (files)
        # The first file's directory is assumed to be the common path.
        path = os.path.dirname(full_paths[0])
        files = [os.path.basename(p) for p in full_paths]

    for file_name in files:
        data: Dict[str, Any] = {'error': None, 'file': file_name, 'path': path,
                                'name': os.path.splitext(os.path.basename(file_name))[0].split('.')[0]}

        full_file_path = os.path.join(path, file_name)

        try:
            # Read all lines and clean them up
            with open(full_file_path, 'r') as f:
                raw_lines = f.readlines()
            # Process lines: trim whitespace and convert to lowercase
            lines = [line.strip().lower() for line in raw_lines]
            lines = [line for line in lines if line]  # Skip empty lines
        except Exception as e:
            _error(data, f"Error: cannot read {file_name}. Details: {e}")
            continue

        # Use a list to simulate line reading and consumption (nextline)
        lines_iterator = list(lines)

        def _nextline() -> str:
            """Simulates the MATLAB nextline function by popping the first element."""
            if lines_iterator:
                return lines_iterator.pop(0)
            return ""

        # --- File Structure Check and Parsing ---

        # 1. 'data' keyword check
        first_line = _nextline()
        if 'data' not in first_line:
            _error(data, f"Error: {file_name} does not appear to be a proximity data file. (Missing 'data')")
            continue

        # 2. distances or similarities
        second_line = _nextline()
        data['isdistance'] = 'dis' in second_line

        # 3. nnodes
        data['nnodes'] = int(_getnum(_nextline()) or 0)
        if data['nnodes'] <= 0:
            _error(data, f"Error: cannot get number of items from {file_name}")
            continue

        # 4. terms
        data['terms'] = _get_terms(data['path'], data['name'], data['nnodes'])

        # 5. comment
        data['comment'] = _nextline()

        # 6. minprx
        min_val = _getnum(_nextline())
        if min_val is None:
            _error(data, f"Error: cannot get minimum value from {file_name}")
            continue
        data['minprx'] = min_val

        # 7. maxprx
        max_val = _getnum(_nextline())
        if max_val is None:
            _error(data, f"Error: cannot get maximum value from {file_name}")
            continue
        data['maxprx'] = max_val

        # 8. data shape and number of data values
        shp = _nextline()
        normval = 2  # Default norm
        hamming = False
        standardize = False
        cosine = False

        if 'coord' in shp or 'attrib' in shp or 'featur' in shp:
            data['isdistance'] = True
            data['comment'] += " " + shp
            data['shape'] = 'coord'

            ndims = int(_getnum(_nextline()) or 0)
            nvals = ndims * data['nnodes']

            metric = _nextline()
            if 'euclid' in metric:
                normval = 2
            elif 'city' in metric:
                normval = 1
            elif 'domin' in metric:
                normval = float('inf')
            else:
                normval = 2

            hamming = 'hamming' in metric
            standardize = 'standard' in metric
            cosine = 'cosine' in metric

        elif 'lower' in shp:
            data['shape'] = 'lower'
            nvals = (data['nnodes'] - 1) * data['nnodes'] // 2

        elif 'upper' in shp:
            data['shape'] = 'upper'
            nvals = (data['nnodes'] - 1) * data['nnodes'] // 2

        elif 'matrix' in shp:
            nvals = data['nnodes'] ** 2
            data['shape'] = 'matrix'

        elif 'list' in shp:
            data['shape'] = 'list'
            npairs = int(_getnum(_nextline()) or 0)
            nvals = 3 * npairs

            isdir = _nextline()
            data['isdirected'] = not ('non' in isdir or 'as' in isdir or 'uns' in isdir or isdir.startswith('dir'))

        else:
            _error(data, f"Error: unknown data shape in {file_name}")
            continue

        # 9. Read raw data values

        # Join the remaining lines and split into values (using numpy for efficiency)
        try:
            # All remaining lines should be the data points
            data_values_str = ' '.join(lines_iterator)
            dat = np.array([float(x) for x in data_values_str.split() if x])
        except ValueError:
            _error(data, f"Error in: {file_name}. Data values could not be parsed as numbers.")
            continue

        if dat.size != nvals:
            _error(data, f"Error in: {file_name}. Expected {nvals} data values, found {dat.size}")
            continue

        # 10. Initialize proximity matrix
        prx = np.full((data['nnodes'], data['nnodes']), np.inf)

        # 11. Populate proximity matrix based on shape

        if data['shape'] == 'coord':
            # Coordinate-based distance calculation
            data['minprx'] = float('inf')
            data['maxprx'] = float('-inf')

            # Reshape coordinates: (nnodes, ndims)
            dat = dat.reshape(data['nnodes'], ndims)

            if standardize:
                # Normalize each vector by its L1 norm
                l1_norms = np.linalg.norm(dat, ord=1, axis=1, keepdims=True)
                # Avoid division by zero, set to 1 if norm is 0
                l1_norms[l1_norms == 0] = 1
                dat = dat / l1_norms

            for r in range(data['nnodes']):
                for c in range(r):  # Only calculate lower triangle (r > c)
                    vec_r = dat[r, :]
                    vec_c = dat[c, :]

                    if hamming:
                        d = np.sum(vec_r != vec_c)
                    elif cosine:
                        d = 1 - _cosang(vec_r, vec_c)
                    elif normval == 2:
                        d = euclidean(vec_r, vec_c)
                    elif normval == 1:
                        d = cityblock(vec_r, vec_c)
                    elif normval == float('inf'):
                        d = np.linalg.norm(vec_r - vec_c, ord=np.inf)
                    else:
                        # Fallback to Euclidean
                        d = euclidean(vec_r, vec_c)

                    # Floating point precision correction
                    if abs(d) < 1e-14:
                        d = 0.0

                    if np.isnan(d):
                        d = 0.0

                    prx[r, c] = d
                    prx[c, r] = d  # Symmetric

                    data['minprx'] = min(d, data['minprx'])
                    data['maxprx'] = max(d, data['maxprx'])
            data['isdirected'] = False

        elif data['shape'] == 'lower':
            # Lower triangle (symmetric)
            k = 0
            for r in range(1, data['nnodes']):
                for c in range(r):
                    prx[r, c] = dat[k]
                    prx[c, r] = dat[k]
                    k += 1
            data['isdirected'] = False

        elif data['shape'] == 'upper':
            # Upper triangle (symmetric)
            k = 0
            for r in range(data['nnodes']):
                for c in range(r + 1, data['nnodes']):
                    prx[r, c] = dat[k]
                    prx[c, r] = dat[k]
                    k += 1
            data['isdirected'] = False

        elif data['shape'] == 'matrix':
            # Full matrix (row-major order in the MATLAB code)
            prx = dat.reshape(data['nnodes'], data['nnodes'])

        elif data['shape'] == 'list':
            # List of (row, col, value)
            dat = dat.reshape(3, npairs).astype(float)
            for k in range(npairs):
                r = int(dat[0, k]) - 1  # Adjust to 0-based index
                c = int(dat[1, k]) - 1  # Adjust to 0-based index
                d = dat[2, k]

                # Check bounds
                if 0 <= r < data['nnodes'] and 0 <= c < data['nnodes']:
                    prx[r, c] = d
                # else: Data point out of bounds, skip or log error

        # 12. Final processing and cleanup

        # Convert NaN to inf
        prx[np.isnan(prx)] = np.inf

        # Apply min/max proximity filter (values outside range are inf)
        prx[(prx < data['minprx']) | (prx > data['maxprx'])] = np.inf

        # Determine directedness if not already set (only for 'matrix' and 'list')
        if 'isdirected' not in data or data['shape'] == 'matrix' or data['shape'] == 'list':
            # Check for asymmetry: prx != prx.T
            data['isdirected'] = not np.allclose(prx, prx.T, equal_nan=True)

        # Convert similarity to distance (if not a distance measure)
        if not data['isdistance']:
            # The transformation is: dis = minprx + maxprx - sim
            prx = data['minprx'] + data['maxprx'] - prx
            # Re-apply range filter
            prx[(prx < data['minprx']) | (prx > data['maxprx'])] = np.inf

        # Ensure symmetry for undirected graphs by taking the minimum of A and A.T
        if not data['isdirected']:
            prx = np.minimum(prx, prx.T)

        # Final range filter again
        prx[(prx < data['minprx']) | (prx > data['maxprx'])] = np.inf

        # Set diagonal to 0
        np.fill_diagonal(prx, 0)

        data['dismat'] = prx
        col[data['name']] = data

        # 13. Write Excel file
        if write_excel:
            excel_file_path = os.path.join(data['path'], f"{data['name']}.prx.xlsx")
            _write_excel_file(data['terms'], data['dismat'], excel_file_path)

    return col

if __name__ == '__main__':
    from pypf.proximity import Proximity
    dat = legacy_prx_files("data",["aviation.prx.txt"])
    print(f"dat {dat}")
    dat = dat[list(dat.keys())[0]]
    print(dat["name"], dat["terms"][1], dat["dismat"][1,1])
    prx = Proximity(name=dat["name"], terms=dat["terms"], dismat=dat["dismat"])
    prx.prxprint()


