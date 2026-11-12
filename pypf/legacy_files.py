import numpy as np
import os
from scipy.spatial.distance import euclidean, cityblock  # For norm 2 and 1

# --- Helper Functions  ---

def _getnum(line: str) -> float | None:
    """
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
    Calculates the cosine of the angle between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0  # Handle zero vectors gracefully
    return dot_product / (norm_v1 * norm_v2)

# --- Main Function ---

def legacy_txt_files(prx_file:str, raw_lines:list[str]) :
    """
    Reads proximity data from legacy text files and generates a proximity matrix.
    :param prx_file: The name of the proximity file
    :param raw_lines: a list of txt lines in the file

    :return: A dictionary with keys for error, path, file, name, isdistance, nnodes, comment,
                        minprx, maxprx, shape, isdirected, dismat.
          a user may be mainly interested in the name and dismat, the distance matrix, a numpy ndarray
    """
    pdict: dict = {}
    pdict["error"] = None
    path, fname = os.path.split(prx_file)
    pdict["path"] = path
    pdict["file"] = fname
    pdict["name"] = os.path.splitext(fname)[0].split(".")[0]

    lines = [line.strip().lower() for line in raw_lines]
    lines = [line for line in lines if line]  # Skip empty lines
    lines_iterator = list(lines)

    def _nextline() -> str:
        if lines_iterator:
            return lines_iterator.pop(0)
        return ""

    # 'data' keyword check
    first_line = _nextline()
    if 'data' not in first_line:
        pdict["error"] = f"Error: {fname} does not appear to be a proximity data file. (Missing 'data')"
        return pdict

    # distances or similarities
    second_line = _nextline()
    pdict['isdistance'] = 'dis' in second_line

    # nnodes
    pdict['nnodes'] = int(_getnum(_nextline()) or 0)
    if pdict['nnodes'] <= 0:
        pdict["error"] = f"Error: cannot get number of items"
        return pdict

    # comment
    pdict['comment'] = _nextline()

    # minprx
    min_val = _getnum(_nextline())
    if min_val is None:
        pdict["error"] = f"Error: cannot get minimum value "
        return pdict
    pdict['minprx'] = min_val

    # maxprx
    max_val = _getnum(_nextline())
    if max_val is None:
        pdict["error"] = f"Error: cannot get maximum value"
        return pdict
    pdict['maxprx'] = max_val

    # data shape and number of data values
    shp = _nextline()
    normval = 2  # Default norm
    hamming = False
    standardize = False
    cosine = False

    if 'coord' in shp or 'attrib' in shp or 'featur' in shp:
        pdict['isdistance'] = True
        pdict['comment'] += " " + shp

        ndims:int = int(_getnum(_nextline()) or 0)
        if ndims == 0:
            pdict["error"] = f"Error: cannot get number of vectors"
            return pdict
        nvals = ndims * pdict['nnodes']

        metric = _nextline()
        pdict['shape'] = "vectors: " + metric
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
        pdict['shape'] = 'lower'
        nvals = (pdict['nnodes'] - 1) * pdict['nnodes'] // 2

    elif 'upper' in shp:
        pdict['shape'] = 'upper'
        nvals = (pdict['nnodes'] - 1) * pdict['nnodes'] // 2

    elif 'matrix' in shp:
        nvals = pdict['nnodes'] ** 2
        pdict['shape'] = 'matrix'

    elif 'list' in shp:
        pdict['shape'] = 'list'
        npairs = int(_getnum(_nextline()) or 0)
        nvals = 3 * npairs

        isdir = _nextline()
        pdict['isdirected'] = False
        if (isdir.startswith('dir') or isdir.startswith('nons') or
                isdir.startswith('as') or isdir.startswith('und')):
            pdict['isdirected'] = True
    else:
        pdict["error"] = f"Error: unknown data shape"
        return pdict

    # Read raw data values
    # Join the remaining lines and split into values (using numpy for efficiency)
    try:
        # All remaining lines should be the data points
        data_values_str = ' '.join(lines_iterator)
        dat = np.array([float(x) for x in data_values_str.split() if x])
    except ValueError:
        pdict["error"] = f"Data values could not be parsed as numbers."
        return pdict

    if dat.size != nvals:
        pdict["error"] = f"Expected {nvals} data values, found {dat.size}"
        return pdict

    #  Initialize proximity matrix
    prx = np.full((pdict['nnodes'], pdict['nnodes']), np.inf)

    #  Populate proximity matrix based on shape

    if "vectors" in pdict['shape']:
        # Vector-based distance calculation
        pdict['minprx'] = float('inf')
        pdict['maxprx'] = float('-inf')

        # Reshape coordinates: (nnodes, ndims)
        dat = dat.reshape(pdict['nnodes'], ndims)

        if standardize:
            # Normalize each vector by its L1 norm
            l1_norms = np.linalg.norm(dat, ord=1, axis=1, keepdims=True)
            # Avoid division by zero, set to 1 if norm is 0
            l1_norms[l1_norms == 0] = 1
            dat = dat / l1_norms

        for r in range(pdict['nnodes']):
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

                pdict['minprx'] = min(d, pdict['minprx'])
                pdict['maxprx'] = max(d, pdict['maxprx'])
        pdict['isdirected'] = False

    elif pdict['shape'] == 'lower':
        # Lower triangle (symmetric)
        k = 0
        for r in range(1, pdict['nnodes']):
            for c in range(r):
                prx[r, c] = dat[k]
                prx[c, r] = dat[k]
                k += 1
        pdict['isdirected'] = False

    elif pdict['shape'] == 'upper':
        # Upper triangle (symmetric)
        k = 0
        for r in range(pdict['nnodes']):
            for c in range(r + 1, pdict['nnodes']):
                prx[r, c] = dat[k]
                prx[c, r] = dat[k]
                k += 1
        pdict['isdirected'] = False

    elif pdict['shape'] == 'matrix':
        # Full matrix (row-major order)
        prx = dat.reshape(pdict['nnodes'], pdict['nnodes'])

    elif pdict['shape'] == 'list':
        # List of (row, col, value)
        dat = dat.reshape(npairs,3).astype(float)
        for k in range(npairs):
            r = int(dat[k, 0]) - 1  # Adjust to 0-based index
            c = int(dat[k, 1]) - 1  # Adjust to 0-based index
            d = dat[k, 2]

            # Check bounds
            if 0 <= r < pdict['nnodes'] and 0 <= c < pdict['nnodes']:
                prx[r, c] = d
                if not pdict['isdirected']:
                    prx[c, r] = d
            else: #Data point out of bounds, skip or log error
                pdict["error"] = "List index out of range."
                return pdict

    # Final processing and cleanup

    # Convert NaN to inf
    prx[np.isnan(prx)] = np.inf

    # Apply min/max proximity filter (values outside range are inf)
    prx[(prx < pdict['minprx']) | (prx > pdict['maxprx'])] = np.inf

    # Determine directedness if not already set (only for 'matrix' and 'list')
    if 'isdirected' not in pdict or pdict['shape'] == 'matrix':
        # Check for asymmetry: prx != prx.T
        pdict['isdirected'] = not np.allclose(prx, prx.T, equal_nan=True)

    # Convert similarity to distance (if not a distance measure)
    if not pdict['isdistance']:
        # The transformation is: dis = minprx + maxprx - sim
        prx = pdict['minprx'] + pdict['maxprx'] - prx
        # Re-apply range filter
        prx[(prx < pdict['minprx']) | (prx > pdict['maxprx'])] = np.inf

    # Ensure symmetry for undirected graphs by taking the minimum of A and A.T
    if not pdict['isdirected']:
        prx = np.minimum(prx, prx.T)

    # Final range filter again
    prx[(prx < pdict['minprx']) | (prx > pdict['maxprx'])] = np.inf

    # Set diagonal to 0
    np.fill_diagonal(prx, 0)

    pdict['dismat'] = np.round(prx, 6)

    return pdict

if __name__ == '__main__':
    from pypf.utility import get_terms
    # p = os.path.join("C:\\","Users","schva","Prog","datasets","prxtest",)
    path = os.path.join("C\\","easy")
    file = "t_liv002007.prx.txt"
    file_path = os.path.join(path, file)
    with open(file_path) as f:
        flines = f.readlines()
    prxdict = legacy_txt_files(prx_file=file, raw_lines=flines)
    print(prxdict)
    print(prxdict['file'])
    print(prxdict['isdirected'])
    print(prxdict['shape'])
