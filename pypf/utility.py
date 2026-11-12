# pypf/utility.py
import os
import math
from typing import Optional, Union
import numpy as np
import pandas as pd
import networkx as nx
import re
import keyword
import hashlib
import json

def canonical(coords:np.ndarray) -> np.ndarray | None:
# returns canonical coordinates (-1 to 1) from input coordinates
    fcoords = coords.astype(np.float64)
    if fcoords.shape[1] != 2:
        print(f"error: Incorrect coord shape")
        return None

    def trans(old: np.ndarray) -> np.ndarray:
        low = np.min(old)
        high = np.max(old)
        new = np.zeros_like(old, dtype=np.float64)
        if low == high:
            new[:] = 0.0
        else:
            # transform to 0 to 1 range then to -1 to 1 range
            new = (old - low) / (high - low)
            new = 2*new - 1
        return new

    canonical_coords = np.zeros_like(fcoords)
    canonical_coords[:,0] = trans(fcoords[:,0])
    canonical_coords[:,1] = trans(fcoords[:,1])
    return canonical_coords

def coherence(dis: Union[np.ndarray, pd.DataFrame], maxprx: float) -> Optional[float]:
    """
    Computes the coherence coefficient between direct and indirect distances.

    Args:
        dis: The pairwise distance matrix.
        maxprx: The maximum allowable distance proxy value.

    Returns:
        The coherence coefficient.
    """
    if isinstance(dis, pd.DataFrame):
        dis = dis.values.astype(float)
    elif dis.dtype != float:
        dis = dis.astype(float)

    dis = np.minimum(dis, dis.T)

    if np.isinf(dis).any():
        infinite_mask = np.isinf(dis)
        dis[infinite_mask] = maxprx*1.1

    np.fill_diagonal(dis, np.nan)
    indirect = pwcorr(dis)
    indirect[np.isnan(indirect)] = 0

    dis_lower = get_lower(dis, diag=-1)
    indirect_lower = get_lower(indirect, diag=-1)

    if dis_lower.size > 0 and indirect_lower.size > 0:
        valid_indices = ~np.isnan(dis_lower) & ~np.isnan(indirect_lower)
        if np.any(valid_indices):
            with np.errstate(invalid='ignore'):
                coh = np.corrcoef(dis_lower[valid_indices], indirect_lower[valid_indices])
                coh = -coh[0,1] if coh.shape == (2, 2) else np.nan
                coh = float(coh)
                return np.round(coh, 3)
        else:
            return np.nan
    else:
        return np.nan

def compute_merged_net_ranks(
    adj_weights: np.ndarray, links: np.ndarray, directed: bool = False
) -> tuple[dict[tuple[int, int], int], list[int]]:
  """Computes link ranks for merged networks based on population count (bit_count).

  Rank 1 corresponds to the most 1-bits (edges present in the most networks),
  descending to the least 1-bits.

  Parameters
  ----------
  adj_weights : np.ndarray (int)
      Adjacency matrix containing the bit-coded integer link weights.
  links : np.ndarray (bool)
      Boolean adjacency matrix indicating active links.
  directed : bool, default=False
      If False, only upper triangle (row < col) links are considered.

  Returns
  -------
  link_ranks : dict[tuple[int, int], int]
      Dictionary mapping (row, col) coordinates to dense rank (1, 2, ...).
  unique_bit_counts : list[int]
      Sorted list of distinct bit counts in descending order (e.g., [4, 3, 1]).
  """
  # Respect network directedness to avoid duplicate undirected edges
  if not directed:
    mask = links & np.triu(np.ones_like(links, dtype=bool), k=1)
  else:
    mask = links

  rows, cols = np.where(mask)
  if len(rows) == 0:
    return {}, []

  # Extract integer bitmask weights for all active links
  edge_weights = adj_weights[rows, cols].astype(np.int64)

  # Calculate bit counts (number of 1s) for each edge
  bit_counts = np.array([int(w).bit_count() for w in edge_weights])

  # Get unique bit counts sorted descending: highest count first
  # Using -bit_counts allows np.unique to sort in descending order
  neg_unique, inverse_indices = np.unique(-bit_counts, return_inverse=True)
  unique_bit_counts = (-neg_unique).tolist()

  # Map (row, col) -> rank (1-indexed, dense ranking)
  link_ranks = {
      (int(r), int(c)): int(rank + 1)
      for r, c, rank in zip(rows, cols, inverse_indices)
  }

  return link_ranks, unique_bit_counts

def dijkstra(dis: np.ndarray, q: int, r: float) -> np.ndarray:
    """Minimum distances over paths of at most q links (Minkowski-r path length).

    Same results as the loop version: each pass extends paths by one link,
    using the previous pass's matrix (m) and the direct distances (dis).
    """
    dis = np.array(dis, dtype=float)
    n = len(dis)
    mindis = dis.copy()
    pass_count = 1
    while pass_count < q:
        pass_count += 1
        m = mindis.copy()
        for k in range(n):
            # direct link row->k, then best (pass-1)-link path k->col
            np.minimum(mindis, minkowski(dis[:, k, None], m[None, k, :], r), out=mindis)
            # best (pass-1)-link path row->k, then direct link k->col
            np.minimum(mindis, minkowski(m[:, k, None], dis[None, k, :], r), out=mindis)
        if np.array_equal(m, mindis):  # converged: no change this pass
            break
    return mindis

def discorr(dis1: np.ndarray, dis2: np.ndarray)->float:
    """
    Computes the correlation between two distance matrices. If both matrices are
    symmetric, only the lower triangular values are considered. Otherwise, the
    off-diagonal elements are used. This helps in determining whether two distance
    matrices have similar structures by computing their correlation.

    :param dis1: The first distance matrix represented as a NumPy ndarray.
    :param dis2: The second distance matrix represented as a NumPy ndarray.
    :return: Correlation coefficient between the two distance matrices. Returns
        NaN if the correlation coefficient cannot be computed.
    :raises ValueError: When the input distance matrices differ in size or are
        empty.
    """
    if dis1.shape != dis2.shape:
        return np.nan
    elif dis1.size == 0:
       return np.nan

    if np.array_equal(dis1, dis1.T) and np.array_equal(dis2, dis2.T):
        # both symmetric: use half-matrix
        dis1 = get_lower(dis1, -1)
        dis2 = get_lower(dis2, -1)
    else:
        # use off-diagonal
        off_diag_mask = ~np.eye(dis1.shape[0], dtype=bool)
        dis1 = dis1[off_diag_mask]
        dis2 = dis2[off_diag_mask]
    pair = np.vstack((dis1, dis2))
    pair = pair.T
    cor = pwcorr(pair)
    return cor[0,1] if cor.shape[0] == 2 else np.nan

def extended_jaccard(matrix_a, matrix_b):
    """
    Computes the Extended Jaccard (Tanimoto) similarity score [0, 1]
    for two pfnet distance matrices. Zeros indicate missing links.
    """
    A = np.array(matrix_a, dtype=float)
    B = np.array(matrix_b, dtype=float)

    # 1. Identify missing links (zeros, NaNs, or Infinities)
    missing_a = (A == 0.0) | np.isnan(A) | np.isinf(A)
    missing_b = (B == 0.0) | np.isnan(B) | np.isinf(B)

    # 2. Extract only valid distances to find global scale boundary
    valid_values = np.concatenate([A[~missing_a], B[~missing_b]])
    if len(valid_values) == 0:
        return 0.0  # Safe boundary fallback for completely empty networks

    d_max = np.max(valid_values)

    # 3. Transform distances to normalized proximities [0, 1]
    prox_a = np.where(missing_a, 0.0, 1.0 - (A / d_max))
    prox_b = np.where(missing_b, 0.0, 1.0 - (B / d_max))

    # 4. Compute Extended Jaccard similarity via minimum/maximum sums
    numerator = np.minimum(prox_a, prox_b).sum()
    denominator = np.maximum(prox_a, prox_b).sum()

    return numerator / denominator if denominator != 0 else 0.0

def floyd(dismat, r):
    m = np.array(dismat, dtype=float)
    for k in range(len(m)):
        a, b = m[:, k, None], m[None, k, :]
        if r == 1:          ind = a + b
        elif np.isinf(r):   ind = np.maximum(a, b)
        else:               ind = np.round((a**r + b**r)**(1/r), 12)
        np.minimum(m, ind, out=m)
    return m

def gen_var_name(proposed:str, exclude:list[str]=None) -> str:
    """
    Generates a valid and unique variable name based on a proposed name. It ensures the
    generated name adheres to identifier rules in Python and avoids conflicts with existing
    names provided in the list. The function will automatically replace invalid characters,
    prepend an underscore if the name starts with a digit, and append an underscore if the
    name conflicts with a Python keyword or an existing name.

    :param proposed: The proposed name for the variable.
    :type proposed: str
    :param exclude: A list of existing variable names to check against to ensure uniqueness.
                     Defaults to None if uniqueness is not required.
    :type exclude: list[str], optional
    :return: A valid and unique Python variable name derived from the proposed name.
    :rtype: str
    """
    legal = re.sub(r'[^a-zA-Z0-9_]', '_', proposed)
    if legal[0].isdigit():
        legal = "v_" + legal
    if keyword.iskeyword(legal):
        legal = legal + "_"
    count = 0
    if exclude is not None:
        while legal in exclude:
            legal = legal + str(count)
            count += 1
    return legal

def get_link_ranks(
    dismat: np.ndarray, links: np.ndarray, directed: bool = False
) -> tuple[dict[tuple[int, int], int], list[float]]:
  """Constructs a sparse link rank dictionary and unique distance list.

  Parameters
  ----------
  dismat : np.ndarray
      Distance / dissimilarity matrix.
  links : np.ndarray (bool)
      Boolean adjacency matrix indicating active links.
  directed : bool, default=False
      If False, only the upper triangle (row < col) is indexed.

  Returns
  -------
  link_ranks : dict[tuple[int, int], int]
      Dictionary mapping (row, col) to dense rank (1, 2, ...).
  unique_distances : list[float]
      Sorted list of distinct distance values across active links.
  """
  # For undirected networks, keep only the upper triangle to avoid duplicates
  if not directed:
    mask = links & (np.triu(np.ones_like(links, dtype=bool), k=1))
  else:
    mask = links

  rows, cols = np.where(mask)

  if len(rows) == 0:
    return {}, []

  # Extract distances, find unique values, and compute dense ranks
  distances = dismat[rows, cols]
  unique_dists, inverse_indices = np.unique(distances, return_inverse=True)

  # Build the (row, col) -> rank mapping (1-indexed)
  link_ranks = {
      (int(r), int(c)): int(rank + 1)
      for r, c, rank in zip(rows, cols, inverse_indices)
  }

  return link_ranks, unique_dists.tolist()

def get_lower(m: np.ndarray, diag: int = -1) -> np.ndarray:
    """
    Extracts the lower triangle of a matrix.
    Args:
        m: The input square matrix.
        diag: The diagonal offset.
    Returns:
        A 1D numpy array containing the elements of the lower triangle.
    """
    rows, cols = np.tril_indices_from(m, k=diag)
    return m[rows, cols]

def get_off_diagonal(m: np.ndarray) -> np.ndarray:
    """
    Extracts the off-diagonal elements of a matrix.
    """
    n = m.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return m[mask]

def get_parent_dir()->str:
    """
    Determine the parent directory of the current file.
    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(this_dir)
    return parent_dir

def get_terms(name: str, n: int, fdir: str) -> list[str]:
    possible = [f"{name}.trm.txt", "terms.txt"]
    for file in possible:
        fpath = os.path.join(fdir, file)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                terms = [cleaned for line in f if (cleaned := line.strip())]
            if len(terms) == n:
                return terms
        except (OSError, UnicodeDecodeError):
            continue
    # Fallback to index strings
    return [str(i) for i in range(n)]

def get_termsid(terms: list[str]) -> str | None:
    n = len(terms)
    if n < 2:
        return None
    first = terms[0].strip().replace(" ", "_")[:5]
    last = terms[-1].strip().replace(" ", "_")[:5]
    serialized = json.dumps(terms, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:6]
    # Structure: [count]_[first]...[last]_[short-hash]
    return f"{n}_{first}.{last}{digest}"

def get_unique_name(base_name: str, existing_names: list[str]) -> str:
  base_name = base_name.strip() or 'default'
  if base_name not in existing_names:
    return base_name

  counter = 2
  candidate = f"{base_name}_{counter}"
  while candidate in existing_names:
    counter += 1
    candidate = f"{base_name}_{counter}"
  return candidate

def graph_from_adjmat(adjmat:np.ndarray, terms:list[str])->nx.Graph | nx.DiGraph:
    n = len(terms)
    isdirected = not np.array_equal(adjmat, adjmat.T)
    if isdirected:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    for i, term in enumerate(terms):
        graph.add_node(term)
    if isdirected:
        for i in range(n):
            for j in range(n):
                if adjmat[i, j] > 0:
                    graph.add_edge(terms[i], terms[j], weight=adjmat[i,j])
    else:
        for i in range(n-1):
            for j in range(i+1, n):
                if adjmat[i,j] > 0:
                    graph.add_edge(terms[i], terms[j], weight=adjmat[i,j])
    return graph

def map_indices_to_terms(list_of_index_lists: list[list[int]], terms: list[str]) -> list[list[str]]:
    """
    Converts a list of lists of indices to a list of lists of terms.

    Args:
        list_of_index_lists: A list where each element is a list of integer indices.
        terms: A list of strings representing the terms.

    Returns:
        A list of lists, where each inner list contains the terms corresponding to the indices.

    Raises:
        IndexError: If any index is out of bounds for the `terms` list.
    """

    result_list = []
    for index_list in list_of_index_lists:
        term_list = []
        for index in index_list:
            try:
                term = terms[index]
                term_list.append(term)
            except IndexError:
                raise IndexError(f"Index {index} is out of bounds for terms list of length {len(terms)}")
        result_list.append(term_list)
    return result_list

def mds_coord(dismat:np.ndarray, ndims=2, metric=True)->np.ndarray:
    from sklearn.manifold import MDS
    mds = MDS(n_components=ndims, init='random', metric_mds=metric, n_init=1,
              max_iter=1000, eps=1e-6, metric='precomputed')
    coord = mds.fit_transform(dismat)
    coord = canonical(coord)
    return coord

def minkowski(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Element-wise Minkowski combination of two path lengths (broadcasts)."""
    if r == 1:
        return a + b
    if np.isinf(r):
        return np.maximum(a, b)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.round((a**r + b**r) ** (1.0 / r), 12)

def mypfnets_dir():
    """
    Returns the path to the user's mypfnets directory and
    creates it if it does not already exist.
    """
    # Identify the user's home directory
    home = os.path.expanduser("~")
    base_dir = os.path.join(home, "mypfnets")

    # Create the directory (and any necessary parent folders)
    # exist_ok=True means it won't throw an error if the folder exists
    os.makedirs(base_dir, exist_ok=True)

    return base_dir

def netsim(adj1:np.ndarray, adj2:np.ndarray) -> dict[str, float | int | None]:
    """
    Calculates network similarity between two adjacency matrices.

    Args:
        adj1 (numpy.ndarray): The first adjacency matrix.
        adj2 (numpy.ndarray): The second adjacency matrix.

    Returns:
        dict: A dictionary containing various similarity measures.
              Returns None if the input matrices are incompatible.
    """

    if adj1.shape != adj2.shape:
        return None

    adj1 = adj1.astype(bool)
    adj2 = adj2.astype(bool)
    n = adj1.shape[0]

    is_symmetric = np.array_equal(adj1, adj1.T) and np.array_equal(adj2, adj2.T)
    mask = ~np.eye(n, dtype=bool)

    if is_symmetric:
        mask = np.triu(mask)

    # Find linear indices of links within the masked matrix
    links1_flat_indices = np.where(adj1[mask].flatten())[0]
    links2_flat_indices = np.where(adj2[mask].flatten())[0]

    n1 = len(links1_flat_indices)
    n2 = len(links2_flat_indices)

    # Find the number of common links
    common = len(np.intersect1d(links1_flat_indices, links2_flat_indices))
    union = len(np.union1d(links1_flat_indices, links2_flat_indices))

    sim = round(common / union, 4) if union > 0 else np.nan

    # Hypergeometric test: Calculate the MEAN directly
    if is_symmetric:
        possible = int(n * (n - 1) / 2)
    else:
        possible = int(n * (n - 1))

    if possible > 0:
        mean = (n1 * n2) / possible
        mean = round(mean, 4)
        p = n2/possible
        variance = (n1*p*(1-p)*(possible-n1)) / (possible-1)
        variance = round(variance, 4)
    else:
        mean = np.nan
        variance = np.nan

    adjcommon = common - mean if not np.isnan(mean) else np.nan
    adjsim = round(adjcommon / union, 4) if union > 0 and not np.isnan(adjcommon) else np.nan

    return {
        "possible": possible,
        "links1": n1,
        "links2": n2,
        "common": common,
        "union": union,
        "similarity": sim,
        "mean": mean,
        "variance": variance,
        "adjcommon": adjcommon,
        "adjsimilarity": adjsim,
    }

def map_coords_to_frame(canonicalxy: np.ndarray, width=600, height=600,
                        squeezex=0.0, squeezey=0.0) -> np.ndarray:
    """
    Transforms an ndarray of canonical 2D coordinates to fit within a defined
    width and height, applying squeeze factors to adjust the scaling of the
    transformation.
    :param squeezey: shrink or expand the y-coordinates
    :param squeezex: shrink or expand the x-coordinates
    :param canonicalxy: An ndarray of canonical 2D coordinates (x, y).
    :param width: The width of the bounding box representing the target area for
        the transformed x-coordinates.
    :param height: The height of the bounding box representing the target area for
        the transformed y-coordinates.
    :params squeezex and squeeze y: Scaling factors applied to the dimensions during transformation,
        ranging from -1 (no squeeze) to 1 (maximum squeeze).  Positive values shrink, negative values expand.
    :return: A ndarray of transformed coordinates
    """
    # get new coordinates from old coordinates
    half_width = width / 2
    half_height = height / 2
    newx = (canonicalxy[:,0] * (1 - squeezex)) * half_width + half_width
    newy = (canonicalxy[:,1] * (1 - squeezey)) * half_height + half_height
    newxy = np.column_stack((newx, newy))
    newxy = np.round(newxy, 3)
    return newxy

def prop_table(objs: list, props: list) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame where rows are objects and columns are their properties
    Args:
        objs (list): A list of objects.
        props (list): A list of strings, where each string is the name of a property.
    Returns:
        pd.DataFrame: A DataFrame with object properties.
    """
    data = []
    for obj in objs:
        row_data = [getattr(obj, prp, None) for prp in props] # None for missing attrs
        data.append(row_data)
    df = pd.DataFrame(data, columns=props)
    return df

def pwcorr(dis:np.ndarray) -> np.ndarray:
    """
    Calculates the pairwise correlation coefficient between columns in a given
    2D numpy array.

    Args:
        dis: A 2D numpy array where each column represents a variable.

    Returns:
        A 2D numpy array of pairwise correlation coefficients.
    """
    num_cols = dis.shape[1]
    corrs = np.zeros((num_cols, num_cols))
    with np.errstate(invalid='ignore'):
        for i in range(num_cols):
            for j in range(num_cols):
                col1 = dis[:, i]
                col2 = dis[:, j]
                valid_indices = ~np.isnan(col1) & ~np.isnan(col2) & ~np.isinf(col1) & ~np.isinf(col2)
                if np.sum(valid_indices) > 1:
                    corr = np.corrcoef(col1[valid_indices], col2[valid_indices])
                    if corr.shape == (2, 2):
                        corrs[i, j] = corr[0, 1]
                    else:
                        corrs[i, j] = np.nan
                else:
                    corrs[i, j] = np.nan
        return np.round(corrs, 3)

def sanitize_for_json(obj):
    """Recursively converts NumPy types into native Python types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def sig_figs_flat(x, n):
    if x == 0:
        return 0

    # Calculate the power of 10 for the first significant digit
    mag = math.floor(math.log10(abs(x)))
    if mag > n-1:
        return int(round(x,0))

    # Calculate how many decimal places to keep
    decimals = n - 1 - mag

    # Rounding to a negative number rounds to the left of the decimal
    rounded = round(x, decimals)

    # Convert to int if it's a whole number to get rid of the .0
    if rounded == int(rounded):
        return int(rounded)

    return rounded

def sorted_keys(dic: dict, attrs: tuple | list = ("termsid", "name")) -> list:
    """
    Sorts dictionary keys by value attributes.
    - Default: sorts by ("termsid", "name").
    - attrs == [] or (): sorts by dictionary keys directly.
    - custom attrs: sorts by the provided attribute sequence.
    """
    if not attrs:
        return sorted(dic.keys())

    return sorted(
        dic.keys(),
        key=lambda k: tuple(str(getattr(dic[k], attr, "")) for attr in attrs)
    )

def split_long_term(term:str)->str:
    n = len(term)
    if n < 15:
        return term
    sep = [' ','_','-', '.',',']
    mid = n/2
    loc = []
    for i in range(n):
        if term[i] in sep:
            loc.append(i)
    if loc:
        mind = n + 1
        minp = loc[0]
        for pos in loc:
            dis = abs(pos-mid)
            if dis < mind:
                mind = dis
                minp = pos
        return term[0:minp] + '\n' + term[minp+1:n]
    else:
        if n > 20:
            mid = round(n/2)
            return term[0:mid] + '-\n' + term[mid:n]
        else:
            return term

if __name__ == '__main__':
    from pypf.proximity import Proximity
    from pypf.collection import Collection
    tcol = Collection()
    psyprx = Proximity(os.path.join("data", "psy.prx.xlsx"))
    tcol.add_proximity(psyprx, add_pfnet=True)
    bioprx = Proximity(os.path.join("data", "bio.prx.xlsx"))
    tcol.add_proximity(bioprx, add_pfnet=True)
    dismat = psyprx.dismat
    links = tcol.pfnets["psy_pf"].adjmat
    links = links.astype(bool)
    # unique_distances, link_rank = compute_link_ranks(dismat, links)
    # print(unique_distances)
    # print(link_rank)
    u_d, linklist = get_link_ranks(dismat, links, False)
    print("List format")
    print(u_d)
    print(linklist)


