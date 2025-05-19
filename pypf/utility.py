# pypf/utility.py
# coherence
# graph_from_adjmat
# dijkstra
# discorr
# eccentricity
# floyd
# get_lower
# get_off_diagonal
# get_parent_dir
# map_indices_to_terms
# minkowski
# netsim
# newcoords
# pwcorr
# get_test_pf
import os
from typing import Optional, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx

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

def newcoords(oldxy: dict, width: float, height: float, squeezex: float, squeezey: float) -> dict:
    """
    Transforms and normalizes a dictionary of 2D coordinates to fit within a defined
    width and height, applying squeeze factors to adjust the scaling of the
    transformation.

    The function uses a helper method to process and transform the x and y
    coordinates separately using numpy arrays to increase efficiency. Each coordinate
    dimension is normalized around its mid-point and then scaled according to the
    given squeeze factor and bounding box size.

    :param oldxy: A dictionary with keys as node identifiers and values as tuples
        representing 2D coordinates (x, y).
    :type oldxy: dict
    :param width: The width of the bounding box representing the target area for
        the transformed x-coordinates.
    :type width: float
    :param height: The height of the bounding box representing the target area for
        the transformed y-coordinates.
    :type height: float
    :param squeezex: A scaling factor applied to the x-dimension during transformation,
        ranging from 0 (no squeeze) to 1 (maximum squeeze).
    :type squeezex: float
    :param squeezey: A scaling factor applied to the y-dimension during transformation,
        ranging from 0 (no squeeze) to 1 (maximum squeeze).
    :type squeezey: float
    :return: A dictionary of transformed coordinates with the same keys as the input
        dictionary and transformed (x, y) tuples as values.
    :rtype: dict

    """
    def _trans(old: np.ndarray, half: float, squeeze: float) -> np.ndarray:
        """
        Helper function to transform a single coordinate dimension (x or y).
        Args:
            old (np.ndarray): A 1D numpy array of the coordinates to transform.
            half (float): Half of the width or height.
            squeeze (float): The squeeze factor for this dimension.
        Returns:
            np.ndarray: The transformed coordinates.
        """
        low = np.min(old)
        high = np.max(old)
        if low == high:
            new = np.zeros_like(old)  # Correct way to create zeros with the same shape
        else:
            mid = (low + high) / 2
            new = (old - mid) / (high - low)
        new = new * (1 - squeeze)
        new = new * half
        new = new + half
        return new

    # Extract coordinates into numpy arrays for efficient processing
    node_ids = list(oldxy.keys())
    oldx = np.array([oldxy[node_id][0] for node_id in node_ids])
    oldy = np.array([oldxy[node_id][1] for node_id in node_ids])

    # Transform x and y coordinates
    newx = _trans(oldx, width / 2, squeezex)
    newy = _trans(oldy, height / 2, squeezey)

    # Rebuild the dictionary with transformed coordinates
    newxy_dict = {node_id: (newx[i], newy[i]) for i, node_id in enumerate(node_ids)}
    return newxy_dict

def get_test_pf(name="psy"):
    from pypf.pfnet import PFnet
    from pypf.proximity import Proximity
    filepath = os.path.join("data", name + ".prx.xlsx")
    psyprx = Proximity(filepath)
    pf = PFnet(psyprx)
    return pf

def graph_from_adjmat(adjmat:npt.NDArray, terms:list):
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
                if adjmat > 0:
                    graph.add_edge(terms[i], terms[j], weight=adjmat[i,j])
    else:
        for i in range(n-1):
            for j in range(i+1, n):
                if adjmat[i,j] > 0:
                    graph.add_edge(terms[i], terms[j])
    return graph

def floyd(dismat, r):
    """
    Computes the shortest paths in a weighted graph using the Floyd-Warshall algorithm
    and applies the Minkowski distance-based interpolation to update the path weights.

    The function takes as input a distance matrix `dismat` and a Minkowski parameter `r`.
    The `dismat` parameter represents the adjacency matrix of the graph, where each element
    is the weight of the edge between vertices. The function applies an iterative approach
    to compute the shortest paths between nodes, updating the distance matrix at each step.

    :param dismat:
        A 2D array or list containing the distances between nodes in the graph. The
        dimensions of this matrix should be NxN, where N is the number of nodes.
        The element at position (i, j) represents the weight of the edge from node
        i to node j.

    :param r:
        A numeric value representing the Minkowski power parameter. This is used
        in the Minkowski distance calculation to adjust the behavior of path weight
        comparisons and updates.

    :return:
        A 2D numpy array representing the shortest distance between every pair of
        nodes in the graph after the algorithm has been applied. The dimensions
        of the array will match the input `dismat`.
    """
    dismat = np.array(dismat, dtype=float)
    n = len(dismat)
    mindis = np.copy(dismat)
    for ind in range(n):
        for row in range(n):
            for col in range(n):
                indirect = minkowski(mindis[row, ind], mindis[ind, col], r)
                if indirect < mindis[row, col]:
                    mindis[row, col] = indirect
    return mindis

def dijkstra(dis, q, r):
    dis = np.array(dis, dtype=float)
    n = len(dis)
    mindis = np.copy(dis)
    pass_count = 1
    changed = True
    while changed and (pass_count < q):
        pass_count += 1
        changed = False
        m = np.copy(mindis)
        for row in range(n):
            for col in range(n):
                for ind in range(n):
                    indirect1 = minkowski(dis[row, ind], m[ind, col], r)
                    indirect2 = minkowski(m[row, ind], dis[ind, col], r)
                    indirect = min(indirect1, indirect2)
                    if indirect < mindis[row, col]:
                        mindis[row, col] = indirect
                        changed = True
        # Check for convergence (no changes in this pass)
        if np.array_equal(m, mindis):
            changed = False
    return mindis

def minkowski(dis1, dis2, r):
    if r == 1:
        dis = dis1 + dis2
    elif np.isinf(r):
        dis = max(dis1, dis2)
    else:
        dis = (dis1**r + dis2**r)**(1/r)
        # Remove rounding errors
        dis = round(dis, 12)
    return dis

def get_parent_dir():
    """
    Determine the parent directory of the current file.
    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    proj_dir = os.path.dirname(this_dir)
    return proj_dir

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

def pwcorr(dis: np.ndarray = np.array([])) -> np.ndarray:
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
    for i in range(num_cols):
        for j in range(num_cols):
            col1 = dis[:, i]
            col2 = dis[:, j]
            valid_indices = ~np.isnan(col1) & ~np.isnan(col2)
            if np.sum(valid_indices) > 1:
                corr = np.corrcoef(col1[valid_indices], col2[valid_indices])
                if corr.shape == (2, 2):
                    corrs[i, j] = corr[0, 1]
                else:
                    corrs[i, j] = np.nan
            else:
                corrs[i, j] = np.nan
    return np.round(corrs, 3)

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
        finite_distances = dis[~infinite_mask]
        if finite_distances.size > 0:
            sorted_finite = np.sort(finite_distances)
            max_finite = sorted_finite[-1]
            finite_before_max = sorted_finite[-2] if sorted_finite.size > 1 else max_finite
            increment = max_finite - finite_before_max
            if increment == 0:
                increment = 1
            dis[infinite_mask] = max_finite + increment
        else:
            dis[infinite_mask] = maxprx + 1

    np.fill_diagonal(dis, np.nan)
    indirect = pwcorr(dis)
    indirect[np.isnan(indirect)] = 0

    dis_lower = get_lower(dis, diag=-1)
    indirect_lower = get_lower(indirect, diag=-1)

    if dis_lower.size > 0 and indirect_lower.size > 0:
        valid_indices = ~np.isnan(dis_lower) & ~np.isnan(indirect_lower)
        if np.any(valid_indices):
            coh = np.corrcoef(dis_lower[valid_indices], indirect_lower[valid_indices])
            coh = -coh[0,1] if coh.shape == (2, 2) else np.nan
            return np.round(coh, 3)
        else:
            return np.nan
    else:
        return np.nan

def netsim(adj1: npt.NDArray, adj2: npt.NDArray) -> dict:
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
        print("Incompatible nets")
        return None

    adj1 = adj1.astype(bool)
    adj2 = adj2.astype(bool)
    n = adj1.shape[0]

    is_symmetric = np.array_equal(adj1, adj1.T) and np.array_equal(adj2, adj2.T)
    n_possible_links = (n**2 - n) // 2
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

def discorr(dis1: np.ndarray, dis2: np.ndarray):
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
        raise ValueError("Proximities differ in size")
    elif dis1.size == 0:
        raise ValueError("Invalid Data")

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

def eccentricity(adjmat: np.ndarray) -> dict:
    """
    Calculates the eccentricity, radius, diameter, and center of a graph
    represented by an adjacency matrix.

    Args:
        adjmat (numpy.ndarray): A square adjacency matrix.

    Returns:
        dict: A dictionary containing the eccentricity, radius, diameter, and center.
              - 'ecc': A numpy array of eccentricities for each node.
              - 'radius': The radius of the graph.
              - 'diameter': The diameter of the graph.
              - 'center': A numpy array of the indices (0-based) of the center nodes.
    """
    n = adjmat.shape[0]
    adjmat = np.maximum(adjmat, adjmat.T)
    links = adjmat.astype(bool)
    dis = np.full((n, n), np.inf)
    dis[links] = 1
    np.fill_diagonal(dis, 0)

    dist_matrix = floyd(dis, 1)

    ecc = np.max(dist_matrix, axis=1)
    radius = np.min(ecc)
    diameter = np.max(ecc)
    center = np.where(ecc == radius)[0]

    return {
        'ecc': ecc,
        'radius': radius,
        'diameter': diameter,
        'center': center
    }

if __name__ == '__main__':
    from pypf.pfnet import PFnet
    from pypf.proximity import Proximity
    psyprx = Proximity(os.path.join("data", "psy.prx.xlsx"))
    bioprx = Proximity(os.path.join("data", "bio.prx.xlsx"))
    psy = PFnet(psyprx)
    bio = PFnet(bioprx)
    tl = map_indices_to_terms(psy.layers, psy.terms)
    print(psy.layers)
    print(tl)
