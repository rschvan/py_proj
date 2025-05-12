#   pypf/utility.py
import os
from typing import Optional, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
from networkx import nodes

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

    n_links1 = len(links1_flat_indices)
    n_links2 = len(links2_flat_indices)

    # Find the number of common links
    n_both = len(np.intersect1d(links1_flat_indices, links2_flat_indices))
    n_union = len(np.union1d(links1_flat_indices, links2_flat_indices))

    sim = round(n_both / n_union, 4) if n_union > 0 else np.nan

    # Hypergeometric test: Calculate the MEAN directly
    if is_symmetric:
        M = int(n * (n - 1) / 2)
    else:
        M = int(n * (n - 1))

    if M > 0:
        rand_both = (n_links1 * n_links2) / M
        rand_both = round(rand_both, 4)
    else:
        rand_both = np.nan

    cn_both = n_both - rand_both if not np.isnan(rand_both) else np.nan
    c_sim = round(cn_both / n_union, 4) if n_union > 0 and not np.isnan(cn_both) else np.nan

    return {
        "npos": M,
        "nlinks1": n_links1,
        "nlinks2": n_links2,
        "nboth": n_both,
        "nunion": n_union,
        "sim": sim,
        "randboth": rand_both,
        "cnboth": cn_both,
        "csim": c_sim,
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


if __name__ == '__main__':
    from pypf.proximity import Proximity
    from pypf.pfnet import PFnet
    from pypf.display import NetworkDisplay
    from pypf.utility import get_parent_dir
    import os

    psyprx = Proximity(os.path.join(get_parent_dir(), "data", "psy.prx.xlsx"))
    #bioprx = Proximity(os.path.join(get_parent_dir(), "data", "bio.prx.xlsx"))
    psy = PFnet(psyprx)
    #bio = PFnet(bioprx)

    # # Example netsim with weighted adjacency matrices
    # adj_matrix1 = np.array([[1.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    # adj_matrix2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    #
    # parts = netsim(adj_matrix1, adj_matrix2)
    #
    # print("Adjacency Matrix 1:\n", adj_matrix1)
    # print("Adjacency Matrix 2:\n", adj_matrix2)
    # print("Parts:\n", parts)
    # print("similarity index:", parts['sim'])

    # dir = get_parent_dir()
    # print(dir)

    # # Example test for discorr
    # p1 = Proximity(os.path.join(get_parent_dir(), "data", "psy.prx.xlsx"))
    # p2 = Proximity(os.path.join(get_parent_dir(), "data", "bio.prx.xlsx"))
    # tcor = discorr(p1.dismat, p2.dismat)
    # print(f"Proximity correlation: {tcor}")
