#   pypf/utility.py
import os
from typing import Optional, Union
import numpy as np
import pandas as pd

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

def netsim(adj1, adj2):
    """
    Calculates a similarity index between two adjacency matrices.

    Args:
        adj1 (numpy.ndarray): The first adjacency matrix.
        adj2 (numpy.ndarray): The second adjacency matrix.

    Returns:
        dict or float: A dictionary containing similarity details,
                     or NaN if the matrices are not of the same size.
                     The dictionary has the following keys:
                     - 'common':  Indices where common adj1 and adj2 are True.
                     - 'num_common': Number of such indices.
                     - 'net1not2': Indices where adj1 is True and adj2 is False.
                     - 'num_only_net1': Number of such indices.
                     - 'net2not1': Indices where adj2 is True and adj1 is False.
                     - 'num_only_net2': Number of such indices.
                     - 'sim':  The similarity index.
    """

    if adj1.shape != adj2.shape:
        return np.nan

    adj1_bool = adj1.astype(bool)
    adj2_bool = adj2.astype(bool)

    # Find indices where common matrices are True (or non-zero)
    common = np.where(adj1_bool & adj2_bool)[0]  # Using numpy's logical AND

    # Find indices where adj1 is True and adj2 is False
    net1not2 = np.where(adj1_bool & ~adj2_bool)[0]

    # Find indices where adj2 is True and adj1 is False
    net2not1 = np.where(adj2_bool & ~adj1_bool)[0]

    num_common = len(common)
    num_only_net1 = len(net1not2)
    num_only_net2 = len(net2not1)
    num_union = num_common + num_only_net1 + num_only_net2

    sim_index = num_common / num_union if num_union > 0 else np.nan  # Handle division by zero

    return {
        'common': common,
        'num_common': num_common,
        'net1not2': net1not2,
        'num_only_net1': num_only_net1,
        'net2not1': net2not1,
        'num_only_net2': num_only_net2,
        'sim': np.round(sim_index, 3)
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
