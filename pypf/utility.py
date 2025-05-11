#   pypf/utility.py
import numpy as np
import pandas as pd

def get_lower(m=np.array([]), diag=-1):
    rows, cols = np.tril_indices_from(m, k=diag)
    return m[rows, cols]

def pwcorr(dis=np.array([])):
    """
    Calculates the pairwise correlation coefficient between columns in a given
    2D numpy array. The function takes into account NaN values in the input
    array, and only calculates the correlation for the valid (non-NaN) entries.
    If there are less than two valid entries, the corresponding correlation value
    is set to NaN.

    :param dis: A 2D numpy array where each column represents a variable, and
        rows represent observations. Values can include NaNs.
    :type dis: numpy.ndarray
    :return: A 2D numpy array containing pairwise correlation coefficients
        between columns of the input array. For columns where valid
        correlations cannot be computed, NaN is returned.
    :rtype: numpy.ndarray
    """
    num_cols = dis.shape[1]
    corrs = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            col1 = dis[:, i]
            col2 = dis[:, j]
            valid_indices = ~np.isnan(col1) & ~np.isnan(col2)
            if np.sum(valid_indices) > 1:  # Need at least 2 points to calculate correlation
                corr = np.corrcoef(col1[valid_indices], col2[valid_indices])
                if corr.shape == (2, 2):
                    corrs[i, j] = corr[0, 1]
                else:
                    corrs[i, j] = np.nan
            else:
                corrs[i, j] = np.nan
    return corrs

def coherence(dis=np.array([]), maxprx=float):
    """
    Computes the coherence coefficient between the direct and indirect distances
    in a pairwise distance matrix.

    :param dis: The pairwise distance matrix. If provided as a DataFrame, it is
        converted to a NumPy array. Distances must be numeric, and values can
        include `nan` or infinite values.
    :type dis: numpy.ndarray or pandas.DataFrame
    :param maxprx: The maximum allowable distance proxy value, used to replace
        infinite distances in the input distance matrix.
    :type maxprx: float
    :return: The negative coherence coefficient between the direct and indirect
        distances as a float. If the coherence coefficient cannot be calculated,
        `nan` is returned.
    :rtype: float
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
            return -coh[0, 1] if coh.shape == (2, 2) else np.nan
        else:
            return np.nan
    else:
        return np.nan
