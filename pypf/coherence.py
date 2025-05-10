##pypf/coherence.py
import numpy as np
import pandas as pd

def get_lower(m, diag=-1):
    rows, cols = np.tril_indices_from(m, k=diag)
    return m[rows, cols]

def coherence(dis, maxprx):
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

    num_cols = dis.shape[1]
    indirect = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            col1 = dis[:, i]
            col2 = dis[:, j]
            valid_indices = ~np.isnan(col1) & ~np.isnan(col2)
            if np.sum(valid_indices) > 1:  # Need at least 2 points to calculate correlation
                corr = np.corrcoef(col1[valid_indices], col2[valid_indices])
                if corr.shape == (2, 2):
                    indirect[i, j] = corr[0, 1]
                else:
                    indirect[i, j] = np.nan
            else:
                indirect[i, j] = np.nan

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