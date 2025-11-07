# pypf/utility.py
# df = pd.DataFrame(matrix, index=rows, columns=col) - creates table with row and col labels
import os
from typing import Optional, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as nx
import re
import keyword
import shutil
import inspect # Used to find the path of the current module

def average_proximity(proxes:list, method="mean"):
    """
    Computes the average proximity matrix based on a list of proximity matrices and a specified method
    (mean or median). The function generates a new proximity object as a result, encapsulating the
    aggregated proximity data.

    :param proxes: A list of Proximity objects for which the average proximity matrix is to be computed.
                   All Proximity objects in the list must have matching terms and dimensions.
    :type proxes: list[Proximity]

    :param method: The method to use for computing the average proximity. Accepted values are "mean"
                   and "median". Defaults to "mean". The "mean" method computes the arithmetic mean
                   across matrices, while the "median" method computes the median element-wise.
    :type method: str, optional

    :return: A new Proximity object containing the aggregated proximity matrix as per the specified
             method. The resulting proximity includes updated statistics and coherence measurements.
    :rtype: Proximity
    """
    from pypf.proximity import Proximity
    import copy
    numprx = len(proxes)
    if numprx < 2:
        return None
    ave_prx:Proximity = copy.deepcopy(proxes[0])
    ave_prx.filename = None
    ave_prx.filepath = None
    ave_prx.name = method + "_" + numprx.__str__() + "_" + ave_prx.name + "_" + proxes[numprx-1].name
    for prx in proxes:
        if prx.terms != ave_prx.terms:
            return None
    nterms = ave_prx.nterms
    # set ave to square matrix of zeros
    ave = np.zeros((nterms,nterms))
    match method:
        case "mean":
            for prx in proxes:
                ave = ave + prx.dismat
            ave = ave / numprx
        case "median":
            for i in range(nterms):
                for j in range(nterms):
                    ave[i,j] = np.median([prx.dismat[i,j] for prx in proxes])
    issymmetric = np.equal(ave,ave.T).all()
    ave_prx.dismat = ave
    ave_prx.issymmetric = issymmetric
    ave_prx.calculate_stats()
    ave_prx.coh = coherence(ave_prx.dismat, ave_prx.max)
    return ave_prx

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
            with np.errstate(invalid='ignore'):
                coh = np.corrcoef(dis_lower[valid_indices], indirect_lower[valid_indices])
                coh = -coh[0,1] if coh.shape == (2, 2) else np.nan
                return np.round(coh, 3)
        else:
            return np.nan
    else:
        return np.nan

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
    dist_noinf = dist_matrix.copy()
    dist_noinf[np.isinf(dist_noinf)] = 0
    ecc = np.max(dist_noinf, axis=1)
    radius = np.min(ecc[ecc>0])
    diameter = np.max(ecc)
    center = np.where(ecc == radius)[0].tolist()
    ecc = ecc.astype(int).tolist()
    radius = int(radius.astype(int))
    diameter = int(diameter.astype(int))

    return {
        'ecc': ecc,
        'radius': radius,
        'diameter': diameter,
        'center': center
    }

def file_format():
    tb6 = ["account","bank","flood","money","river", "save"]
    bank6 = pd.DataFrame(np.array([[0, 3,32,26,32,32],[3,0,27,21,14,30],
                                           [32,27,0,31,23,29],[26,21,31,0,31,23],
                                           [32,14,23,31,0,31],[32,30,29,23,31,0]]),
                                 index=tb6, columns=tb6)
    return bank6

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

def get_parent_dir():
    """
    Determine the parent directory of the current file.
    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    proj_dir = os.path.dirname(this_dir)
    return proj_dir

def get_termsid(terms:list):
    n = len(terms)
    if n < 2:
        return None
    termsid = "n_" + str(int(n)) + terms[0] + terms[1] + terms[-1]
    termsid = re.sub(r'[^a-zA-Z0-9_]', '_', termsid)
    return termsid



def get_test_pf(name="psy"):
    from pypf.pfnet import PFnet
    from pypf.proximity import Proximity
    filepath = os.path.join("data", name + ".prx.xlsx")
    prx = Proximity(filepath)
    pf = PFnet(prx)
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

def mds_coord(dismat:np.ndarray, ndims=2, metric=True):
    from sklearn.manifold import MDS
    mds = MDS(n_components=ndims, metric=metric, n_init=1,
              max_iter=1000, eps=1e-6, dissimilarity='precomputed')
    coord = mds.fit_transform(dismat)
    coord = canonical(coord)
    return coord

def merge_networks(nets):
    from pypf.pfnet import PFnet
    import copy
    n = len(nets)
    if n < 2:
        return None
    mrg:PFnet = copy.deepcopy(nets[0])
    mrg.type = "merged"
    # extract adjmat and convert to bool
    adj = nets[0].adjmat
    adj = adj.astype(bool)
    adj = adj.astype(int)
    mrg.name = "Merged_" + mrg.name
    for i in range(1, n):
        if mrg.terms != nets[i].terms:
            return None
        addon = nets[i].adjmat
        addon = addon.astype(bool)
        addon = addon.astype(int)
        # make link values 2^i which gives a binary coding to links in nets
        addon = addon * (2**i)
        adj = adj + addon
        mrg.name += "_" + nets[i].name
    mrg.nlinks = adj.astype(bool).sum()
    # if not mrg.isdirected:  -- apparently adjmat deals with directedness
    #     mrg.nlinks = mrg.nlinks / 2
    mrg.dismat = None
    mrg.mindis = None
    mrg.q = None
    mrg.r = None
    mrg.isdirected = not np.array_equal(adj, adj.T)
    if not mrg.isdirected:
        mrg.nlinks = mrg.nlinks // 2
    mrg.graph = graph_from_adjmat(adj, mrg.terms)
    mrg.graph.name = mrg.name
    mrg.eccentricity = eccentricity(adj)
    mrg.layers = mrg.shells(mrg.eccentricity["center"])
    mrg.adjmat = adj
    return mrg

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

def mypfnets_dir() -> str:
    """
    Determines the user's home directory and whether the directory, 'mypfnets', is present.
    If so, returns the path to the 'mypfnets' directory.
    If the directory does not exist, the directory will be created and the files in the
    'data' directory within the 'pypf' package will be copied into the 'mypfnets' directory.

    Returns:
        str: The absolute path to the 'mypfnets' directory.
    """
    # 1. Determine the user's home directory
    user_home_dir = os.path.expanduser("~")

    # 2. Define the path to the 'mypfnets' directory within the user's home
    mypfnets_path = os.path.join(user_home_dir, "mypfnets")

    # 3. Check if the 'mypfnets' directory exists
    if not os.path.exists(mypfnets_path):
        print(f"Creating user data directory: {mypfnets_path}")
        try:
            # Create the directory
            os.makedirs(mypfnets_path)

            # 4. If the directory does not exist, copy files from pypf/data
            #    Find the path of the current module (utility.py)
            current_module_file = inspect.getfile(mypfnets_dir)
            # Go one level up to get the pypf package root directory
            pypf_package_root = os.path.dirname(current_module_file)
            # Construct the path to the 'data' directory within pypf
            source_data_dir = os.path.join(pypf_package_root, "data")

            if os.path.exists(source_data_dir) and os.path.isdir(source_data_dir):
                print(f"Copying initial data from {source_data_dir} to {mypfnets_path}")
                for item_name in os.listdir(source_data_dir):
                    source_item_path = os.path.join(source_data_dir, item_name)
                    destination_item_path = os.path.join(mypfnets_path, item_name)

                    if os.path.isfile(source_item_path):
                        shutil.copy2(source_item_path, destination_item_path)
                    elif os.path.isdir(source_item_path):
                        # If there are subdirectories in 'data', copy them recursively
                        shutil.copytree(source_item_path, destination_item_path, dirs_exist_ok=True)
            else:
                print(f"Warning: Source data directory '{source_data_dir}' not found within the pypf package.")

        except OSError as e:
            print(f"Error creating or copying files to {mypfnets_path}: {e}")

    # 5. Return the path to the 'mypfnets' directory
    return mypfnets_path

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

    adj1 = adj1.astype(bool)
    adj2 = adj2.astype(bool)
    n = adj1.shape[0]

    is_symmetric = np.array_equal(adj1, adj1.T) and np.array_equal(adj2, adj2.T)
    #n_possible_links = (n**2 - n) // 2
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

def sample_data() -> dict:
    from pypf.collection import Collection
    from pypf.proximity import Proximity
    from pypf.pfnet import PFnet
    from pypf.netpic import Netpic
    files = ["bank6","bio", "psy", "bank", "cities", "statecaps"]
    srcs = ["coocurrences in Longman's dictionary",
        "relatedness ratings by biology grad students",
         "relatedness ratings by psychology 101 students",
         "coocurrences in Longman's dictionary",
         "distances between cities in the US",
         "distances between state capitals in the US"]
    sources = pd.DataFrame(data={"file": files, "source": srcs})
    col = Collection()
    ex = None
    for file in files:
        fn = os.path.join("pypf","data", file + ".prx.xlsx")
        px = Proximity(fn)
        col.add_proximity(px)
        pf = PFnet(px)
        col.add_pfnet(pf)
        if file in ["bank6"]:
            ex = pf
        if file in ["bio", "psy"]:
            pf = PFnet(px, q=2)
            col.add_pfnet(pf)
        if file in ["statecaps"]:
            pf = PFnet(px, q=2, r=2)
            col.add_pfnet(pf)
    ex.get_layout(method="kamada_kawai")
    dic = {"collection": col, "sources": sources, "net_plot": Netpic(ex)}
    return dic

def split_long_term(term):
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
    from pypf.pfnet import PFnet
    from pypf.proximity import Proximity
    psyprx = Proximity(os.path.join("data", "psy.prx.xlsx"))
    bioprx = Proximity(os.path.join("data", "bio.prx.xlsx"))
    psy = PFnet(psyprx)
    bio = PFnet(bioprx)
    prxset = [psyprx, bioprx]
    netset = [psy, bio]
    # mrg = merge_networks(netset)
    # print(mrg.netprint())
    # aveprx = average_proximity(prxset, method="mean")
    # aveprx.prxprint()
    bio.get_layout(method="MDS")

