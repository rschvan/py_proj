# pypf/pfnet.py
import numpy as np
from pypf.proximity import Proximity
from pypf.utility import (floyd, dijkstra, graph_from_adjmat, get_lower,
                          get_off_diagonal, get_termsid, get_link_ranks,
                          )
import networkx as nx

_INFINITE_WORDS = ("", "inf", "infinity", "none", "∞")


def parse_q(text) -> float:
    """Reads a q value typed by a user: a whole number of at least 1, or infinity.
    Blank, "inf" or anything unreadable means infinity; values below 1 become 1."""
    text = str(text).strip().lower()
    if text in _INFINITE_WORDS:
        return np.inf
    try:
        value = float(text)
    except ValueError:
        return np.inf
    if np.isinf(value) or np.isnan(value):
        return np.inf
    return max(1, int(value))


def parse_r(text) -> float:
    """Reads an r value typed by a user: a number of at least 1, or infinity.
    Blank, "inf" or anything unreadable means infinity; values below 1 become 1."""
    text = str(text).strip().lower()
    if text in _INFINITE_WORDS:
        return np.inf
    try:
        value = float(text)
    except ValueError:
        return np.inf
    if np.isnan(value):
        return np.inf
    return max(1.0, value)


class PFnet:
    """
    PFnet class is used for creating and managing Proximity File Networks, which represent
    graph structures with n_nodes, n_edges, and distances. It is capable of handling weighted
    graphs, both directed and undirected, and generates a pfnet(q,r).

    The class is initialized using a proximity object containing terms and a distance matrix,
    along with other optional parameters for controlling graph behavior. It supports multiple
    algorithms for computing minimum distances between n_nodes (Floyd's algorithm and Dijkstra's).
"""
    def __init__(self, proximity: Proximity, q=np.inf, r=np.inf, type="pf"):
        """
        Creates a PFnet object.
        :param proximity: Object containing graph data, specifically a list of terms and a distance matrix.
        :param q: Optional parameter to control the subset of n_nodes considered for minimum
                  distance computations. Defaults to `np.inf`.
        :param r: Optional parameter to control Minkowski distance calculation. Defaults to `np.inf`.
        :param type: Optional parameter to control graph type. Defaults to "pf".

        """
        self.q = q # Inf yields nterms-1
        self.r = r
        #self.infchar = "\u221E"
        self.terms:list[str] = []  # term list


        self.termsid:str = ""
        self.nnodes:int = 0
        self.nlinks:int = 0
        self.dismat = np.array([])  # distance matrix
        self.mindis = np.array([])  # minimum distances
        self.adjmat = np.array([])  # adjacency matrix with link weights
        self.links: dict[tuple[int, int], int] = {}  # (source index, target index) -> rank
        self.isconnected = True
        self.isdirected = False  # true for directed graph
        self.name:str = "default"  # Initialize the name property
        self.graph: nx.Graph | nx.DiGraph = nx.Graph()  # replaced once links are known
        self.is_planar:bool
        self.layers = None  # nodes in each shell
        self.eccentricity:dict = {}
        self.type:str = type
        self.layout_method:str = "auto"
        self.coords:np.ndarray | None = None
        self.unique_weights:np.ndarray | None = None
        self.sources:list[str] = []

        if not (hasattr(proximity, 'terms') and hasattr(proximity, 'dismat')):
            raise TypeError("PFnet needs a Proximity (an object with terms and dismat)")
        if type not in ("pf", "nn"):
            raise ValueError(f"PFnet type must be 'pf' or 'nn', not {type!r}")
        self.name = proximity.name
        self.proximity = proximity
        self.terms = proximity.terms
        self.nnodes = len(self.terms)
        self.dismat = np.array(proximity.dismat)
        self.termsid = proximity.termsid
        if self.termsid == "":
            self.termsid = get_termsid(self.terms)

        links = np.full((self.nnodes, self.nnodes), False, dtype=bool)

        # noinspection PyUnreachableCode
        match self.type:
            case "pf":
                self.name = self.name + "_pf"
                if self.q < self.nnodes - 1:
                    self.name = self.name + f"_q{self.q:g}"
                if self.r != np.inf:
                    self.name = self.name + f"_r{self.r:g}"  # 1.0 -> "r1", 1.5 -> "r1.5"
                self.isdirected = not np.array_equal(self.dismat, self.dismat.T)
                self.mindis = self._getmindis(self.q, self.r, self.dismat)
                limit = 1.1 * self.proximity.max
                links = (self.dismat < limit) & (self.dismat == self.mindis)
                np.fill_diagonal(links, False)
                self.adjmat = np.copy(self.dismat)
                self.adjmat[~links] = 0
                self.nlinks = np.count_nonzero(self.adjmat)
                if not self.isdirected:
                    self.nlinks = int(self.nlinks / 2)
            case "nn": # nearest neighbor
                self.isdirected = True
                self.q = None
                self.r = None
                self.name = self.name + "_nn"
                dis = self.dismat.copy()
                # set diagonal to np.inf
                np.fill_diagonal(dis, np.inf)
                # get the minimum from each row
                min_neighbor_val = np.min(dis, axis=1)
                self.adjmat = np.copy(self.dismat)
                for i in range(self.nnodes):
                    for j in range(self.nnodes):
                        if (self.dismat[i,j] >= self.proximity.max or
                                self.dismat[i,j] != min_neighbor_val[i]):
                            self.adjmat[i,j] = 0
                        else:
                            if i != j:
                                links[i,j] = True
                self.nlinks = np.count_nonzero(links)

        np.fill_diagonal(links, False)
        self.adjmat = np.round(self.adjmat, 6)
        self.adjmat[links & (self.adjmat == 0)] = 1.0e-20 # placeholder for 0 weight links
        self.links, self.unique_weights = get_link_ranks(self.dismat, links, self.isdirected)
        self.eccentricity = self.get_eccentricity()
        self.graph = graph_from_adjmat(self.adjmat, self.terms)
        self.graph.name = self.name
        self.is_planar = nx.is_planar(self.graph)
        self.get_layout() # default coords

    def get_network_properties(self) -> dict:
        """Per-node properties as columns (lists in term order). "max degree", "center" and
        "median" are True for the nodes that have that property. "mean link dist" and
        "median" are present only for connected networks."""
        properties = {"node": list(self.terms)}
        links = self.adjmat > 0
        links = links.astype(int)
        indegree = np.sum(links, axis=0)
        outdegree = np.sum(links, axis=1)
        degree = (indegree + outdegree).tolist()
        if self.isdirected:
            properties["indegree"] = indegree.tolist()
            properties["outdegree"] = outdegree.tolist()
            properties["degree"] = degree
        else:
            properties["degree"] = indegree.tolist()
            degree = indegree.tolist()
        max_degree = max(degree)
        properties["max degree"] = [bool(d == max_degree) for d in degree]
        ecc = self.eccentricity
        properties["eccentricity"] = ecc["ecc"]
        properties["center"] = [i in ecc["center"] for i in range(self.nnodes)]
        if self.isconnected:
            properties["mean link dist"] = ecc["mean link dist"]
            properties["median"] = [bool(m == ecc["min mean links"]) for m in ecc["mean link dist"]]
        return properties

    def _getmindis(self, q, r, dis) -> np.ndarray:
        if q >= (self.nnodes - 1):
            mindis = floyd(dis, r)
        else:
            mindis = dijkstra(dis, q, r)
        return mindis

    def get_info(self) -> dict:
        info: dict = {}
        info["name"] = self.name
        info["nnodes"] = self.nnodes
        info["nlinks"] = self.nlinks
        info["connected?"] = self.isconnected
        info["planar?"] = self.is_planar
        info["directed?"] = self.isdirected
        info["type"] = self.type
        info["q"] = self.q  # number (np.inf for infinity), or None for nn and merged nets
        info["r"] = self.r
        return info

    def get_layout(self, method="auto") -> None:
        """
        Creates layout coordinates for PFnet visualization in self.coords.
        """
        from pypf.layouts import get_coords
        self.coords, self.layout_method = get_coords(network=self, method=method)
        self.layout_coords = self.coords.copy()  # as computed, before any user moves
        self.layout_request = method  # method asked for ("auto" etc.); layout_method is what ran

    def set_coords(self, coords: np.ndarray, method: str = "saved") -> None:
        """Uses given layout coordinates (n x 2, rows in term order) instead of computing them."""
        coords = np.array(coords, dtype=float)
        if coords.shape != (self.nnodes, 2):
            raise ValueError(f"coords shape {coords.shape} does not match {self.nnodes} nodes")
        self.coords = coords
        self.layout_coords = coords.copy()
        self.layout_method = method
        self.layout_request = method

    def netprint(self, note=None):
        if note is None:
            note = self.name
        print(f"Network: {note}")
        print(f"name: {self.name}")
        print(f"type: {self.type}")
        print(f"terms[0]: {self.terms[0]}")
        print(f"termsid: {self.termsid}")
        print(f"nnodes: {self.nnodes}")
        print(f"q: {self.q}")
        print(f"r: {self.r}")
        print(f"nlinks: {self.nlinks}")
        print(f"isdirected?: {self.isdirected}")
        print(f"graph: {self.graph}")
        print(f"layers: {self.layers}")

    def get_json_node_link_string(self) -> str:
        import json
        from pypf.utility import sanitize_for_json
        # noinspection PyArgumentList
        nld = nx.node_link_data(self.graph, edges="edges")
        snld = sanitize_for_json(nld)
        jst = json.dumps(snld, indent=2)
        return jst

    def get_link_list(self) -> dict:
        n = self.nnodes
        properties = {
            "si": [],
            "ti": [],
            "source": [],
            "target": [],
            "weight": []
        }
        if self.type == "mg":
            properties["nets"] = []
            properties["count"] = []
        for i in range(n):
            start = 0 if self.isdirected else i + 1
            for j in range(start, n):
                weight = self.adjmat[i, j]
                if weight == 0: continue
                properties["si"].append(i)
                properties["ti"].append(j)
                properties["source"].append(self.terms[i])
                properties["target"].append(self.terms[j])
                properties["weight"].append(int(weight) if self.type == "mg" else float(weight))
                if self.type == "mg":
                    binw = f"{int(weight):b}"
                    properties["nets"].append(binw)
                    properties["count"].append(binw.count("1"))
        return properties

    def get_eccentricity(self) -> dict:
        """
        Calculates the eccentricity, radius, diameter, center, and median of a graph

        Returns:
            dict: A dictionary containing the eccentricity, radius, diameter, and center.
                  - 'ecc': A numpy array of eccentricities for each node.
                  - 'radius': The radius of the graph.
                  - 'diameter': The diameter of the graph.
                  - 'center': A numpy array of the indices (0-based) of the center nodes.
                  - 'mean links': The mean number of links between a node and other nodes.
                  - 'min mean links': The minimum mean number of links between a node and other nodes.
                  - 'median': A numpy array of the indices (0-based) of the median nodes.
        """
        adjmat = self.adjmat
        n = self.nnodes
        adjmat = np.maximum(adjmat, adjmat.T)
        links = adjmat.astype(bool)
        dis = np.full((n, n), np.inf)
        dis[links] = 1
        np.fill_diagonal(dis, 0)
        dist_matrix = floyd(dis, 1)
        self.isconnected = not np.any(np.isinf(dist_matrix))
        dist_noinf = dist_matrix.copy()
        dist_noinf[np.isinf(dist_noinf)] = 0
        ecc = np.max(dist_noinf, axis=1)
        try:
            radius = np.min(ecc[ecc > 0])
            diameter = np.max(ecc)
        except ValueError:  # no links: every eccentricity is 0
            radius = np.int64(0)
            diameter = np.int64(0)
        center = np.where(ecc == radius)[0].astype(int).tolist()
        mean_links = np.mean(dist_noinf, axis=1).astype(float).tolist()
        min_mean_links = np.min(mean_links).astype(float)
        median = np.where(mean_links == min_mean_links)[0].astype(int).tolist()
        return {
            'ecc': ecc.astype(int).tolist(),
            'radius': radius.astype(int),
            'diameter': diameter.astype(int),
            'center': center,
            'mean link dist': mean_links,
            'min mean links': min_mean_links,
            'median': median
        }
