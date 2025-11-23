# pypf/pfnet.py
import numpy as np
import os
from pypf.proximity import Proximity
from pypf.utility import floyd, dijkstra, graph_from_adjmat, get_lower, get_off_diagonal, get_termsid
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx

class PFnet:
    """
    PFnet class is used for creating and managing Proximity File Networks, which represent
    graph structures with n_nodes, n_edges, and distances. It is capable of handling weighted
    graphs, both directed and undirected, and generates a pfnet(q,r).

    The class is initialized using a proximity object containing terms and a distance matrix,
    along with other optional parameters for controlling graph behavior. It supports multiple
    algorithms for computing minimum distances between n_nodes (Floyd's algorithm and Dijkstra's).
"""
    def __init__(self, proximity:Proximity=None, q=np.inf, r=np.inf, type="pf"):
        """
        Creates a PFnet object.
                :param proximity: Object containing graph data, specifically a list of terms and a distance matrix.
                            if None, user is prompted to select a proximity file.
        :param q: Optional parameter to control the subset of n_nodes considered for minimum
                  distance computations. Defaults to `np.inf`.
        :param r: Optional parameter to control Minkowski distance calculation. Defaults to `np.inf`.
        :param type: Optional parameter to control graph type. Defaults to "pf".

        """
        self.q = q # Inf yields nterms-1
        self.r = r
        #self.infchar = "\u221E"
        self.terms = []  # node labels
        self.termsid = ""
        self.nnodes:int = 0
        self.nlinks:int = 0
        self.dismat = np.array([])  # distance matrix
        self.mindis = np.array([])  # minimum distances
        self.adjmat = np.array([])  # adjacency matrix with link weights
        self.isconnected = True
        self.isdirected = False  # true for directed graph
        self.name:str = "default"  # Initialize the name property
        self.graph = nx.Graph() # or nx.Digraph if directed
        self.layers = None  # nodes in each shell
        self.eccentricity:dict = {}
        self.type:str = type
        self.coords:np.ndarray | None = None

        if proximity is None:
            proximity = Proximity()
        if hasattr(proximity, 'terms') and hasattr(proximity, 'dismat'):
            self.name = proximity.name
            self.proximity = proximity
            self.terms = proximity.terms
            self.nnodes = len(self.terms)
            self.dismat = np.array(proximity.dismat)
            self.termsid = proximity.termsid
            if self.termsid == "":
                self.termsid = get_termsid(self.terms)

            links = np.full((self.nnodes, self.nnodes), True, dtype=bool)
            np.fill_diagonal(links, False)

            # noinspection PyUnreachableCode
            match self.type:
                case "pf":
                    self.name = self.name + "_pf"
                    if self.q < self.nnodes - 1:
                        self.name = self.name + "_q" + str(self.q)
                    if self.r != np.inf:
                        self.name = self.name + "_r" + str(self.r)
                    self.isdirected = not np.array_equal(self.dismat, self.dismat.T)
                    self.mindis = self._getmindis(self.q, self.r, self.dismat)
                    links = (self.dismat < np.inf) & (self.dismat == self.mindis)
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
                    dis[np.diag_indices_from(dis)] = np.inf
                    # get the minimum from each row
                    min_neighbor_val = np.min(dis, axis=1)
                    self.adjmat = np.copy(self.dismat)
                    for i in range(self.nnodes):
                        for j in range(self.nnodes):
                            if self.dismat[i,j] == np.inf or self.dismat[i,j] != min_neighbor_val[i]:
                                self.adjmat[i,j] = 0
                                links[i,j] = 0
                    self.nlinks = np.count_nonzero(links)
                case "th": # threshold
                    self.q = None
                    self.r = None
                    self.name = self.name + "_th"
                    if proximity.issymmetric:
                        values = get_lower(self.dismat)
                        self.isdirected = False
                    else:
                        values = get_off_diagonal(self.dismat)
                        self.isdirected = True
                    values = np.sort(values)
                    cut = values[self.nnodes-1]
                    links[self.dismat > cut] = False
                    self.adjmat = np.copy(self.dismat)
                    self.adjmat[~links] = 0
                    self.nlinks = np.count_nonzero(self.adjmat)
                    if not self.isdirected:
                        self.nlinks = int(self.nlinks / 2)
                case _:
                    print("Error: PFnet type not valid")

            np.fill_diagonal(links, False)
            self.adjmat[links & (self.adjmat == 0)] = 1.0e-20
            self.eccentricity = self.get_eccentricity()
            self.graph = graph_from_adjmat(self.adjmat, self.terms)
            self.graph.name = self.name
            self.get_layout() # default coords
        else:
            print("Error: proximity not valid")

    def get_network_properties(self) -> dict:
        properties = {"node": self.terms}
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
        properties["max degree"] = [term if degree[i] == max_degree else "" for i, term in enumerate(self.terms)]
        ecc = self.eccentricity
        properties["eccentricity"] = ecc["ecc"]
        center_indices = ecc["center"]
        properties["center"] = [term if i in center_indices else "" for i, term in enumerate(self.terms)]
        if self.isconnected:
            properties["mean link dist"] = ecc["mean link dist"]
            properties["median"] = [term if ecc["mean link dist"][i] == ecc["min mean links"]
                                    else "" for i, term in enumerate(self.terms)]
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
        info["isconnected"] = self.isconnected
        info["isdirected"] = self.isdirected
        info["type"] = self.type
        info["q"] = f"{self.q:.0f}" if self.q else None
        info["r"] = f"{self.r:g}" if self.r else None
        return info # a dictionary

    def get_layout(self, method="kamda-kawai") -> None:
        """
        Creates layout coordinates for PFnet visualization and returns a layout dictionary.
        """
        from pypf.utility import canonical
        graph = self.graph
        graph.name = "temp"  # avoid illegal characters in graphviz names
        seed = np.random.randint(1000000)
        center = None
        randpos = nx.random_layout(graph, seed=seed)
        root = self.eccentricity["center"] if self.eccentricity else None
        root = list(root) if root else None
        root = root[0] if root else None
        # print(f"root: {root}")
        layout = None
        coord = None
        try:
            match method:
                case "gravity":
                    layout = nx.forceatlas2_layout(graph, strong_gravity=True, seed=seed, scaling_ratio=2.0,
                                                   max_iter=100, pos=randpos)
                case "spring": # gravity is better
                    layout = nx.spring_layout(graph, iterations=50, seed=seed, pos=randpos, center=center)
                case "circle":
                    layout = nx.circular_layout(graph, scale=1, center=center)
                case "shells":
                    from pypf.utility import map_indices_to_terms
                    nlist = map_indices_to_terms(self.layers, self.terms)
                    layout = nx.shell_layout(graph, nlist=nlist, scale=1, center=center)
                case "kamda-kawai":
                    # noinspection PyTypeChecker
                    layout = nx.kamada_kawai_layout(graph, weight=None, pos=randpos, center=center)
                case "spiral":
                    layout = nx.spiral_layout(graph, resolution=0.5, scale=1, center=center, equidistant=True)
                case "force":
                    layout = nx.forceatlas2_layout(graph, strong_gravity=False, seed=seed, scaling_ratio=2.0,
                                                   max_iter=100, pos=randpos)
                case "dot":
                    layout = graphviz_layout(graph, prog='dot', root=root)
                case "neato":
                    layout = graphviz_layout(graph, prog='neato', root=root)
                case "fdp": #fdp
                    layout = graphviz_layout(graph, prog='fdp', root=root)
                case "circo":
                    layout = graphviz_layout(graph, prog='circo', root=root)
                case "link distances":
                    layout = nx.kamada_kawai_layout(graph, weight='weight', pos=randpos)
                case "MDS":
                    from pypf.utility import mds_coord
                    coord = mds_coord(self.dismat, metric=True)

        except Exception as e:
            print(f"Error in get_layout: {e}")

        if layout is None:
           layout = nx.forceatlas2_layout(graph, strong_gravity=True, seed=seed, scaling_ratio=2.0, max_iter=1000,
                                          pos=randpos)
        if coord is None:
            coord = np.ndarray(shape=(self.nnodes, 2))
            for i, node in enumerate(graph.nodes):
                coord[i, :] = layout[node]

        coord = canonical(coord)
        # reverse x coords
        coord[:,0] = -coord[:,0]
        self.coords = coord
        graph.name = self.name  # Restore original name

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

    def get_link_list(self) -> dict:
        n = self.nnodes
        properties = {
            "si": [],
            "ti": [],
            "source": [],
            "target": [],
            "weight": []
        }
        if self.type == "merge":
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
                properties["weight"].append(f"{weight:.7g}")
                if self.type == "merge":
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
        radius = np.min(ecc[ecc > 0])
        diameter = np.max(ecc)
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

def get_test_pf(name:str="psy") -> PFnet:
    from pypf.proximity import Proximity
    filepath = os.path.join("data", name + ".prx.xlsx")
    prx = Proximity(filepath)
    pf = PFnet(prx)
    return pf

if __name__ == '__main__':
    import pandas as pd
    ap = Proximity("data/statecaps.prx.xlsx")
    an = PFnet(ap, type="pf", q=2, r=2)
    # an.netprint()
    print(an.get_info())
    tab = pd.DataFrame(an.get_network_properties())
    print(tab)

