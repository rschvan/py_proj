#pypf/pfnet.py
import numpy as np
from pypf.proximity import Proximity
from pypf.utility import floyd, dijkstra, graph_from_adjmat, get_lower, get_off_diagonal
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
        self.infchar = "\u221E"
        self.terms = []  # node labels
        self.nnodes = 0
        self.nlinks = 0
        self.dismat = np.array([])  # distance matrix
        self.mindis = np.array([])  # minimum distances
        self.adjmat = np.array([])  # adjacency matrix with link weights
        self.isdirected = False  # true for directed graph
        self.name = "default"  # Initialize the name property
        self.graph = nx.Graph()
        self.layers = list  # nodes in each shell
        self.type = type

        if proximity is None:
            proximity = Proximity()
        if hasattr(proximity, 'name'):
            self.name = proximity.name
        if hasattr(proximity, 'terms') and hasattr(proximity, 'dismat'):
            self.proximity = proximity
            self.terms = proximity.terms
            self.nnodes = len(self.terms)
            self.dismat = np.array(proximity.dismat)
            self.proximity = proximity

            match self.type:
                case "pf":
                    if self.q >= self.nnodes -1:
                        self.q = np.inf
                        self.name = self.name + "_q" + self.infchar
                    else:
                        self.name = self.name + "_q" + str(self.q)
                    if self.r == np.inf:
                        self.name = self.name + "_r" + self.infchar
                    else:
                        self.name = self.name + "_r" + str(self.r)

                    self.isdirected = not np.array_equal(self.dismat, self.dismat.T)
                    self.mindis = self._getmindis(self.q, self.r, self.dismat)
                    links = (self.dismat < np.inf) & (self.dismat == self.mindis)
                    self.adjmat = np.copy(self.dismat)
                    self.adjmat[~links] = 0
                    self.nlinks = np.count_nonzero(self.adjmat)
                    if not self.isdirected:
                        self.nlinks = self.nlinks / 2
                    ecc = self.eccentricity()
                    self.layers = self.shells(ecc["center"])
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
                            if self.dismat[i,j] != min_neighbor_val[i]:
                                self.adjmat[i,j] = 0
                    self.nlinks = np.count_nonzero(self.adjmat)
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
                    links = (self.dismat < np.inf) & (self.dismat <= cut)
                    self.adjmat = np.copy(self.dismat)
                    self.adjmat = self.adjmat * links
                    self.nlinks = np.count_nonzero(self.adjmat)
                    self.graph = graph_from_adjmat(self.adjmat, self.terms)
                case _:
                    print("Error: type not valid")
            self.graph = graph_from_adjmat(self.adjmat, self.terms)
            self.graph.name = self.name

        else:
            print("Error: proximity not valid")

    def _getmindis(self, q, r, dis):
        if q >= (self.nnodes - 1):
            mindis = floyd(dis, r)
        else:
            mindis = dijkstra(dis, q, r)
        return mindis

    def eccentricity(self):
        from pypf.utility import eccentricity
        return eccentricity(self.adjmat)

    def shells(self, source) -> list:
        """
        Creates layers of nodes starting from the center nodes of a graph.
        Args:
            adjmat (numpy.ndarray): A square adjacency matrix.
        Returns:
            list: A list of lists, where each inner list represents a shell of nodes.
                  Node indices are 0-based.
        """
        adjmat = self.adjmat
        n = adjmat.shape[0]
        used = list(source)  # Convert to list
        lists = [used]  # Initialize as a list containing a list
        while len(used) < n:
            keep = []
            last_layer = lists[-1]  # Get the last layer
            for i in range(len(last_layer)):
                new_nodes = np.where(adjmat[last_layer[i], :])[0].tolist()  # Find neighbors
                new_nodes = [node for node in new_nodes if node not in used]  # Remove used nodes
                keep.extend(new_nodes)
            used = list(np.unique(used).astype(int)) + keep
            used = list(np.unique(used).astype(int))

            if len(used) < n and not keep:
                new_nodes = [i for i in range(n) if i not in used]
                keep = new_nodes
                used = list(range(n))

            if keep and len(used) < n:
                lists.append(keep)

        return lists

    def netprint(self, note=None):
        print("\nPFnet Object:", note)
        print(f"proximity: {self.proximity}")
        print(f"name: {self.name}")
        print(f"type: {self.type}")
        print(f"terms[0]: {self.terms[0]}")
        print(f"nnodes: {self.nnodes}")
        print(f"q: {self.q}")
        print(f"r: {self.r}")
        print(f"nlinks: {self.nlinks}")
        print(f"isdirected?: {self.isdirected}")
        print(f"graph: {self.graph}")
        print(f"layers: {self.layers}")

if __name__ == '__main__':
    from pypf.utility import get_test_pf
    import matplotlib.pyplot as plt
    import networkx as nx

    psy = get_test_pf("rbank")
    psy.netprint()
    seed = np.random.randint(1000000)
    # layout = nx.forceatlas2_layout(psy.graph, strong_gravity=True, seed=seed, scaling_ratio=2.0, max_iter=1000)
    # nx.draw_networkx(psy.graph, pos=layout, with_labels=True, node_size=0, arrows=True, min_source_margin=100,
    #                  min_target_margin=100, node_color="white", font_size=9)



    pos = nx.drawing.nx_pydot.pydot_layout(psy.graph, prog='dot')  # 'dot' is for layered
    nx.draw(psy.graph, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=12)

    plt.show()
