#pypf/pfnet.py
import numpy as np
from pypf.proximity import Proximity
from pypf.utility import floyd, dijkstra, graph_from_adjmat
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
    def __init__(self, proximity:Proximity=None, q=np.inf, r=np.inf):
        """
        Creates a PFnet object.
                :param proximity: Object containing graph data, specifically a list of terms and a distance matrix.
                            if None, user is prompted to select a proximity file.
        :param q: Optional parameter to control the subset of n_nodes considered for minimum
                  distance computations. Defaults to `np.inf`.
        :param r: Optional parameter to control Minkowski distance calculation. Defaults to `np.inf`.

        """
        self.q = q # Inf yields nterms-1
        self.r = r
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

        if proximity is None:
            proximity = Proximity()
        if hasattr(proximity, 'name'):
            self.name = proximity.name
        if hasattr(proximity, 'terms'):
            self.terms = proximity.terms
            self.nnodes = len(self.terms)
            if self.q >= self.nnodes -1:
                self.q = np.inf
            if hasattr(proximity, 'dismat'):
                self.dismat = np.array(proximity.dismat)
                self.isdirected = not np.array_equal(self.dismat, self.dismat.T)
                self.mindis = self._getmindis(self.q, self.r, self.dismat)
                links = (self.dismat < np.inf) & (self.dismat == self.mindis)
                self.adjmat = np.copy(self.dismat)
                self.adjmat[~links] = 0
                self.nlinks = np.count_nonzero(self.adjmat)
                self.graph = graph_from_adjmat(self.adjmat, self.terms)
                self.graph.name = self.name
                ecc = self.eccentricity()
                self.layers = self.shells(ecc["center"])
                if not self.isdirected:
                    self.nlinks = self.nlinks / 2
            else:
                print("Warning: proximity object missing 'dismat' property.")
        else:
            print("Error: Input 'proximity' object missing 'terms' property.")

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
        print(f"name: {self.name}")
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

    psy = get_test_pf()
    psy.netprint()
    seed = np.random.randint(1000000)
    layout = nx.forceatlas2_layout(psy.graph, strong_gravity=True, seed=seed, scaling_ratio=2.0, max_iter=1000)
    nx.draw_networkx(psy.graph, pos=layout, with_labels=True, node_size=0, arrows=True, min_target_margin=100)
    plt.show()
