#pypf/pfnet.py
import numpy as np
from pypf.proximity import Proximity
from pypf.utility import floyd, dijkstra, graph_from_adjmat, get_lower, get_off_diagonal, eccentricity, get_termsid
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

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
        self.nnodes = 0
        self.nlinks = 0
        self.dismat = np.array([])  # distance matrix
        self.mindis = np.array([])  # minimum distances
        self.adjmat = np.array([])  # adjacency matrix with link weights
        self.isdirected = False  # true for directed graph
        self.name = "default"  # Initialize the name property
        self.graph = nx.Graph() # or nx.Digraph if directed
        self.layers = None  # nodes in each shell
        self.eccentricity = None
        self.type = type
        self.coords:np.ndarray = None

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
            self.eccentricity = eccentricity(self.adjmat)
            self.layers = self.shells(source=self.eccentricity["center"])
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

    def get_info(self):
        info: dict = {}
        info["name"] = self.name
        info["nnodes"] = self.nnodes
        info["nlinks"] = self.nlinks
        info["isdirected"] = self.isdirected
        info["type"] = self.type
        info["q"] = self.q
        info["r"] = self.r
        return info # a list

    def get_layout(self, method="gravity") -> dict:
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
        layout = None  # Initialize layout
        try:
            match method:
                case "gravity":
                    layout = nx.forceatlas2_layout(graph, strong_gravity=True, seed=seed, scaling_ratio=2.0,
                                                   max_iter=100, pos=randpos)
                case "arf":  # too many crossings
                    layout = nx.arf_layout(graph, seed=seed, max_iter=1000)
                case "spring": # gravity is better
                    layout = nx.spring_layout(graph, iterations=50, seed=seed, pos=randpos, center=center)
                case "circle":
                    layout = nx.circular_layout(graph, scale=1, center=center)
                case "shells":
                    from pypf.utility import map_indices_to_terms
                    nlist = map_indices_to_terms(self.layers, self.terms)
                    layout = nx.shell_layout(graph, nlist=nlist, scale=1, center=center)
                case "kamda-kawai":
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
                case "distance":
                    dis = self.dismat.copy()
                    # make dis symmetrical using min
                    dis = np.minimum(dis, dis.T)
                    # make a complete graph from dis
                    cgr = graph_from_adjmat(dis, self.terms)
                    layout = nx.kamada_kawai_layout(cgr, weight="weight", pos=randpos,)
                    # layout = nx.forceatlas2_layout(cgr, strong_gravity=False, seed=seed,
                    #                                scaling_ratio=0.5,
                    #                                max_iter=10000, pos=randpos,
                    #                                weight="weight", linlog=False)

        except Exception as e:
            print(f"Error in get_layout: {e}")

        if layout is None:  # Fallback if initial method somehow didn't set a layout
            layout = nx.forceatlas2_layout(graph, strong_gravity=True, seed=seed, scaling_ratio=2.0, max_iter=1000,
                                           pos=randpos)
        graph.name = self.name  # Restore original name
        coord = np.ndarray(shape=(self.nnodes, 2))
        count = 0
        for node in graph.nodes:
            coord[count, :] = layout[node]
            count += 1
        ccoord = canonical(coord)
        self.coords = ccoord
        canonlo = nx.rescale_layout_dict(layout, scale=1)
        return {"layout": layout, "coord": coord, "canonical_coord": ccoord, "cannonical_layout": canonlo}

    def netprint(self, note=None):
        print("\nPFnet Object:", note)
        print(f"proximity: {self.proximity}")
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

if __name__ == '__main__':
    from pypf.proximity import Proximity
    from pypf.pfnet import PFnet
    from pypf.utility import get_test_pf
    import matplotlib.pyplot as plt
    import networkx as nx
    from pypf.netpic import Netpic
    ap = Proximity("data/attn_example (14).csv")
    an = PFnet(ap)
    an.netprint()

    # prx = Proximity("data/statecaps.prx.xlsx")
    # net = PFnet(prx, q=2, r=2, type="pf")
    # net.get_layout(method="kamda-kawai")
    # pic = Netpic(net)
    # fig = pic.create_view()
    # fig.show()


    # print(net.coords)
    # c = net.get_layout()
    # print(c == net.coords)
    # print(c)

    # net.netprint()
    # print(net.get_info())
    # eccentricity = eccentricity(net.adjmat)
    # print(eccentricity)


    #seed = np.random.randint(1000000)
    # layout = nx.forceatlas2_layout(psy.graph, strong_gravity=True, seed=seed, scaling_ratio=2.0, max_iter=1000)
    # nx.draw_networkx(psy.graph, pos=layout, with_labels=True, node_size=0, arrows=True, min_source_margin=100,
    #                  min_target_margin=100, node_color="white", font_size=9)

    # pos = nx.drawing.nx_pydot.pydot_layout(psy.graph, prog='dot')  # 'dot' is for layered
    # nx.draw(psy.graph, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=12)
    # plt.show()
