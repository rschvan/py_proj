# pypf/layouts
import igraph as ig
import networkx as nx
import numpy as np
from pypf.utility import canonical

def resolve_auto_algorithm(graph: ig.Graph) -> str:
    """Mirrors igraph's internal heuristic to report the auto-selected layout."""
    num_nodes = graph.vcount()
    num_edges = graph.ecount()

    if num_edges == 0:
        return "circle"
    if graph.is_tree():
        return "reingold_tilford"
    if num_nodes <= 100:
        return "kamada_kawai"
    if num_nodes <= 10000:
        return "fruchterman_reingold"
    return "drl"

def igraph_coords(graph: ig.Graph, algorithm="auto", **kwargs) -> (np.ndarray, str):
    iglo = graph.layout(algorithm, **kwargs)
    coords = np.array(iglo.coords, dtype=float)
    if algorithm == "auto":
        auto_algorithm = resolve_auto_algorithm(graph)
    else:
        auto_algorithm = algorithm
    return coords, auto_algorithm

def find_center_roots(graph) -> list[int]:
    """Finds center node(s) with minimum eccentricity for each connected component."""
    # Use an undirected view so DiGraphs calculate topological centers cleanly
    undirected_g = (
        graph.to_undirected(as_view=True) if graph.is_directed() else graph )
    node_to_idx = {node: i for i, node in enumerate(graph.nodes)}

    roots = []
    for comp in nx.connected_components(undirected_g):
        sub = undirected_g.subgraph(comp)
        centers = nx.center(sub)
        # Pick the center node with the highest degree
        best_center = max(centers, key=lambda n: sub.degree[n])
        roots.append(node_to_idx[best_center])

    return roots

def get_coords(network, method="auto") -> (np.ndarray, str):
    """Creates layout coordinates for PFnet visualization using network.coords.
        coordinates are returned as well as assigned to network.coords
    """
    graph = network.graph
    ig_graph = ig.Graph.from_networkx(graph)
    actual = method
    seed = np.random.randint(1000000)
    randpos = nx.random_layout(graph, seed=seed)

    layout = None
    coord = None

    try:
        match method:
            case "gravity":
                layout = nx.forceatlas2_layout(
                    graph,
                    strong_gravity=True,
                    seed=seed,
                    scaling_ratio=2.0,
                    weight=None,
                    max_iter=100,
                    pos=randpos,
                    linlog=False,
                )

            case "spring":
                k = 4.0 / np.sqrt(network.nnodes) if network.nnodes > 0 else 1.0
                layout = nx.spring_layout(
                    graph, pos=randpos, iterations=100, k=k, seed=seed, )

            case "circle":
                coord, _ = igraph_coords(ig_graph, algorithm="circle")

            case "kamada_kawai" | "kk":
                rand_coords = [randpos[node] for node in graph.nodes]
                coord, _ = igraph_coords(
                    ig_graph, algorithm="kamada_kawai", seed=rand_coords, )

            case "fruchterman_reingold" | "fr" | "force":
                coord, _ = igraph_coords(ig_graph, algorithm="fruchterman_reingold")

            case "davidson_harel" | "dh":
                coord, _ = igraph_coords(ig_graph, algorithm="davidson_harel", weight_edge_crossings=3)

            case "radial_tree":
                roots = find_center_roots(graph)
                coord, _ = igraph_coords(
                    ig_graph,
                    algorithm="reingold_tilford_circular",
                    root=roots,
                    mode="all", )

            case "tree":
                roots = find_center_roots(graph)
                coord, _ = igraph_coords(
                    ig_graph, algorithm="reingold_tilford", root=roots, mode="all", )

            case "spiral":
                layout = nx.spiral_layout(
                    graph,
                    resolution=0.5,
                    scale=1.0,
                    center=[0, 0],
                    equidistant=True, )

            case "planar":
                if nx.is_planar(graph):
                    layout = nx.planar_layout(graph,)
                else:
                    rand_coords = [randpos[node] for node in graph.nodes]
                    coord, _ = igraph_coords(
                        ig_graph, algorithm="kamada_kawai", seed=rand_coords, )

            case "MDS" | "mds" | "mds_distance":
                coord, _ = igraph_coords(
                    ig_graph, algorithm="mds", dist=network.dismat, )

            case "mds_graph":
                coord, _ = igraph_coords(ig_graph, algorithm="mds")

            case "sugiyama" | "sg":
                coord, _ = igraph_coords(ig_graph, algorithm="sugiyama")

            case _:
                coord, actual = igraph_coords(ig_graph, algorithm="auto")

    except Exception as e:
        print(f"Error computing layout '{method}': {e}")
        layout = None
        coord, actual = igraph_coords(ig_graph, algorithm="auto")

    # Convert NetworkX dictionary layout to (N, 2) numpy array
    if coord is None and layout is not None:
        coord = np.empty((network.nnodes, 2), dtype=float)
        for i, node in enumerate(graph.nodes):
            coord[i, :] = layout[node]

    # Final fallback if coordinate generation failed
    if coord is None:
        coord, actual = igraph_coords(ig_graph, algorithm="auto")

    coord = canonical(coord)
    # Reverse x coords
    coord[:, 0] = -coord[:, 0]
    if actual in ["tree", "sugiyama"]:
        coord[:, 1] = -coord[:, 1] # trees grow down
    #network.coords = coord --  make assignment in network

    return coord, actual

if __name__ == "__main__":
    import os
    from pypf.pfnet import PFnet
    from pypf.proximity import Proximity

    data_set = "bio"
    lomethod = "auto"
    pfile = os.path.join("data", data_set + ".prx.xlsx")
    ap = Proximity(pfile)
    an = PFnet(ap, type="pf")
    xy, used = get_coords(network=an, method=lomethod)
    print(f"data: {data_set}, method: {used}")
    print("Coordinates shape:", xy.shape)
    print(an.coords[:3])