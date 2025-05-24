import numpy as np
import networkx as nx
from pypf.pfnet import PFnet
from pypf.utility import newcoords

def networkData(net:PFnet, method = None, canvas_width: int=600, canvas_height: int=600) -> tuple[dict, list]:
    """
    Creates node and edge data for network visualization, ensuring nodes are within the canvas.
        Args:
            terms (list): List of node labels.
            adj_matrix (numpy.ndarray): Weighted adjacency matrix.
            canvas_width (int): Width of the canvas.
            canvas_height (int): Height of the canvas.
        Returns:
            tuple: (nodes: dict, edges: list).
            :param method: layout method: arf, spring, circle, shells, kamda-kawai, spiral, force, force with gravity
            :param canvas_height:
            :param canvas_width:
            :param net:
    """
    seed = np.random.randint(1000000)
    center = [canvas_width // 2, canvas_height // 2]
    randpos = nx.random_layout(net.graph, seed=seed)
    match method:
        case "arf":
            layout = nx.arf_layout(net.graph, seed=seed, max_iter=1000)
        case "spring":
            layout = nx.spring_layout(net.graph, iterations=100, seed=seed, center=center)
        # case "bfs":
        #     start = graph.nodes[center[0]] if center else graph.nodes[0]
        #     layout = nx.bfs_layout(graph, start=start)
        case "circle":
            layout = nx.circular_layout(net.graph, scale=1, center=center)
        case "shells":
            from pypf.utility import map_indices_to_terms
            nlist = map_indices_to_terms(net.layers, net.terms)
            layout = nx.shell_layout(net.graph, nlist=nlist, scale=1, center=center)
        case "kamda-kawai":
            layout = nx.kamada_kawai_layout(net.graph, weight=None, pos=nx.random_layout(net.graph, seed=seed), center=center)
        case "spiral":
            layout = nx.spiral_layout(net.graph, resolution=0.5, scale=1, center=center, equidistant=True)
        case "force":
            layout = nx.forceatlas2_layout(net.graph, strong_gravity=False, seed=seed, scaling_ratio=2.0, max_iter=100,
                                 pos=randpos)
        case _:  # force with gravity
            layout = nx.forceatlas2_layout(net.graph, strong_gravity=True, seed=seed, scaling_ratio=2.0, max_iter=100,
                                           pos=randpos)

    layout = newcoords(layout, canvas_width, canvas_height, 0, 0)

    nodes = {}
    for term, (x, y) in layout.items():
        nodes[term] = {'x': x, 'y': y, 'label': term}

    edges = list(net.graph.edges())

    return nodes, edges

if __name__ == "__main__":
    import tkinter as tk
    from pypf.networkDisplay import NetworkDisplay
    from pypf.utility import get_test_pf

    net = get_test_pf()
    net.netprint()
    # layout = networkData(net, method="force", canvas_width=600, canvas_height=400)
    # print(layout)
    root = tk.Tk()
    root.title("Network with Layout")
    canvas_width = 400
    canvas_height = 400
    network_display = NetworkDisplay(root, nodes={}, edges=[], width=canvas_width,
                                     height=canvas_height)  # Create NetworkDisplay first
    network_display.pack(fill=tk.BOTH, expand=True)

    n_nodes, n_edges = networkData(net, method="gravity", canvas_width=canvas_width, canvas_height=canvas_height)
    network_display.nodes = n_nodes  # Update NetworkDisplay with correct data
    network_display.edges = n_edges
    network_display.draw_network()  # Redraw the network

    root.mainloop()