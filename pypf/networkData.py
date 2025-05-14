import numpy as np
import networkx as nx

def networkData(terms: list, adj_matrix: np.ndarray, canvas_width: int, canvas_height: int) -> tuple[dict, list]:
    """
    Creates node and edge data for network visualization, ensuring nodes are within the canvas.

    Args:
        terms (list): List of node labels.
        adj_matrix (numpy.ndarray): Weighted adjacency matrix.
        canvas_width (int): Width of the canvas.

        canvas_height (int): Height of the canvas.

    Returns:
        tuple: (nodes: dict, edges: list).
    """

    num_terms = len(terms)
    if adj_matrix.shape != (num_terms, num_terms):
        raise ValueError("Adjacency matrix must be square and match the number of terms.")

    # Create a networkx graph
    graph = nx.Graph()
    for i, term in enumerate(terms):
        graph.add_node(term)
        for j in range(i + 1, num_terms):
            if adj_matrix[i, j] != 0:
                graph.add_edge(terms[i], terms[j], weight=adj_matrix[i, j])

    # Apply layout algorithm with a random seed
    seed = np.random.randint(1000)
    layout = nx.spring_layout(graph, seed=seed)

    # Find the bounding box of the layout
    min_x = min(coord[0] for coord in layout.values())
    max_x = max(coord[0] for coord in layout.values())
    min_y = min(coord[1] for coord in layout.values())
    max_y = max(coord[1] for coord in layout.values())

    width = max_x - min_x
    height = max_y - min_y
    epsilon = 1e-6  # To avoid division by zero

    nodes = {}
    padding_x = 20  # Adjust this value to control the x-axis limit
    padding_y = 20  # Adjust this value to control the y-axis limit
    scale_x = (canvas_width - 2 * padding_x) / max(width, epsilon) if width > 0 else 1.0
    scale_y = (canvas_height - 2 * padding_y) / max(height, epsilon) if height > 0 else 1.0
    offset_x = padding_x - min_x * scale_x if width > 0 else (canvas_width - 2 * padding_x) / 2
    offset_y = padding_y - min_y * scale_y if height > 0 else (canvas_height - 2 * padding_y) / 2

    for term, (x, y) in layout.items():
        x_canvas = x * scale_x + offset_x
        y_canvas = y * scale_y + offset_y
        nodes[term] = {'x': x_canvas, 'y': y_canvas, 'label': term}

    edges = list(graph.edges())
    return nodes, edges


if __name__ == "__main__":
    from pypf.proximity import Proximity
    from pypf.pfnet import PFnet
    import tkinter as tk
    from pypf.networkDisplay import NetworkDisplay
    # from pypf.show import networkData
    import os.path
    from pypf.utility import get_parent_dir

    # file_path = os.path.join(get_parent_dir(),"data","psy.prx.xlsx")
    # prx = Proximity(file_path)
    prx = Proximity()
    net = PFnet(prx)

    root = tk.Tk()
    root.title("Network with Layout")
    cwidth = 600
    cheight = 400
    network_display = NetworkDisplay(root, nodes={}, edges=[], width=cwidth,
                                     height=cheight)  # Create NetworkDisplay first
    network_display.pack(fill=tk.BOTH, expand=True)

    n_nodes, n_edges = networkData(net.terms, net.adjmat, cwidth, cheight)
    network_display.nodes = n_nodes  # Update NetworkDisplay with correct data
    network_display.edges = n_edges
    network_display.draw_network()  # Redraw the network

    root.mainloop()