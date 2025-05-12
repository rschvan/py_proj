import numpy as np
import networkx as nx

def create_network_data_with_layout(terms: list, adj_matrix: np.ndarray, canvas_width: int, canvas_height: int) -> tuple[dict, list]:
    """
    Creates node and edge data for network visualization, ensuring n_nodes are within the canvas.

    Args:
        terms (list): List of node labels.
        adj_matrix (numpy.ndarray): Weighted adjacency matrix.
        canvas_width (int): Width of the canvas.
        canvas_height (int): Height of the canvas.

    Returns:
        tuple: (n_nodes: dict, n_edges: list).
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

    # Apply layout algorithm
    layout = nx.spring_layout(graph, seed=42)

    # Find the bounding box of the layout
    min_x = min(coord[0] for coord in layout.values())
    max_x = max(coord[0] for coord in layout.values())
    min_y = min(coord[1] for coord in layout.values())
    max_y = max(coord[1] for coord in layout.values())

    width = max_x - min_x
    height = max_y - min_y
    max_dimension = max(width, height)  # Find the larger dimension

    # Calculate scaling factor to fit within the canvas with padding
    padding = 50  # Adjust padding as needed
    scale_factor = min(
        (canvas_width - 2 * padding) / max_dimension,
        (canvas_height - 2 * padding) / max_dimension
    )

    # Calculate offsets to center the layout
    x_offset = (canvas_width - scale_factor * width) / 2
    y_offset = (canvas_height - scale_factor * height) / 2

    # Normalize and scale the coordinates
    nodes = {}
    for term, coords in layout.items():
        x = (coords[0] - min_x) * scale_factor + x_offset
        y = (coords[1] - min_y) * scale_factor + y_offset
        nodes[term] = {'x': x, 'y': y, 'label': term}

    edges = list(graph.edges())
    return nodes, edges

# Example Usage
if __name__ == "__main__":
    # terms = ["animal", "bird", "blood", "red", "living thing"]
    # adj_matrix = np.array([
    #     [0, 0.8, 0.2, 0.1, 0.9],
    #     [0.8, 0, 0, 0, 0.7],
    #     [0.2, 0, 0, 0.6, 0.3],
    #     [0.1, 0, 0.6, 0, 0.2],
    #     [0.9, 0.7, 0.3, 0.2, 0]
    # ])
    from pypf.proximity import Proximity
    from pypf.pfnet import PFnet
    from pypf.utility import get_parent_dir
    import os
    pprx = Proximity(os.path.join(get_parent_dir(), "data", "psy.prx.xlsx"))
    #bprx = Proximity(os.path.join(get_parent_dir(), "data", "bio.prx.xlsx"))
    psy = PFnet(pprx)
    #bio = PFnet(prx)
    from pypf.display import NetworkDisplay
    import tkinter as tk
    root = tk.Tk()
    root.title("Network with Layout")
    cwidth = 600
    cheight = 400
    network_display = NetworkDisplay(root, {}, [], width=cwidth, height=cheight) # Create NetworkDisplay first
    network_display.pack(fill=tk.BOTH, expand=True)

    n_nodes, n_edges = create_network_data_with_layout(psy.terms, psy.adjmat, cwidth, cheight)
    network_display.nodes = n_nodes # Update NetworkDisplay with correct data
    network_display.edges = n_edges
    network_display.draw_network() # Redraw the network

    root.mainloop()