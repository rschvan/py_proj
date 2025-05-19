import networkx as nx
import numpy as np
from collections import deque


def get_graph_center(graph: nx.Graph) -> list:
    """
    Calculates the center nodes of a NetworkX graph.

    Args:
        graph: The input NetworkX graph.

    Returns:
        A list of the center nodes.
    """
    if not nx.is_connected(graph):
        # Handle disconnected graph case
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        eccentricity_map = nx.eccentricity(subgraph)
        radius = min(eccentricity_map.values())
        center_nodes = [node for node, ecc in eccentricity_map.items() if ecc == radius]
        # Map the center nodes back to the original graph's node IDs
        center_nodes_original_ids = [graph.subgraph(largest_cc).nodes[i] for i in range(len(graph.subgraph(largest_cc).nodes)) if graph.subgraph(largest_cc).nodes[i] in center_nodes]

        return center_nodes_original_ids
    else:
        eccentricity_map = nx.eccentricity(graph)
        radius = min(eccentricity_map.values())
        center_nodes = [node for node, ecc in eccentricity_map.items() if ecc == radius]
        return center_nodes


def shells_nx(graph: nx.Graph) -> list:
    """
    Creates layers of nodes (shells) starting from the center nodes of a NetworkX graph
    using Breadth-First Search (BFS).

    Args:
        graph: The input NetworkX graph.

    Returns:
        A list of lists, where each inner list represents a shell of nodes.
    """
    center_nodes = get_graph_center(graph)
    num_nodes = graph.number_of_nodes()
    used_nodes = set(center_nodes)
    node_shells = [center_nodes]
    bfs_queue = deque(center_nodes)

    while len(used_nodes) < num_nodes and bfs_queue:
        current_shell = []
        next_level_queue = deque()

        # Process all nodes at the current level
        for _ in range(len(bfs_queue)):
            current_node = bfs_queue.popleft()
            for neighbor_node in graph.neighbors(current_node):
                if neighbor_node not in used_nodes:
                    used_nodes.add(neighbor_node)
                    current_shell.append(neighbor_node)
                    next_level_queue.append(neighbor_node)

        if current_shell:
            node_shells.append(current_shell)
            bfs_queue = next_level_queue

        # Handle disconnected components
        if len(used_nodes) < num_nodes and not bfs_queue:
            remaining_nodes = [node for node in graph.nodes() if node not in used_nodes]
            if remaining_nodes:
                # Start a new BFS from an arbitrary remaining node
                start_node = remaining_nodes[0]
                used_nodes.add(start_node)
                node_shells.append([start_node])
                bfs_queue.append(start_node)

    return node_shells


if __name__ == "__main__":
    from pypf.proximity import Proximity
    from pypf.pfnet import PFnet
    import os.path
    from pypf.utility import get_parent_dir

    prx = Proximity(os.path.join(get_parent_dir(),"data","psy.prx.xlsx"))
    net = PFnet(prx)
    graph = nx.from_numpy_array(net.adjmat)
    shells_list = shells_nx(graph)
    print(shells_list)

    # # Example Usage:
    # G = nx.Graph()
    # G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5)])
    # print("Graph Edges:", G.edges())
    # shells_list = shells_nx(G)
    # print("Shells (NetworkX):", shells_list)
    #
    # # Example of a disconnected graph
    # G_disconnected = nx.Graph()
    # G_disconnected.add_edges_from([(0, 1), (1, 2)])
    # G_disconnected.add_edges_from([(3, 4)])
    # print("\nGraph Edges (Disconnected):", G_disconnected.edges())
    # shells_list_disconnected = shells_nx(G_disconnected)
    # print("Shells (NetworkX, Disconnected):", shells_list_disconnected)