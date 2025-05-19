import tkinter as tk

class NetworkDisplay(tk.Canvas):
    def __init__(self, parent, nodes, edges, **kwargs):
        super().__init__(parent, **kwargs)
        self.nodes = nodes
        self.edges = edges
        self.node_items = {}
        self.edge_items = {}
        self.dragging_item = None
        self.drag_start_x_canvas = 0
        self.drag_start_y_canvas = 0
        self.drag_start_node_x_logical = 0
        self.drag_start_node_y_logical = 0
        self.canvas_height = int(kwargs.get('height', 200))

        self.draw_network()
        self.bind("<Configure>", self._on_resize)
        self.bind("<Button-1>", self.start_drag)
        self.bind("<B1-Motion>", self.do_drag)
        self.bind("<ButtonRelease-1>", self.stop_drag)

    def _on_resize(self, event):
        self.canvas_height = event.height
        self.draw_network()

    def _to_canvas_y(self, logical_y):
        return self.canvas_height - logical_y

    def _to_logical_y(self, canvas_y):
        return self.canvas_height - canvas_y

    def draw_network(self):
        self.delete("all")
        self.edge_items = {}
        self.node_items = {}

        # Draw n_edges
        for id1, id2 in self.edges:
            if id1 in self.nodes and id2 in self.nodes:
                x1, y1 = self.nodes[id1]['x'], self._to_canvas_y(self.nodes[id1]['y'])
                x2, y2 = self.nodes[id2]['x'], self._to_canvas_y(self.nodes[id2]['y'])
                sorted_edge = tuple(sorted((id1, id2)))
                edge_id = self.create_line(x1, y1, x2, y2, fill="gray", tags="edge")
                self.edge_items[sorted_edge] = edge_id

        # Draw n_nodes (as text)
        for node_id, data in self.nodes.items():
            x, y = data['x'], self._to_canvas_y(data['y'])
            label = data['label']
            text_id = self.create_text(x, y, text=label, anchor=tk.CENTER, tags=("node", node_id))
            self.node_items[node_id] = text_id

    def start_drag(self, event):
        item = self.find_closest(event.x, event.y)[0]
        tags = self.gettags(item)
        if "node" in tags:
            self.dragging_item = item
            self.drag_start_x_canvas = event.x
            self.drag_start_y_canvas = event.y
            node_id = [tag for tag in tags if tag != "node"][0]
            self.drag_start_node_x_logical = self.nodes[node_id]['x']
            self.drag_start_node_y_logical = self.nodes[node_id]['y']

    def do_drag(self, event):
        if self.dragging_item:
            dx_canvas = event.x - self.drag_start_x_canvas
            dy_canvas = event.y - self.drag_start_y_canvas

            # Calculate the new logical coordinates RELATIVE to the start
            tags = self.gettags(self.dragging_item)
            node_id = [tag for tag in tags if tag != "node"][0]
            new_x_logical = self.drag_start_node_x_logical + dx_canvas
            new_y_logical = self.drag_start_node_y_logical - dy_canvas

            # Update the node's logical coordinates
            self.nodes[node_id]['x'] = new_x_logical
            self.nodes[node_id]['y'] = new_y_logical

            # Move the visual element on the canvas
            self.move(self.dragging_item, dx_canvas, dy_canvas)
            self.update_edge_positions()

            # Reset drag offsets after moving
            self.drag_start_x_canvas = event.x
            self.drag_start_y_canvas = event.y
            self.drag_start_node_x_logical = new_x_logical
            self.drag_start_node_y_logical = new_y_logical

    def stop_drag(self, event):
        self.dragging_item = None

    def update_edge_positions(self):
        updated_edges = set()
        for id1, id2 in self.edges:
            if id1 in self.nodes and id2 in self.nodes:
                x1, y1 = self.nodes[id1]['x'], self._to_canvas_y(self.nodes[id1]['y'])
                x2, y2 = self.nodes[id2]['x'], self._to_canvas_y(self.nodes[id2]['y'])
                sorted_edge = tuple(sorted((id1, id2)))
                if sorted_edge in self.edge_items and sorted_edge not in updated_edges:
                    self.coords(self.edge_items[sorted_edge], x1, y1, x2, y2)
                    updated_edges.add(sorted_edge)

# Example Usage
if __name__ == "__main__":
    import matplotlib
    from pypf.networkData import networkData
    from pypf.utility import get_test_pf
    psy = get_test_pf()

    width = 600
    height = 600
    nodes, edges = networkData(psy, "force", canvas_width=width, canvas_height=height)

    root = tk.Tk()
    root.title("Basic Network Display")
    network_display = NetworkDisplay(root, nodes, edges, width=width, height=height)
    network_display.pack(fill=tk.BOTH, expand=True)
    root.mainloop()