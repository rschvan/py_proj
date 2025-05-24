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

        # Draw edges, adjusting endpoints to text boundaries
        for id1, id2 in self.edges:
            if id1 in self.nodes and id2 in self.nodes:
                x1_center, y1_center = self.nodes[id1]['x'], self._to_canvas_y(self.nodes[id1]['y'])
                x2_center, y2_center = self.nodes[id2]['x'], self._to_canvas_y(self.nodes[id2]['y'])

                text_id1 = self.create_text(x1_center, y1_center, text=self.nodes[id1]['label'], anchor=tk.CENTER, tags=("temp"))
                bbox1 = self.bbox(text_id1)
                self.delete(text_id1)
                x1_min, y1_min, x1_max, y1_max = bbox1

                text_id2 = self.create_text(x2_center, y2_center, text=self.nodes[id2]['label'], anchor=tk.CENTER, tags=("temp"))
                bbox2 = self.bbox(text_id2)
                self.delete(text_id2)
                x2_min, y2_min, x2_max, y2_max = bbox2

                x1, y1 = self._get_edge_endpoint(x1_center, y1_center, x2_center, y2_center, x1_min, y1_min, x1_max, y1_max)
                x2, y2 = self._get_edge_endpoint(x2_center, y2_center, x1_center, y1_center, x2_min, y2_min, x2_max, y2_max)

                sorted_edge = tuple(sorted((id1, id2)))
                edge_id = self.create_line(x1, y1, x2, y2, fill="gray", tags="edge")
                self.edge_items[sorted_edge] = edge_id

        # Draw nodes (as text) on top of edges
        for node_id, data in self.nodes.items():
            x, y = data['x'], self._to_canvas_y(data['y'])
            label = data['label']
            text_id = self.create_text(x, y, text=label, anchor=tk.CENTER, tags=("node", node_id))
            self.node_items[node_id] = text_id

    def _get_edge_endpoint(self, x1, y1, x2, y2, x_min, y_min, x_max, y_max):
        """
        Calculates the endpoint of an edge that intersects the bounding box of a text.
        """
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return x1, y1  # Avoid division by zero for self-loops (though unlikely here)

        # Calculate intersection points with the bounding box edges
        t_values = []
        if dx != 0:
            t_left = (x_min - x1) / dx
            t_right = (x_max - x1) / dx
            t_values.extend([t_left, t_right])
        if dy != 0:
            t_top = (y_min - y1) / dy
            t_bottom = (y_max - y1) / dy
            t_values.extend([t_top, t_bottom])

        # Filter for t values between 0 and 1 (intersection along the line segment)
        valid_t = [t for t in t_values if 0 <= t <= 1]

        # Find the intersection point closest to the text center (t=0)
        closest_t = -1
        min_dist_sq = float('inf')

        for t in valid_t:
            ix = x1 + t * dx
            iy = y1 + t * dy
            dist_sq = (ix - (x_min + x_max) / 2)**2 + (iy - (y_min + y_max) / 2)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_t = t

        if closest_t != -1:
            return x1 + closest_t * dx, y1 + closest_t * dy
        else:
            # If no intersection with the bounding box, return the center
            return x1, y1

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
        # ... (modify this to use _get_edge_endpoint as well if edges need to dynamically adjust during drag)
        updated_edges = set()
        for id1, id2 in self.edges:
            if id1 in self.nodes and id2 in self.nodes:
                x1_center, y1_center = self.nodes[id1]['x'], self._to_canvas_y(self.nodes[id1]['y'])
                x2_center, y2_center = self.nodes[id2]['x'], self._to_canvas_y(self.nodes[id2]['y'])

                text_id1 = self.create_text(x1_center, y1_center, text=self.nodes[id1]['label'], anchor=tk.CENTER, tags=("temp"))
                bbox1 = self.bbox(text_id1)
                self.delete(text_id1)
                x1_min, y1_min, x1_max, y1_max = bbox1

                text_id2 = self.create_text(x2_center, y2_center, text=self.nodes[id2]['label'], anchor=tk.CENTER, tags=("temp"))
                bbox2 = self.bbox(text_id2)
                self.delete(text_id2)
                x2_min, y2_min, x2_max, y2_max = bbox2

                x1, y1 = self._get_edge_endpoint(x1_center, y1_center, x2_center, y2_center, x1_min, y1_min, x1_max, y1_max)
                x2, y2 = self._get_edge_endpoint(x2_center, y2_center, x1_center, y1_center, x2_min, y2_min, x2_max, y2_max)

                sorted_edge = tuple(sorted((id1, id2)))
                if sorted_edge in self.edge_items and sorted_edge not in updated_edges:
                    self.coords(self.edge_items[sorted_edge], x1, y1, x2, y2)
                    updated_edges.add(sorted_edge)

# Example Usage
if __name__ == "__main__":
    import matplotlib
    from pypf.networkData import networkData
    from pypf.utility import get_test_pf
    pfn = get_test_pf("bio")

    width = 600
    height = 600
    nodes, edges = networkData(pfn, "gravity", canvas_width=width, canvas_height=height)

    root = tk.Tk()
    root.title(pfn.name)
    root.bordersize = 5
    network_display = NetworkDisplay(root, nodes, edges, width=width, height=height)
    network_display.pack(fill=tk.BOTH, expand=True)
    root.mainloop()