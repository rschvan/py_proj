import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
from typing import Dict, Tuple
from matplotlib.transforms import Bbox

from pypf.pfnet import PFnet
import streamlit as st

class Netpic:
    """
    A class for creating and managing a network plot using Matplotlib.
    The controls are managed externally.
    """
    def __init__(self, net: PFnet):
        self.net = net
        self.nodes = net.terms
        self.adjmat = net.adjmat
        self.is_directed = net.isdirected
        self.adjmat_norm = self.adjmat / np.max(self.adjmat)
        self.fig, self.ax = None, None

        # State variables for interactivity
        self.dragging_node = None
        self.drag_start_pos = None

        # Mappings for plot objects and data
        self.text_objects: Dict[str, Text] = {}
        self.node_positions: Dict[str, np.ndarray] = {}
        self.edge_lines: Dict[Tuple, FancyArrowPatch] = {}
        self.weight_texts: Dict[Tuple, Text] = {}

    def get_link_coords_fast(self, s: Text, t: Text) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the start, end, and midpoint of a link connecting the bounding
        boxes of two matplotlib.text.Text objects using an efficient method.

        Args:
            s (Text): The source Text object.
            t (Text): The target Text object.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - start_point (np.ndarray): The [x, y] coordinates for the start of the link.
                - end_point (np.ndarray): The [x, y] coordinates for the end of the link.
                - mid_point (np.ndarray): The [x, y] coordinates for the midpoint of the link.
        """
        fig = self.fig
        ax = self.ax
        renderer = fig.canvas.get_renderer()

        # Get the text positions (which are the centers due to ha/va='center')
        s_pos = s.get_position()
        t_pos = t.get_position()

        # Get bounding boxes in data coordinates
        s_bbox_data = s.get_window_extent(renderer).transformed(ax.transData.inverted())
        t_bbox_data = t.get_window_extent(renderer).transformed(ax.transData.inverted())

        # Define the line connecting the two text centers
        delta_x = t_pos[0] - s_pos[0]
        delta_y = t_pos[1] - s_pos[1]

        # Handle vertical and horizontal lines
        if delta_x == 0:
            slope = np.inf
            y_intercept = s_pos[0]  # For a vertical line, the 'intercept' is the x-value
        else:
            slope = delta_y / delta_x
            y_intercept = s_pos[1] - slope * s_pos[0]

        def _find_intersection(bbox: Bbox, direction_vector: tuple[float, float]) -> np.ndarray:
            """
            Finds the single intersection point of a line (defined by slope and intercept)
            with a bounding box, based on the direction of the vector.
            """
            left, bottom, right, top = bbox.extents
            dx, dy = direction_vector

            # Check for intersection with vertical or horizontal edges based on slope
            if abs(slope) < (top - bottom) / (right - left):
                # Intersects left or right side
                if dx > 0:
                    x_intersect = right
                else:
                    x_intersect = left
                y_intersect = y_intercept + slope * x_intersect

                if bottom <= y_intersect <= top:
                    return np.array([x_intersect, y_intersect])

            # Intersects top or bottom side
            if dy > 0:
                y_intersect = top
            else:
                y_intersect = bottom

            # Handle vertical line case
            if slope == np.inf:
                x_intersect = s_pos[0]
            else:
                x_intersect = (y_intersect - y_intercept) / slope

            return np.array([x_intersect, y_intersect])

        # Find the intersection points for source and target
        start_point = _find_intersection(s_bbox_data, (delta_x, delta_y))
        end_point = _find_intersection(t_bbox_data, (-delta_x, -delta_y))  # Use reversed direction for target

        mid_point = (start_point + end_point) / 2
        return start_point, end_point, mid_point

    def create_view(self, font_size=10):
        """
        Creates and returns a matplotlib figure with the network plot.
        The plot's properties are determined by the input arguments.
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-1.1, 1.1)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.axis('off')

        self.text_objects.clear()
        self.node_positions.clear()
        coords = self.net.coords
        if coords is None:
            coords = np.random.rand(len(self.nodes), 2) * 2 - 1
        # draw nodes as text
        for i, node in enumerate(self.nodes):
            x, y = coords[i]
            text_obj = self.ax.text(x, y, node, ha='center', va='center', fontsize=font_size,
                                    bbox=dict(facecolor='white', alpha=0, edgecolor='white',
                                    boxstyle='round,pad=0.3',),
                                    picker=True, alpha=1, zorder=2, visible=True,)
            self.text_objects[node] = text_obj
            self.node_positions[node] = np.array([x, y])

        self._draw_edges_and_weights(font_size)
        self.ax.margins(0.1)

        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)

        return self.fig

    def _draw_edges_and_weights(self, font_size):
        """Redraws edges and weights based on current node positions."""
        for line in self.edge_lines.values():
            line.remove()
        for text in self.weight_texts.values():
            text.remove()
        self.edge_lines.clear()
        self.weight_texts.clear()

        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if not self.is_directed and i > j:
                    continue
                weight = self.adjmat[i, j]
                if weight > 0:
                    start_point, end_point, midpoint = (self.get_link_coords_fast(
                        self.text_objects[self.nodes[i]],
                        self.text_objects[self.nodes[j]]))
                    color = 'blue'
                    # alpha = self.adjmat_norm[i, j]
                    if self.is_directed:
                        edge = FancyArrowPatch(
                            start_point, end_point,
                            mutation_scale=15,
                            color=color,
                            alpha=1,
                            connectionstyle="arc3,rad=0",
                            zorder=0,
                            arrowstyle='->',
                            shrinkA=1, shrinkB=1,
                        )
                    else:
                        edge = PathPatch(
                            Path([start_point, end_point]),
                            facecolor='none',
                            edgecolor=color,
                            alpha=1,
                            lw=1,
                            zorder=0
                        )

                    self.ax.add_patch(edge)
                    self.edge_lines[(self.nodes[i], self.nodes[j])] = edge

                    weight_text = self.ax.text(
                        midpoint[0], midpoint[1],
                        f'{weight:.0f}',
                        ha='center', va='center',
                        fontsize=font_size * 0.8,
                        bbox=dict(facecolor='white', alpha=1, edgecolor='white',
                                  boxstyle='round,pad=0.3'),
                        zorder=1,
                        visible=False,
                    )
                    self.weight_texts[(self.nodes[i], self.nodes[j])] = weight_text
        self.fig.canvas.draw_idle()

    def toggle_weights(self):
        for text in self.weight_texts.values():
            text.set_visible(not text.get_visible())

    def change_font_size(self, font_size):
        for text in self.text_objects.values():
            text.set_fontsize(font_size)
        for text in self.weight_texts.values():
            text.set_fontsize(font_size * 0.8)
        self.fig.canvas.draw_idle()

    # The drag-and-drop methods remain the same as they handle in-plot interactivity
    def _on_press(self, event):
        if event.inaxes != self.ax: return
        for text_obj in self.text_objects.values():
            if text_obj.contains(event)[0]:
                self.dragging_node = text_obj.get_text()
                self.drag_start_pos = (event.xdata, event.ydata)
                break

    def _on_motion(self, event):
        if self.dragging_node is None or event.inaxes != self.ax: return
        dx = event.xdata - self.drag_start_pos[0]
        dy = event.ydata - self.drag_start_pos[1]
        node_idx = self.nodes.index(self.dragging_node)
        self.net.coords[node_idx] += [dx, dy]
        self.node_positions[self.dragging_node] += [dx, dy]
        self.text_objects[self.dragging_node].set_position(self.node_positions[self.dragging_node])
        self.drag_start_pos = (event.xdata, event.ydata)
        self.create_view(font_size=st.session_state.font_size)

    def _on_release(self):
        self.dragging_node = None
        self.drag_start_pos = None
        self.create_view(font_size=st.session_state.font_size)
