# pypf/shownet.py
# assembles network data and displays network
import tkinter as tk
from pypf.pfnet import PFnet
from pypf.networkDisplay import NetworkDisplay
from pypf.utility import map_coords_to_frame

def shownet(parent_window, net:PFnet, method="gravity", width=600, height=600):
    # get nodes and edges and layout data
    #nodes, edges = networkData(net, method, width, height)

    net.get_layout(method=method)
    newxy = map_coords_to_frame(canonicalxy=net.coords, width=width, height=height, squeezex=.2, squeezey=.2)
    nodes = {}
    for i in range(net.nnodes):
        nodes[net.terms[i]] = {'x': newxy[i][0], 'y': newxy[i][1], 'label': net.terms[i]}

    edges = list(net.graph.edges())

    # set up and show the display
    # Use Toplevel instead of Tk() for a secondary window
    display_window = tk.Toplevel(parent_window)
    display_window.title(net.name)

    # Set initial geometry if desired, but let it expand with content
    display_window.geometry(f"{width}x{height}")
    display_window.grid_rowconfigure(0, weight=1)
    display_window.grid_columnconfigure(0, weight=1)

    network_display = NetworkDisplay(display_window, nodes={}, edges={}, is_directed=net.isdirected, width=width,
                                     height=height) # Create NetworkDisplay first
    network_display.pack(fill=tk.BOTH, expand=True)
    network_display.nodes = nodes # Update NetworkDisplay with correct data
    network_display.edges = edges
    network_display.draw_network() # Redraw the network

    # IMPORTANT: Do NOT call root.mainloop() here.
    # The parent_window's mainloop will handle this Toplevel window.

    # You might want to add a resize binding for the new window if you want dynamic resizing
    # (similar to what you had in gui.py's _on_network_display_resize, but adapted)
    # def _on_resize(event):
    #     try:
    #         # Recalculate layout based on new canvas size
    #         # Note: Using 'gravity' or the original method for resize is simpler
    #         # than 'force' which might re-run layout calculations unnecessarily.
    #         new_nodes, new_edges = networkData(net, method=method, # Use the original method or 'gravity'
    #                                            canvas_width=event.width,
    #                                            canvas_height=event.height)
    #         network_display.nodes = new_nodes
    #         network_display.edges = new_edges
    #         network_display.canvas_height = event.height # Update internal canvas height for correct y-axis inversion
    #         network_display.draw_network() # Redraw the network
    #     except Exception as e:
    #         print(f"Error during network redraw on resize in shownet: {e}")
    #
    # network_display.bind("<Configure>", _on_resize)

if __name__ == "__main__":
    from pypf.utility import get_test_pf
    from pypf.pfnet import PFnet

    test_root = tk.Tk()
    test_root.withdraw() # Hide the main test root window
    nt = get_test_pf("psy")
    nt = PFnet(nt.proximity, type="pf")
    nt.netprint()
    shownet(test_root, net=nt, method="neato") # Pass the test_root as parent
    test_root.mainloop() # This mainloop will now keep the Toplevel window open