# assembles network data and displays network
import tkinter as tk
from pypf.pfnet import PFnet
from pypf.networkDisplay import NetworkDisplay
from pypf.networkData import networkData

def shownet(net:PFnet, method="gravity", width=600, height=600):
    # get nodes and edges with layout data
    nodes, edges = networkData(net, method, width, height)

    # set up and show the display
    root = tk.Tk()
    root.title(net.name)
    network_display = NetworkDisplay(root, nodes={}, edges=[], width=width,
                                     height=height)  # Create NetworkDisplay first
    network_display.pack(fill=tk.BOTH, expand=True)
    network_display.nodes = nodes  # Update NetworkDisplay with correct data
    network_display.edges = edges
    network_display.draw_network()  # Redraw the network
    root.mainloop()

if __name__ == "__main__":
    from pypf.utility import get_test_pf
    nt = get_test_pf("psy")
    nt = PFnet(nt.proximity, r=2)
    nt.netprint()
    shownet(nt)