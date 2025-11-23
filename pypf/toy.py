import networkx as nx
import numpy as np
import sys
import os
from pypf.pfnet import PFnet, get_test_pf
from pypf.utility import graph_from_adjmat

def paths():
    # Get the absolute path of the directory containing 'toy.py'
    print(f" This file: {__file__}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current Dir: {current_dir}")
    print(f"os.pardir: {os.pardir} ")
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    print(f"Parent Dir: {parent_dir}")
    print(f"Home Dir: {os.environ['HOMEPATH']}")

    # Get the parent directory (which should be 'py_proj')
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    print(f"Project Root: {project_root}")
    # Add the project root to sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def layout_method(self):
    import tkinter as tk
    self.layout = tk.StringVar()  # Initialize as blank
    self.layout_combobox = ttk.Combobox(
        project_dir_frame,
        textvariable=self.layout,
        values=["gravity", "neato", "distance", "dot"],  # Set dropdown values
        state="readonly")
    self.layout_combobox.grid(row=1, column=1, padx=5, pady=2, sticky="ew")



if __name__ == "__main__":
    paths()
    # net:PFnet = get_test_pf("cities")
    # gr = net.graph
    # dis = net.dismat.copy()
    # cgr = graph_from_adjmat(dis, net.terms)
    # layout = nx.forceatlas2_layout(cgr, strong_gravity=True, seed=np.random.randint(10000), scaling_ratio=2.0,
    #                                max_iter=10000, pos=None, weight=cgr.edges.data("weight"))
    # print(cgr)
    # print(cgr.edges.data("weight"))

    #
    # # ag = nx.nx_agraph.to_agraph(net.graph)
    # lo = graphviz_layout(net.graph, prog="neato", root=None)
    # print(lo)

    # dot = graphviz.Digraph('my_graph', comment='A simple graph')
    #
    # dot.node('A', 'Node A')
    # dot.node('B', 'Node B')
    # dot.edge('A', 'B', label='Edge A to B')
    #
    # dot.render('my_graph', view=True)

    # import os
    # from os.path import join
    #
    # print(os.path.abspath(__file__))
    # print(os.path.dirname(os.path.abspath(__file__)))
    # print(join("C:",os.environ['HOMEPATH']))
    # print(os.getcwd())
    # #print(os.environ.keys())