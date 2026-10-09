import igraph as ig
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
import numpy as np
import os

def prxnet(name):
    pfile = os.path.join("data", name + ".prx.xlsx")
    prx = Proximity(filepath=pfile)
    netw = PFnet(proximity=prx)
    return netw

def bit_coding(n):
    for i in range(n):
        print(f"{i}  binary: {bin(i)}  length: {i.bit_length()} bit count: {i.bit_count()}")

def test_coords(tnet:PFnet, algorithm:str):
    adj = tnet.adjmat.tolist()
    gr = ig.Graph.Weighted_Adjacency(adj)
    layo = gr.layout(algorithm)
    ig.config['plotting.backend'] = 'matplotlib'
    ig.plot(gr)
    npa = np.array(layo.coords, dtype=float)
    print(algorithm)
    print(type(npa))
    print(npa)
    return layo

if __name__ == '__main__':
    net = prxnet("psy")
    lo = test_coords(tnet=net, algorithm="reingold_tilford_circular")
    print(type(lo))

"""
    gr.layout_reingold_tilford_circular()
    gr.layout_kamada_kawai()
    gr.layout_davidson_harel(seed=None, maxiter=10, fineiter=-1, cool_fact=0.75, weight_node_dist=1.0,
                                  weight_border=0.0, weight_edge_lengths=-1, weight_edge_crossings=-1,
                                  weight_node_edge_dist=-1)
    gr.layout_reingold_tilford(mode='out', root=None, rootlevel=None) # tree with defaults
    gr.layout_reingold_tilford_circular(mode='out', root=None, rootlevel=None) # circular tree
"""