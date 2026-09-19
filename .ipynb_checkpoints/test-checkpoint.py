# test.py
import numpy as np
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
import pypf as pf
import os.path
import networkx as nx
import matplotlib.pyplot as plt
import _tkinter

TclError = _tkinter.TclError

all = False
pic = False

prx = Proximity()
dis1 = prx.dismat
prx.prxprint(note="dialog")
net = pf.PFnet(prx)
net.netprint(note="dialog")

# prx = Proximity(os.path.join("data","psy.prx.csv"))
# prx.prxprint(note="filepath")
# net = PFnet(prx)
# net.netprint(note="filepath")
# prx = Proximity(os.path.join("data","psynct.prx.xlsx"))
# prx.prxprint(note="filepath-nct")
# net = PFnet(prx, 2)
# net.netprint(note=prx.filename)
# prx2 = Proximity(terms=prx.terms, dismat=prx.dismat)
# prx2.prxprint(note="variables")
# net = PFnet(prx2)
# net.netprint(note=prx.filename)

if all:
    prx = Proximity(os.path.join("data", "psy.prx.xlsx"))
    prx.prxprint("file provided")

    prx2 = Proximity(terms=prx.terms, dismat=prx.dismat)
    prx2.prxprint("terms and dismat given")

    pf1 = PFnet(prx2)
    pf1.netprint("no q or r")
    pf2 = PFnet(prx2, 2)
    pf2.netprint("q = 2")
    pf3 = PFnet(prx, np.inf)
    pf3.netprint("q = Inf")

if pic:
    graph = nx.from_numpy_array(net.adjmat)
    labels = {i: term for i, term in enumerate(net.terms)}
    pos = nx.spring_layout(graph)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(graph, pos)
    plt.show(graph)