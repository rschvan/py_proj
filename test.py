# test.py
import numpy as np
import pypf as pf
import os.path
import networkx as nx
import matplotlib.pyplot as plt

all = True
pic = False

prx = pf.Proximity()
dis1 = prx.dismat
prx.prxprint(note="dialog")
net = pf.PFnet(prx)
net.netprint(note="dialog")

# prx = pf.Proximity(os.path.join("data","psy.prx.csv"))
# prx.prxprint(note="filepath")
# net = pf.PFnet(prx)
# net.netprint(note="filepath")
# prx = pf.Proximity(os.path.join("data","psynct.prx.xlsx"))
# prx.prxprint(note="filepath-nct")
# net = pf.PFnet(prx, 2)
# net.netprint(note=prx.filename)
# prx2 = pf.Proximity(terms=prx.terms, dismat=prx.dismat)
# prx2.prxprint(note="variables")
# net = pf.PFnet(prx2)
# net.netprint(note=prx.filename)

if all:
    prx = pf.Proximity(os.path.join("data","psy.prx.xlsx"))
    prx.prxprint("file provided")

    prx2 = pf.Proximity(terms=prx.terms, dismat=prx.dismat)
    prx2.prxprint("terms and dismat given")

    pf1 = pf.PFnet(prx2)
    pf1.netprint("no q or r")
    pf2 = pf.PFnet(prx2,2)
    pf2.netprint("q = 2")
    pf3 = pf.PFnet(prx,np.inf)
    pf3.netprint("q = Inf")

if pic:
    graph = nx.from_numpy_array(pf.adjmat)
    labels = {i: term for i, term in enumerate(pf.terms)}
    pos = nx.spring_layout(graph)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(graph, pos)
    plt.show()