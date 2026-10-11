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


if __name__ == '__main__':
    n = 5
    mat = np.full((n, n), fill_value=np.nan)
    np.fill_diagonal(mat, None)
    k = 1
    for i in range(n):
        for j in range(i+1, n):
            v = k
            if j % 2 == 0:  # j is even
                v = None
            mat[i, j] = v
            mat[j, i] = v
            k = k + 1
    print(mat)

    #net = prxnet("psy")
