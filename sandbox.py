import pypf.utility as util
import networkx as nx
import numpy as np

i = 2
f = 7.2
v = [1,2,4.1]

va = np.asanyarray(v)
print("va", va, type(va), va[1], type(va[1]))

print(i, type(i), f, type(f), v, type(v), v[1], type(v[1]))

# convert v to np.array
npv = np.array(v)
print("npv", npv, type(npv), npv[1], type(npv[1]))

# cast npv to int
npvi = npv.astype(int)
print("npvi", npvi, type(npvi), npvi[1], type(npvi[1]))

# convert npv to list as int
npvl = npv.astype(int).tolist()
print("npvl", npvl, type(npvl), npvl[1], type(npvl[1]))

print(np.asanyarray(v))

