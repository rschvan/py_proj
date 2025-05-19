from pypf.toy import toy
import pypf as pf
import pypf.utility as util
import networkx as nx

g = nx.balanced_tree(2, 3)

print(dir(util))

# print("pf: ",dir(pf))
#
# x = 7
# y = 22
# print("x: ", x, "y: ", y)
#
# s, d = toy(x, y)
# print("s: ", s, "d: ", d)
#
# x = (s - d)/2
# y = d + x
# print("x: ",x, "y: ", y)
