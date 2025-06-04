import pypf.utility as util
import networkx as nx
import numpy as np
import re

s = "3ab'c de-fg_h"
print(s)
# make s a legal variable name
s = re.sub(r'[^a-zA-Z0-9_]', '_', s)
print(s)

