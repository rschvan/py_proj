import os
from igraph import Graph
import matplotlib.pyplot as plt
import numpy as np
from pypf.pfnet import PFnet
from pypf.proximity import Proximity

# 1. Load test data
data_set = "bank"
pfile = os.path.join("pypf", "data", data_set + ".prx.xlsx")
prx = Proximity(filepath=pfile)
net = PFnet(proximity=prx)

# 2. Build igraph instance
adj_list = net.adjmat.tolist() if hasattr(net.adjmat, "tolist") else net.adjmat
gr = Graph.Weighted_Adjacency(adj_list, mode="undirected", attr="weight")

# 3. Algorithms to evaluate
root = net.eccentricity["center"]
print(root)
print([net.terms[i] for i in root])
algorithms = {
    "sugiyama": ("sugiyama", {}),
    "Fruchterman-Reingold": ("fruchterman_reingold", {"niter": 500}),
    "Radial Tree (Root center)": ("rt_circular", {"root": [0]}),
    "Davidson-Harel": ("davidson_harel", {}),
}

# 4. Plot comparative 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for ax, (title, (alg, kwargs)) in zip(axes, algorithms.items()):
  layout = gr.layout(alg, **kwargs)
  coords = np.array(layout.coords, dtype=float)

  # Draw edges
  for edge in gr.es:
    p1, p2 = coords[edge.source], coords[edge.target]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#888888", lw=1, zorder=1)

  # Draw nodes
  ax.scatter(
      coords[:, 0], coords[:, 1], color="#1f77b4", s=100, zorder=2, edgecolors="k"
  )

  # Label nodes if small N
  for idx, (x, y) in enumerate(coords):
    ax.text(x, y, str(idx), fontsize=8, ha="center", va="center", color="white")

  ax.set_title(title, fontsize=11, fontweight="bold")
  ax.axis("off")

plt.tight_layout()
plt.show()