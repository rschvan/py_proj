import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.xlim = (-1, 1)
ax.ylim = (-1, 1)

def node(x, y, label):
    n = plt.text(x, y, label, ha='center', va='center', fontsize=12, figure=fig, axes=ax)
    return n


for i in range(10):
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)
    n = node(x, y, str(i))
    print(n)

# plt.text(0.2, 0.2, 'word')
# plt.text(0.5, 0.5, 'mid')
# plt.text(0.8, 0.8, 'upper')
plt.show()

# fig, ax = plt.subplots()
# ax.xlim = (-1, 1)
# ax.ylim = (-1, 1)
# ax.rescale_data = False
# ax.plot(-0.5, -0.5, label='word')
# ax.plot(.0, .0, label='mid')
# ax.plot(.5, .5, label='upper')
# ax.rescale_data = False
# plt.show()

