#pypf/__init___
"""
PyPathfinder (pypf) is a Python package for creating and analyzing Pathfinder Networks.

It provides tools to:
- Generate network graphs from proximity data files.
- Visualize and interact with network structures.
- Analyze network properties.
"""
__version__ = "0.1.0"

from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.collection import Collection