# pypf/__init__.py
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.coherence import coherence

"""
import pypf as pf
would allow expressions like:
self = pf.Proximity()
self = pf.PFnet(self, q=2)

"""