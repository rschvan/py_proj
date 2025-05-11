# pypf/__init__.py
from pypf.proximity import Proximity
from pypf.pfnet import PFnet

"""
import pypf as pf
would allow expressions like:
self = pf.Proximity()
self = pf.PFnet(self, q=2)
coh = pf.coherence(dis)
cor = pf.pwcorr(dis)

"""