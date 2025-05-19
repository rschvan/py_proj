# pypf/__init__.py
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.utility import pwcorr, coherence, discorr, netsim, get_lower, get_parent_dir
from pypf.networkDisplay import NetworkDisplay
from pypf.networkData import networkData

"""
import pypf as pf may have circular import issues
would allow expressions like:
self = pf.Proximity()
self = pf.PFnet(self, q=2)
coh = pf.coherence(dismat)
cor = pf.pwcorr(dismat)
"""
