# pypf/__init__.py
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.utility import pwcorr, coherence, discorr, netsim, get_lower, get_parent_dir, get_test_pf
from pypf.networkDisplay import NetworkDisplay
from pypf.networkData import networkData
from pypf.shownet import shownet

# TODO
#  arrows
#  labels
#  border
#  multiselect files
"""
import pypf as pf may have circular import issues
would allow expressions like:
self = pf.Proximity()
self = pf.PFnet(self, q=2)
coh = pf.coherence(dismat)
cor = pf.pwcorr(dismat)
"""
