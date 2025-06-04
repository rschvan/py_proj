# from typing import Dict
from pypf.proximity import Proximity
from pypf.pfnet import PFnet

class Collection:

    def __init__(self):
        self.termid = ""
        self.proximities = {}
        self.pfnets = {}

    def add_proximity(self, prx:Proximity):
        if prx.name in self.proximities:
            print(prx.name, "already exists in proximities, replacing it.")
        self.proximities[prx.name] = prx

    def add_pfnet(self, pf:PFnet):
        if pf.name in self.pfnets:
            print(pf.name, "already exists in pfnets, replacing it.")
        self.pfnets[pf.name] = pf

    def get_proximity(self, name:str):
        return self.proximities[name]

    def get_pfnet(self, name:str):
        return self.pfnets[name]

if __name__ == "__main__":
    # from pypf.proximity import Proximity
    # from pypf.pfnet import PFnet
    col = Collection()
    px = Proximity("data/psy.prx.xlsx")
    col.add_proximity(px)
    pfn = PFnet(px)
    pfnq2 = PFnet(px, q=2)

    col.add_pfnet(pfn)
    col.add_pfnet(pfnq2)
    print(col)
    print(col.pfnets)
    print(col.proximities['psy'])
