# from typing import Dict
from pypf.proximity import Proximity
from pypf.pfnet import PFnet

class Collection:

    def __init__(self):
        self.proximities = {}
        self.pfnets = {}

    def add_proximity(self, prx:Proximity):
        name = prx.name
        if name in self.proximities:
            print(name, "already exists in proximities, replacing it.")
        self.proximities[prx.name] = prx

    def add_pfnet(self, pf:PFnet):
        name = pf.name
        if name in self.pfnets:
            print(name, "already exists in pfnets, replacing it.")
        self.pfnets[name] = pf

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
    col.add_pfnet(pfn)
    print(col.pfnets)
    print(col.proximities['psy'])
