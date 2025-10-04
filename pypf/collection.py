import numpy as np
import pandas as pd
import os
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.utility import mypfnets_dir

class Collection:

    def __init__(self, to_csv:bool=False):
        self.proximities: dict[str, Proximity] = {}
        self.pfnets: dict[str, PFnet] = {}
        self.selected_prxs: list[str] = []
        self.selected_nets: list[str] = []
        self.mypfnets_dir: str = mypfnets_dir() # directory in user's directory for sample data and help
        self.project_dirs: list[str] = [] # list of directories
        self.project_dirs.append(self.mypfnets_dir)
        self.to_csv = to_csv # if true output to csv files will be generated and files opened

    def add_proximity(self, prx:Proximity):
        if prx.name in self.proximities:
            return
        else:
            self.proximities[prx.name] = prx

    def delete_proximity(self):
        selected = self.selected_prxs
        for prx_name in selected:
            self.proximities.pop(prx_name)
        self.selected_prxs = []

    def add_pfnet(self, pf:PFnet):
        if pf.name in self.pfnets:
            return
        else:
            self.pfnets[pf.name] = pf

    def delete_pfnet(self):
        selected = self.selected_nets
        for net_name in selected:
            self.pfnets.pop(net_name)
        self.selected_nets = []

    def get_proximity(self, name:str):
        return self.proximities[name]

    def get_pfnet(self, name:str):
        return self.pfnets[name]

    def get_proximity_info(self):
        infolist = []
        for prx_name in self.proximities.keys():
            prx = self.proximities[prx_name]
            infolist.append(prx.get_info())
        df = pd.DataFrame(infolist)
        if self.to_csv:
            f = os.path.join(self.mypfnets_dir, "proximity_info.csv")
            df.to_csv(f, index=False)
            os.startfile(f)
        return df

    def get_pfnet_info(self):
        infolist = []
        for net_name in self.pfnets.keys():
            net = self.pfnets[net_name]
            infolist.append(net.get_info())
        df = pd.DataFrame(infolist)
        if self.to_csv:
            f = os.path.join(self.mypfnets_dir, "pfnet_info.csv")
            df.to_csv(f, index=False)
            os.startfile(f)
        return df

    def get_proximity_correlations(self):
        from pypf.utility import discorr
        prxs = list(self.proximities.keys())
        n = len(prxs)
        cors = np.zeros((n,n))
        np.fill_diagonal(cors, 1)
        if n > 1:
            for i in range(n):
                disi = self.proximities[prxs[i]].dismat
                for j in range(n):
                    if i == j: continue
                    disj = self.proximities[prxs[j]].dismat
                    cors[i,j] = discorr(disi, disj)
            df = pd.DataFrame(cors, columns=prxs, index=prxs)
            if self.to_csv:
                f = os.path.join(self.mypfnets_dir, "proximity_correlations.csv")
                df.to_csv(f, index=True)
                os.startfile(f)
            return df
        else:
            print("Too few proximities for correlations.")
            return None

    def average_proximities(self, choice:str="mean"):
        from pypf.utility import average_proximity
        n = len(self.selected_prxs)
        if n < 2:
            print("Not Enough Proximities", "Please select at least two proximities to average.")
            return
        proxes:list[Proximity] = []
        for name in self.selected_prxs:
            proxes.append(self.proximities[name])
        ave_prx = average_proximity(proxes, method=choice)
        if ave_prx is None:
            print("Average Failed", "Proximities not compatible.")
            return
        self.proximities[ave_prx.name] = ave_prx
        self.selected_prxs = []

    def network_similarity(self):
        from pypf.utility import netsim
        netlist = list(self.pfnets.keys())
        n = len(netlist)
        if n > 1:
            simmat = np.zeros((n,n))
            for i in range(n):
                inet = self.pfnets[netlist[i]]
                for j in range(n):
                    jnet = self.pfnets[netlist[j]]
                    if i == j:
                        simmat[i,j] = 1
                    elif inet.nnodes != jnet.nnodes:
                        simmat[i,j] = np.nan
                    else:
                        sim = netsim(inet.adjmat, jnet.adjmat)
                        simmat[i,j] = sim["similarity"]
            df = pd.DataFrame(simmat, index=netlist, columns=netlist)
            if self.to_csv:
                f = os.path.join(self.mypfnets_dir, "net_sim.csv")
                pd.DataFrame(simmat, index=netlist, columns=netlist).to_csv(f)
                os.startfile(f)
            return df
        else:
            print("Not Enough Networks", "Must have at least two networks for similarity.")
            return None

    def network_link_list(self):
        if self.selected_nets:
            # show link list for the first selected network
            pf_name = self.selected_nets[0]
            pf = self.pfnets[pf_name]
            link_list = list(pf.graph.edges())
            df = pd.DataFrame(link_list, columns=["from", "to"])
            if self.to_csv:
                f = os.path.join(self.mypfnets_dir, "link_list.csv")
                df.to_csv(f)
                os.startfile(f)
            return df
        else:
            print("No Network Selected", "Please select one or more networks first.")
            return None

    def merge_networks(self):
        from pypf.utility import merge_networks
        netnames = self.selected_nets
        nets:list[PFnet] = []
        if len(netnames) > 1:
            for name in netnames:
                nets.append(self.pfnets[name])
            mrg = merge_networks(nets)
            self.pfnets[mrg.name] = mrg
            self.selected_nets = []
        else:
            print("Not Enough Networks.", "Please select at least two nets.")

if __name__ == "__main__":
    from pypf.proximity import Proximity
    from pypf.pfnet import PFnet

    col = Collection()
    px = Proximity("data/psy.prx.xlsx")

    col.add_proximity(px)
    bx = Proximity("data/bio.prx.xlsx")
    col.add_proximity(bx)
    ppf = PFnet(px)
    col.add_pfnet(ppf)
    bpf = PFnet(bx)
    col.add_pfnet(bpf)
    col.selected_prxs = ["psy", "bio"]

    col.selected_nets = ["psy_pf", "bio_pf"]
    #
    # col.network_link_list()
    # col.merge_networks()
    # col.average_proximities()
    # col.network_similarity()
    # col.get_proximity_correlations()
    # col.get_proximity_info()
    # col.get_pfnet_info()
    # print(col.pfnets)