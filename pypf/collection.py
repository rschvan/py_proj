# pypf/collection.py
import copy
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
        self.selected_prxs: list[str] = []  # proximity names
        self.selected_nets: list[str] = []  # network names
        self.mypfnets_dir: str = mypfnets_dir() # directory in user's directory for sample data and help
        self.project_dirs: list[str] = [] # list of directories
        self.project_dirs.append(self.mypfnets_dir)
        self.to_csv = to_csv # if true output to csv files will be generated and files opened

    def add_proximity(self, prx:Proximity):
        if prx.name in self.proximities:
            prx.name = prx.name + "_dup"
        self.proximities[prx.name] = prx

    def delete_proximity(self):
        selected = self.selected_prxs
        for prx_name in selected:
            self.proximities.pop(prx_name)
        self.selected_prxs = []

    def add_pfnet(self, pf:PFnet) -> None:
        if pf.name in self.pfnets:
            pf.name = pf.name + "_dup"
        self.pfnets[pf.name] = pf

    def delete_pfnet(self) -> None:
        selected = self.selected_nets
        for net_name in selected:
            self.pfnets.pop(net_name)
        self.selected_nets = []

    def get_proximity(self, name:str) -> Proximity | None:
        return self.proximities[name]

    def get_pfnet(self, name:str) -> PFnet | None:
        return self.pfnets[name]

    def get_proximity_info(self) -> pd.DataFrame | None:
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

    def get_pfnet_info(self) -> pd.DataFrame | None:
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

    def get_proximity_correlations(self) -> pd.DataFrame | None:
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

    def average_proximities(self, method="mean") -> None:
        from pypf.utility import coherence
        """
        Computes the average proximity matrix based on a list of proximity matrices and a specified method
        (mean or median). The function generates a new proximity object as a result, encapsulating the
        aggregated proximity data.

        :param proxes: A list of Proximity objects for which the average proximity matrix is to be computed.
                       All Proximity objects in the list must have matching terms and dimensions.
        :type proxes: list[Proximity]

        :param method: The method to use for computing the average proximity. Accepted values are "mean"
                       and "median". Defaults to "mean". The "mean" method computes the arithmetic mean
                       across matrices, while the "median" method computes the median element-wise.
        :type method: str, optional

        :return: A new Proximity object containing the aggregated proximity matrix as per the specified
                 method. The resulting proximity includes updated statistics and coherence measurements.
        :rtype: Proximity
        """
        selected = self.selected_prxs
        num = len(selected)
        if num < 2:
            return
        proxes = []
        for name in selected:
            proxes.append(self.proximities[name])
        ave_prx:Proximity = copy.deepcopy(proxes[0])
        ave_prx.filename = None
        ave_prx.filepath = None
        ave_prx.name = method + str(num) + ave_prx.name + "_to_" + proxes[num - 1].name
        for prx in proxes:
            if prx.terms != ave_prx.terms:
                return
        nterms = ave_prx.nterms
        # set ave to square matrix of zeros
        ave = np.zeros((nterms, nterms))
        match method:
            case "mean":
                for prx in proxes:
                    ave_prx.max = max(ave_prx.max, prx.max)
                    ave = ave + prx.dismat
                ave = ave / num
            case "median":
                for i in range(nterms):
                    for j in range(nterms):
                        ave[i, j] = np.median([prx.dismat[i, j] for prx in proxes])
        issymmetric = np.equal(ave, ave.T).all()
        ave_prx.dismat = ave
        ave_prx.issymmetric = issymmetric
        ave_prx.calculate_stats()
        ave_prx.coh = coherence(ave_prx.dismat, ave_prx.max)
        self.add_proximity(ave_prx)

    def network_similarity(self) -> pd.DataFrame | None:
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

    def network_link_list(self) -> pd.DataFrame | None:
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

    def merge_networks(self) -> PFnet:
        from pypf.utility import graph_from_adjmat
        net_names = self.selected_nets
        n_nets = len(net_names)
        if n_nets < 2:
            print("Not Enough Networks", "Please select at least two networks to merge.")
            return
        merged_net = copy.deepcopy(self.pfnets[net_names[0]])
        merged_net.name = "|".join(net_names)
        merged_net.name = "merge" + str(n_nets) + merged_net.name
        merged_net.graph.name = merged_net.name
        merged_net.type = "merge"
        adj = merged_net.adjmat.astype(bool).astype(int)
        for i in range(1, n_nets):
            if merged_net.terms != self.pfnets[net_names[i]].terms:
                print("Merge Failed", "Networks do not have matching terms.")
                return
            add = self.pfnets[net_names[i]].adjmat.astype(bool).astype(int)
            adj = adj + (add * (2**i))
        merged_net.adjmat = adj
        merged_net.nlinks = adj.astype(bool).sum()
        merged_net.dismat  = None
        merged_net.mindis = None
        merged_net.maxdis = None
        merged_net.q = None
        merged_net.r = None
        merged_net.isdirected = not np.array_equal(adj, adj.T)
        if not merged_net.isdirected:
            merged_net.nlinks = int(merged_net.nlinks / 2)
        merged_net.graph = graph_from_adjmat(adj, merged_net.terms)
        merged_net.graph.name = merged_net.name
        merged_net.get_eccentricity()
        self.add_pfnet(merged_net)
        self.selected_nets = []
        return merged_net

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

    mrg = col.merge_networks()
    props = mrg.get_network_properties()
    tab = pd.DataFrame(props)
    print(tab)
