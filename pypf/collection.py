# pypf/collection.py
import copy
import json
import os
import networkx as nx
import numpy as np
import pandas as pd

from pypf import pfnet
from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.utility import get_termsid, mypfnets_dir, sorted_keys


class Collection:

    def __init__(self):
        self.terms: dict[str, tuple[str, ...]] = {}  # termsid -> ('term1', 'term2', ...)
        self.proximities: dict[str, Proximity] = {}
        self.pfnets: dict[str, PFnet] = {}
        self.selected_prxs: list[str] = []  # proximity names
        self.selected_nets: list[str] = []  # network names
        self.focus_net: PFnet | None = None
        self.mypfnets_dir: str = mypfnets_dir()
        self.project_dirs: list[str] = [self.mypfnets_dir]

    def register_terms(self, terms: list[str] | tuple[str, ...]) -> str | None:
        """Registers a sequence of terms as an immutable tuple. Returns termsid."""
        if not terms:
            return None
        tid = get_termsid(list(terms))
        if tid and tid not in self.terms:
            self.terms[tid] = tuple(terms)
        return tid

    def get_terms(self, termsid: str) -> tuple[str, ...] | None:
        return self.terms.get(termsid)

    def get_active_termsids(self) -> set[str]:
        """Returns the set of termsids currently referenced by proximities or networks."""
        active = set()
        for prx in self.proximities.values():
            tid = getattr(prx, "termsid", None)
            if tid:
                active.add(tid)
        for net in self.pfnets.values():
            tid = getattr(net, "termsid", None)
            if tid:
                active.add(tid)
        return active

    def prune_orphaned_terms(self) -> None:
        """Removes terms that are no longer referenced by any proximity or network."""
        active = self.get_active_termsids()
        orphaned = [tid for tid in self.terms if tid not in active]
        for tid in orphaned:
            del self.terms[tid]

    def add_proximity(self, prx: Proximity, add_pfnet=True) -> None:
        if prx.name in self.proximities:
            prx.name = prx.name + "_dup"

        tid = self.register_terms(prx.terms)
        if hasattr(prx, "termsid"):
            prx.termsid = tid
        if tid:
            prx.terms = self.terms[tid]

        self.proximities[prx.name] = prx
        if add_pfnet:
            pfn = PFnet(proximity=prx)
            self.add_pfnet(pfn)

    def delete_proximity(self) -> None:
        for prx_name in self.selected_prxs:
            self.proximities.pop(prx_name, None)
        self.selected_prxs = []
        self.prune_orphaned_terms()

    def add_pfnet(self, pf: PFnet) -> None:
        if pf.name in self.pfnets:
            pf.name = pf.name + "_dup"

        if hasattr(pf, "terms") and pf.terms:
            tid = self.register_terms(pf.terms)
            if hasattr(pf, "termsid") and not pf.termsid:
                pf.termsid = tid
            if tid:
                pf.terms = self.terms[tid]

        self.pfnets[pf.name] = pf

    def delete_pfnet(self) -> None:
        for net_name in self.selected_nets:
            self.pfnets.pop(net_name, None)
        self.selected_nets = []
        self.prune_orphaned_terms()

    def get_proximity(self, name: str) -> Proximity | None:
        return self.proximities.get(name)

    def get_pfnet(self, name: str) -> PFnet | None:
        return self.pfnets.get(name)

    def get_proximity_info(self) -> pd.DataFrame | None:
        infolist = []
        for prx_name in sorted_keys(self.proximities):
            prx = self.proximities[prx_name]
            infolist.append(prx.get_info())
        return pd.DataFrame(infolist) if infolist else None

    def get_pfnet_info(self) -> pd.DataFrame | None:
        infolist = []
        for net_name in sorted_keys(self.pfnets):
            net = self.pfnets[net_name]
            infolist.append(net.get_info())
        return pd.DataFrame(infolist) if infolist else None

    def get_proximity_correlations(self) -> pd.DataFrame | None:
        from pypf.utility import discorr

        prxs = sorted_keys(self.proximities)
        n = len(prxs)
        cors = np.zeros((n, n))
        np.fill_diagonal(cors, 1)
        if n > 1:
            for i in range(n):
                disi = self.proximities[prxs[i]].dismat
                for j in range(n):
                    if i == j:
                        continue
                    disj = self.proximities[prxs[j]].dismat
                    cors[i, j] = discorr(disi, disj)
            return pd.DataFrame(cors, columns=prxs, index=prxs)
        return None

    def average_proximities(self, ave_name=None, method="mean") -> Proximity | None:
        from pypf.utility import coherence

        selected = self.selected_prxs
        num = len(selected)
        if num < 2:
            return None

        proxes = [self.proximities[name] for name in selected]
        first_prx = proxes[0]
        first_tid = getattr(first_prx, "termsid", None) or get_termsid(list(first_prx.terms))

        for prx in proxes[1:]:
            prx_tid = getattr(prx, "termsid", None) or get_termsid(list(prx.terms))
            if prx_tid != first_tid:
                return None

        ave_prx: Proximity = copy.deepcopy(first_prx)
        ave_prx.filename = None
        ave_prx.filepath = None
        ave_prx.sources = selected

        prefix = "~\u0078\u0305" if method == "mean" else "~mdn"
        if not ave_name:
            ave_name = ave_prx.name + "..." + proxes[num - 1].name
        ave_prx.name = f"{prefix}{num}_{ave_name}"

        nterms = ave_prx.nterms
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

        ave_prx.dismat = ave
        ave_prx.issymmetric = np.equal(ave, ave.T).all()
        ave_prx.calculate_stats()
        ave_prx.coh = coherence(ave_prx.dismat, ave_prx.max)
        return ave_prx

    def network_similarity(self) -> pd.DataFrame | None:
        from pypf.utility import netsim

        netlist = sorted_keys(self.pfnets)
        n = len(netlist)
        if n > 1:
            simmat = np.zeros((n, n))
            for i in range(n):
                inet = self.pfnets[netlist[i]]
                for j in range(n):
                    jnet = self.pfnets[netlist[j]]
                    if i == j:
                        simmat[i, j] = 1.0
                    elif inet.nnodes != jnet.nnodes:
                        simmat[i, j] = np.nan
                    else:
                        sim = netsim(inet.adjmat, jnet.adjmat)
                        simmat[i, j] = sim["similarity"]

            return pd.DataFrame(simmat, index=netlist, columns=netlist)
        return None

    def network_link_list(self) -> pd.DataFrame | None:
        if self.selected_nets:
            pf_name = self.selected_nets[0]
            pf = self.pfnets[pf_name]
            link_list = list(pf.graph.edges())
            return pd.DataFrame(link_list, columns=["from", "to"])
        return None

    def merge_networks(self, mrg_name="") -> PFnet | None:
        from pypf.utility import graph_from_adjmat

        net_names = self.selected_nets.copy()
        n_nets = len(net_names)
        if n_nets < 2:
            return None

        first_net = self.pfnets[net_names[0]]
        first_tid = getattr(first_net, "termsid", None) or get_termsid(list(first_net.terms))

        for name in net_names[1:]:
            net = self.pfnets[name]
            net_tid = getattr(net, "termsid", None) or get_termsid(list(net.terms))
            if net_tid != first_tid:
                return None

        merged_net = copy.deepcopy(first_net)
        if not mrg_name:
            mrg_name = "|".join(net_names)

        merged_net.name = f"~~mg{n_nets}_{mrg_name}"
        merged_net.graph.name = merged_net.name
        merged_net.type = "mg"
        merged_net.proximity = None
        merged_net.sources = net_names
        adj = merged_net.adjmat.astype(bool).astype(int)
        for i in range(1, n_nets):
            add = self.pfnets[net_names[i]].adjmat.astype(bool).astype(int)
            adj = adj + (add * (2**i))
        merged_net.adjmat = adj
        merged_net.nlinks = adj.astype(bool).sum()
        merged_net.dismat = None
        merged_net.mindis = None
        merged_net.maxdis = None
        merged_net.q = None
        merged_net.r = None
        merged_net.isdirected = not np.array_equal(adj, adj.T)
        if not merged_net.isdirected:
            merged_net.nlinks = int(merged_net.nlinks / 2)

        terms_seq = self.terms.get(first_tid, getattr(merged_net, "terms", ()))
        merged_net.graph = graph_from_adjmat(adj, list(terms_seq))
        merged_net.graph.name = merged_net.name
        merged_net.get_eccentricity()
        merged_net.is_planar = nx.is_planar(merged_net.graph)
        self.add_pfnet(merged_net)
        self.selected_nets = []
        merged_net.unique_weights = list(range(1, n_nets + 1))
        return merged_net

    def get_project_state(self) -> str:
        """Serializes the unified terms, proximities, and network recipes (including merges) to JSON."""
        self.prune_orphaned_terms()

        state = {
            "terms": {tid: list(terms) for tid, terms in self.terms.items()},
            "proximities": [],
            "network_recipes": []
        }

        # 1. Proximities ordered by termsid and name
        for name in sorted_keys(self.proximities):
            prx = self.proximities[name]
            tid = getattr(prx, "termsid", None) or self.register_terms(prx.terms)
            state["proximities"].append({
                "name": prx.name,
                "termsid": tid,
                "dismat": prx.dismat.tolist(),
                "issymmetric": bool(prx.issymmetric)
            })

        def safe_inf_check(v):
            if isinstance(v, str):
                return v.lower() == "inf"
            return isinstance(v, (int, float)) and np.isinf(v)

        # 2. Network Recipes ordered by termsid and name (base nets first, merges after)
        for name in sorted_keys(self.pfnets):
            net = self.pfnets[name]
            if net.type == "mg":
                state["network_recipes"].append({
                    "name": net.name,
                    "type": "mg",
                    "sources": getattr(net, "sources", [])
                })
            else:
                state["network_recipes"].append({
                    "name": net.name,
                    "type": net.type,
                    "q": "inf" if safe_inf_check(net.q) else net.q,
                    "r": "inf" if safe_inf_check(net.r) else net.r,
                    "parent_proximity": (
                        net.proximity.name
                        if hasattr(net, "proximity") and net.proximity
                        else None
                    )
                })

        return json.dumps(state, indent=4)

    def load_project_state(self, json_data: str) -> None:
        """Reconstructs the collection from JSON, replaying base and merge network recipes."""
        data = json.loads(json_data)

        # 1. Restore shared terms dictionary
        if "terms" in data:
            for tid, terms_list in data["terms"].items():
                self.terms[tid] = tuple(terms_list)

        # 2. Restore Proximities
        for p_data in data.get("proximities", []):
            terms_seq = None
            tid = p_data.get("termsid")

            if tid and tid in self.terms:
                terms_seq = self.terms[tid]
            elif "terms" in p_data and p_data["terms"]:
                tid = self.register_terms(p_data["terms"])
                terms_seq = self.terms[tid]

            new_prx = Proximity(
                name=p_data["name"],
                terms=list(terms_seq) if terms_seq else [],
                dismat=np.array(p_data["dismat"])
            )
            if hasattr(new_prx, "termsid"):
                new_prx.termsid = tid
            if tid:
                new_prx.terms = self.terms[tid]

            self.add_proximity(new_prx, add_pfnet=False)

        # 3. Restore Networks from Recipes
        for r_data in data.get("network_recipes", []):
            net_type = r_data.get("type")

            if net_type == "mg":
                # Reconstruct merge network using its recorded source networks
                sources = r_data.get("sources", [])
                if sources and all(src in self.pfnets for src in sources):
                    self.selected_nets = list(sources)
                    mrg_net = self.merge_networks()
                    if mrg_net:
                        # Ensure the exact original name is preserved
                        if mrg_net.name != r_data["name"]:
                            self.pfnets.pop(mrg_net.name, None)
                            mrg_net.name = r_data["name"]
                            mrg_net.graph.name = r_data["name"]
                            self.pfnets[mrg_net.name] = mrg_net
            else:
                parent_prx = self.get_proximity(r_data["parent_proximity"])
                if parent_prx:
                    q = np.inf if r_data["q"] == "inf" else r_data["q"]
                    r = np.inf if r_data["r"] == "inf" else r_data["r"]

                    new_net = PFnet(parent_prx, q=q, r=r, type=net_type)
                    new_net.name = r_data["name"]
                    self.add_pfnet(new_net)