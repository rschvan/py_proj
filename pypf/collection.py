# pypf/collection.py
import copy
import json
import networkx as nx
import numpy as np
import pandas as pd

from pypf.proximity import Proximity
from pypf.pfnet import PFnet
from pypf.utility import (get_termsid, sorted_keys, get_unique_name,
                          get_link_ranks, compute_merged_net_ranks)


class Collection:

    def __init__(self):
        self.terms: dict[str, tuple[str, ...]] = {}  # termsid -> ('term1', 'term2', ...)
        self.proximities: dict[str, Proximity] = {}
        self.pfnets: dict[str, PFnet] = {}
        # termsid -> (nterms x 2) layout coords, shared by every net with those terms
        self.saved_coords: dict[str, np.ndarray] = {}

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
            self.saved_coords.pop(tid, None)

    def save_layout(self, net: PFnet) -> None:
        """Saves the net's current coords as the layout for all nets with its terms."""
        if net.termsid and net.coords is not None:
            self.saved_coords[net.termsid] = np.array(net.coords, dtype=float).copy()

    def has_saved_layout(self, termsid: str) -> bool:
        return termsid in self.saved_coords

    def apply_saved_layout(self, net: PFnet) -> bool:
        """Gives the net the saved layout for its terms. Returns False if there is none."""
        coords = self.saved_coords.get(net.termsid)
        if coords is None or coords.shape != (net.nnodes, 2):
            return False
        net.set_coords(coords, method="saved")
        return True

    def add_proximity(self, prx: Proximity, add_pfnet=True) -> None:
        unique_name = get_unique_name(prx.name, self.proximities.keys())
        prx.name = unique_name
        if prx.name in self.proximities:
            prx.name = prx.name + "_dup"
        if prx.termsid not in self.terms.keys():
            self.terms[prx.termsid] = tuple(prx.terms)
        prx.terms = self.terms[prx.termsid]
        self.proximities[prx.name] = prx
        if add_pfnet:
            pfn = PFnet(proximity=prx)
            self.add_pfnet(pfn)

    def add_files(self, files: dict[str, bytes], kind: str = "spreadsheet",
                  force_symmetric: bool = False) -> tuple[list[str], list[str]]:
        """Reads proximity files and adds them, each with its minimum Pathfinder network.

        files: file name -> contents. kind: "spreadsheet" (.xlsx/.xls/.csv) or "legacy"
            (text data files plus their terms files; see pypf.readers.read_legacy_files).
        force_symmetric: make asymmetric proximities symmetric (min of dij and dji).
        Returns (names of the proximities added, an error message for each file that failed).
        """
        from pypf.readers import read_files

        proximities, errors = read_files(files, kind=kind)
        added = []
        for prx in proximities:
            if force_symmetric:
                prx = prx.symmetrized()
            self.add_proximity(prx)
            added.append(prx.name)
        return added, errors

    def delete_proximities(self, names) -> None:
        """Deletes the named proximities. Networks derived from them are kept."""
        for name in names:
            self.proximities.pop(name, None)
        self.prune_orphaned_terms()

    def add_pfnet(self, pf: PFnet) -> None:
        unique_name = get_unique_name(pf.name, self.pfnets.keys())
        pf.name = unique_name
        if pf.name in self.pfnets:
            pf.name = pf.name + "_dup"
        self.pfnets[pf.name] = pf

    def delete_pfnets(self, names) -> None:
        """Deletes the named networks."""
        for name in names:
            self.pfnets.pop(name, None)
        self.prune_orphaned_terms()

    def _same_terms_problem(self, items: dict, names, action: str, min_count: int = 2) -> str | None:
        names = list(names)
        if len(names) < min_count:
            return f"Select {min_count} or more items with the same terms to {action}."
        missing = [n for n in names if n not in items]
        if missing:
            return f"Not found: {', '.join(missing)}"
        if len({items[n].termsid for n in names}) > 1:
            return f"All items must have the same terms to {action}."
        return None

    def average_problem(self, names) -> str | None:
        """Why the named proximities cannot be averaged, or None if they can."""
        return self._same_terms_problem(self.proximities, names, "average")

    def merge_problem(self, names) -> str | None:
        """Why the named networks cannot be merged, or None if they can."""
        return self._same_terms_problem(self.pfnets, names, "merge")

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
                prxi = self.proximities[prxs[i]]
                disi = prxi.dismat
                for j in range(i+1, n):
                    prxj = self.proximities[prxs[j]]
                    disj = prxj.dismat
                    if prxi.termsid != prxj.termsid:
                        pcor = np.nan
                    else:
                        pcor = discorr(disi, disj)
                    cors[i, j] = pcor
                    cors[j, i] = pcor

            return pd.DataFrame(cors, columns=prxs, index=prxs)
        return None

    def average_proximities(self, names, ave_name=None, method="mean") -> Proximity:
        """Returns a new proximity averaging the named ones (it is not added to the collection).
        Raises ValueError if they cannot be averaged."""
        from pypf.utility import coherence

        def finite_distances(prx: Proximity) -> np.ndarray:
            """Copy of the distances with infinities replaced by 1.1 x that proximity's max."""
            dismat = np.array(prx.dismat, dtype=float)
            dismat[np.isinf(dismat)] = prx.max * 1.1
            return dismat

        if method not in ("mean", "median"):
            raise ValueError(f"method must be 'mean' or 'median', not {method!r}")
        selected = list(names)
        problem = self.average_problem(selected)
        if problem:
            raise ValueError(problem)
        num = len(selected)

        proxes = [self.proximities[name] for name in selected]
        first_prx = proxes[0]

        ave_prx: Proximity = copy.deepcopy(first_prx)
        ave_prx.filename = None
        ave_prx.filepath = None
        ave_prx.sources = selected

        prefix = "!\u0078\u0305" if method == "mean" else "!mdn"
        if not ave_name:
            ave_name = ave_prx.name + "..." + proxes[num - 1].name
        ave_prx.name = f"{prefix}{num}_{ave_name}"

        # both methods treat an infinite distance as 1.1 x the largest distance of its proximity
        stack = np.stack([finite_distances(prx) for prx in proxes])
        ave = stack.mean(axis=0) if method == "mean" else np.median(stack, axis=0)

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
            np.fill_diagonal(simmat, 1)
            for i in range(n):
                inet = self.pfnets[netlist[i]]
                for j in range(i+1,n):
                    jnet = self.pfnets[netlist[j]]
                    if inet.termsid != jnet.termsid:
                        psim = np.nan
                    else:
                        sim = netsim(inet.adjmat, jnet.adjmat)
                        psim = sim["similarity"]
                    simmat[i, j] = psim
                    simmat[j, i] = psim

            return pd.DataFrame(simmat, index=netlist, columns=netlist)
        return None

    def merge_networks(self, names, mrg_name="") -> PFnet:
        """Merges the named networks into a new network, adds it, and returns it.
        Raises ValueError if they cannot be merged."""
        from pypf.utility import graph_from_adjmat

        net_names = list(names)
        problem = self.merge_problem(net_names)
        if problem:
            raise ValueError(problem)
        n_nets = len(net_names)

        first_net = self.pfnets[net_names[0]]
        first_tid = first_net.termsid

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
        merged_net.eccentricity = merged_net.get_eccentricity()
        merged_net.is_planar = nx.is_planar(merged_net.graph)
        links = merged_net.adjmat.astype(bool)
        merged_net.links, unique_distances = compute_merged_net_ranks(merged_net.adjmat, links, merged_net.isdirected )
        merged_net.unique_weights = unique_distances
        self.add_pfnet(merged_net)
        return merged_net

    def get_project_state(self) -> str:
        """Serializes the unified terms, proximities, and network recipes (including merges) to JSON."""
        self.prune_orphaned_terms()

        state = {
            "terms": {tid: list(terms) for tid, terms in self.terms.items()},
            "proximities": [],
            "network_recipes": [],
            "saved_layouts": {tid: coords.tolist() for tid, coords in self.saved_coords.items()},
        }

        # 1. Proximities ordered by termsid and name
        for name in sorted_keys(self.proximities):
            prx = self.proximities[name]
            state["proximities"].append({
                "name": prx.name,
                "termsid": prx.termsid,
                "dismat": prx.dismat.tolist(),
                "issymmetric": bool(prx.issymmetric)
            })

        def safe_inf_check(v):
            if isinstance(v, str):
                return v.lower() == "inf"
            return isinstance(v, (int, float)) and np.isinf(v)

        # 2. Network recipes ordered by termsid and name.
        #    Merged networks are not saved; users re-create them after loading.
        for name in sorted_keys(self.pfnets):
            net = self.pfnets[name]
            if net.type == "mg":
                continue
            state["network_recipes"].append({
                "name": net.name,
                "type": net.type,
                "q": "inf" if safe_inf_check(net.q) else net.q,
                "r": "inf" if safe_inf_check(net.r) else net.r,
                "parent_proximity": net.proximity.name if getattr(net, "proximity", None) else None,
                "layout": getattr(net, "layout_request", "auto"),
            })

        return json.dumps(state, indent=4)

    def load_project_state(self, json_data: str) -> None:
        """Reconstructs the collection from JSON, replaying base and merge network recipes."""

        data = json.loads(json_data)

        # 1. Restore shared terms dictionary
        if "terms" in data:
            for tid, terms_list in data["terms"].items():
                self.terms[tid] = tuple(terms_list)

        # Restore saved layouts (older project files have none)
        for tid, coords in data.get("saved_layouts", {}).items():
            self.saved_coords[tid] = np.array(coords, dtype=float)

        # 2. Restore Proximities
        prx_name_map = {}
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

            name = new_prx.name
            self.add_proximity(new_prx, add_pfnet=False)
            prx_name_map[name] = new_prx.name

        # 3. Restore Networks from Recipes
        for r_data in data.get("network_recipes", []):
            net_type = r_data.get("type")
            if net_type == "mg":
                continue  # older project files may contain merges; they are not restored
            # Skip networks whose proximity was deleted before saving
            prx_name = prx_name_map.get(r_data.get("parent_proximity"))
            parent_prx = self.get_proximity(prx_name) if prx_name else None
            if parent_prx:
                q = np.inf if r_data["q"] == "inf" else r_data["q"]
                r = np.inf if r_data["r"] == "inf" else r_data["r"]
                net = PFnet(parent_prx, q=q, r=r, type=net_type)
                # PFnet() computed an "auto" layout; redo it if another one was chosen
                layout = r_data.get("layout", "auto")
                if layout == "saved":
                    self.apply_saved_layout(net)
                elif layout and layout != "auto":
                    net.get_layout(method=layout)
                self.add_pfnet(net)
