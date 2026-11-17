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

# Saved layout coords (canonical range about -1..1) keep 4 decimal places:
# far finer than a pixel, and it keeps project files small.
COORD_DECIMALS = 4


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
        """Saves the net's current coords as the layout for all nets with its terms, and
        switches this net to the "saved" layout, so a project file reloads it that way.
        Other nets with the same terms keep their own layout until "saved" is chosen."""
        if net.termsid and net.coords is not None:
            self.saved_coords[net.termsid] = np.round(np.array(net.coords, dtype=float), COORD_DECIMALS)
            self.apply_saved_layout(net)

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

    def add_copy_of(self, other: "Collection") -> tuple[list[Proximity], list[PFnet]]:
        """Adds copies of other's proximities and networks, with the saved layouts it has
        that this collection lacks. Names already in use get a number added.
        Returns (the proximities added, the networks added); other is not changed."""
        other = copy.deepcopy(other)
        for tid, terms in other.terms.items():
            self.terms.setdefault(tid, terms)
        prx_names, net_names = {}, {}
        for name, prx in other.proximities.items():
            self.add_proximity(prx, add_pfnet=False)
            prx_names[name] = prx.name
        for name, net in other.pfnets.items():
            self.add_pfnet(net)
            net_names[name] = net.name
            if getattr(net, "graph", None) is not None:
                net.graph.name = net.name
        # sources name items of other; follow any renaming
        for prx in other.proximities.values():
            prx.sources = [prx_names.get(s, s) for s in prx.sources]
        for net in other.pfnets.values():
            net.sources = [net_names.get(s, s) for s in net.sources]
        for tid, coords in other.saved_coords.items():
            self.saved_coords.setdefault(tid, coords.copy())
        return list(other.proximities.values()), list(other.pfnets.values())

    def dependents(self, prx_names=(), net_names=()) -> tuple[list[str], list[str]]:
        """What depends on the named proximities and networks: networks derived from
        a proximity, and merged networks that used any of those networks.
        Averages are not dependents: they keep their own distances, so they (and their
        networks) outlive the proximities they were made from.
        Returns (proximity names, network names), not including the names given.
        The proximity list is always empty; it is kept so callers need not change."""
        prxs, nets = set(prx_names), set(net_names)
        changed = True
        while changed:
            changed = False
            for name, net in self.pfnets.items():
                if name in nets:
                    continue
                parent = net.proximity
                if (parent is not None and parent.name in prxs) or nets.intersection(net.sources):
                    nets.add(name)
                    changed = True
        return [], sorted_keys({n: self.pfnets[n] for n in nets - set(net_names) if n in self.pfnets})

    def delete_proximities(self, names) -> tuple[list[str], list[str]]:
        """Deletes the named proximities and the networks that descend from them (see
        dependents). Averages made from them are kept.
        Returns (proximity names, network names) of the dependents also deleted."""
        extra_prxs, extra_nets = self.dependents(prx_names=names)
        for name in names:
            self.proximities.pop(name, None)
        for name in extra_nets:
            self.pfnets.pop(name, None)
        self.prune_orphaned_terms()
        return extra_prxs, extra_nets

    def add_pfnet(self, pf: PFnet) -> None:
        unique_name = get_unique_name(pf.name, self.pfnets.keys())
        pf.name = unique_name
        if pf.name in self.pfnets:
            pf.name = pf.name + "_dup"
        self.pfnets[pf.name] = pf

    def delete_pfnets(self, names) -> list[str]:
        """Deletes the named networks and the merged networks that used them.
        Returns the names of the merged networks also deleted."""
        _, extra_nets = self.dependents(net_names=names)
        for name in [*names, *extra_nets]:
            self.pfnets.pop(name, None)
        self.prune_orphaned_terms()
        return extra_nets

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
        problem = self._same_terms_problem(self.pfnets, names, "merge")
        if problem is None and any(self.pfnets[n].type == "mg" for n in names):
            # a merged net's link weights code its sources; merging again would lose them
            problem = "Merged networks can't be merged again."
        return problem

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

    def merge_networks(self, names, mrg_name="", full_name: str | None = None) -> PFnet:
        """Merges the named networks into a new network, adds it, and returns it.
        It is named "~~mg<count>_<mrg_name>" (mrg_name defaults to the names joined by "|"),
        or full_name if given (used when a project rebuilds a saved merge).
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

        merged_net.name = full_name or f"~~mg{n_nets}_{mrg_name}"
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
            # first, so it is easy to find; keyed by the termsids under "terms"
            "saved_layouts": {tid: np.round(coords, COORD_DECIMALS).tolist()
                              for tid, coords in self.saved_coords.items()},
            "terms": {tid: list(terms) for tid, terms in self.terms.items()},
            "proximities": [],
            "network_recipes": [],
        }

        # 1. Proximities ordered by termsid and name
        for name in sorted_keys(self.proximities):
            prx = self.proximities[name]
            state["proximities"].append({
                "name": prx.name,
                "termsid": prx.termsid,
                "dismat": prx.dismat.tolist(),
                "issymmetric": bool(prx.issymmetric),
                "sources": list(prx.sources),  # proximities an average was made from
            })

        def safe_inf_check(v):
            if isinstance(v, str):
                return v.lower() == "inf"
            return isinstance(v, (int, float)) and np.isinf(v)

        # 2. Network recipes ordered by termsid and name.
        #    A merged network is saved as the names of the networks it merged.
        for name in sorted_keys(self.pfnets):
            net = self.pfnets[name]
            if net.type == "mg":
                state["network_recipes"].append({
                    "name": net.name,
                    "type": "mg",
                    "sources": list(net.sources),
                    "layout": getattr(net, "layout_request", "auto"),
                })
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

    @classmethod
    def from_project_state(cls, json_data: str) -> "Collection":
        """A new Collection holding just the project in json_data (see get_project_state)."""
        col = cls()
        col.load_project_state(json_data)
        return col

    def load_project_state(self, json_data: str) -> None:
        """Adds the project in json_data to this collection, rebuilding its networks from
        their recipes. Names already in use get a number added. To replace the collection
        instead, use Collection.from_project_state."""

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
            new_prx.sources = list(p_data.get("sources", []))  # renamed below
            self.add_proximity(new_prx, add_pfnet=False)
            prx_name_map[name] = new_prx.name

        # Sources use the names in the file; follow any renaming. Sources that were
        # deleted before saving keep their names, as a record of what was averaged.
        for name in prx_name_map.values():
            prx = self.proximities[name]
            prx.sources = [prx_name_map.get(s, s) for s in prx.sources]

        # 3. Restore Networks from Recipes
        net_name_map = {}
        merges = []
        for r_data in data.get("network_recipes", []):
            net_type = r_data.get("type")
            if net_type == "mg":
                merges.append(r_data)  # rebuilt below, once the networks they use exist
                continue
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
                net_name_map[r_data["name"]] = net.name

        # 4. Rebuild merged networks. Older project files have merge recipes without
        #    sources; those, and merges whose networks are missing, are skipped.
        for r_data in merges:
            sources = [net_name_map.get(s) for s in r_data.get("sources", [])]
            if len(sources) < 2 or None in sources or self.merge_problem(sources):
                continue
            net = self.merge_networks(sources, full_name=r_data["name"])
            # the merge starts with the layout of its first network; redo it if another was chosen
            layout = r_data.get("layout", "auto")
            if layout == "saved":
                self.apply_saved_layout(net)
            elif layout and layout != getattr(net, "layout_request", "auto"):
                net.get_layout(method=layout)
