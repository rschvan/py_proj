import json

import numpy as np
import pytest

from pypf.collection import Collection
from conftest import load


@pytest.fixture
def col():
    c = Collection()
    for name in ["psy", "bio"]:
        c.add_proximity(load(name))
    return c


def test_add_proximity_derives_minimum_net(col):
    assert set(col.proximities) == {"psy", "bio"}
    assert set(col.pfnets) == {"psy_pf", "bio_pf"}
    assert len(col.terms) == 1  # psy and bio share their terms


def test_duplicate_names_are_made_unique(col):
    col.add_proximity(load("psy"))
    assert len(col.proximities) == 3


def test_correlations_and_similarity(col):
    cors = col.get_proximity_correlations()
    assert cors.loc["psy", "bio"] == pytest.approx(0.937, abs=1e-3)
    sim = col.network_similarity()
    assert sim.loc["psy_pf", "bio_pf"] == pytest.approx(0.3784, abs=1e-4)


def test_average_mean(col):
    ave = col.average_proximities(["psy", "bio"], method="mean")
    assert ave.name.endswith("2_psy...bio")
    assert ave.mean == pytest.approx(57.54, abs=1e-3)
    assert ave.sources == ["psy", "bio"]


def test_average_median_replaces_infinities():
    c = Collection()
    for name in ["psy", "psyinf"]:
        c.add_proximity(load(name))
    med = c.average_proximities(["psy", "psyinf"], method="median")
    assert not np.isinf(med.dismat).any()
    mean = c.average_proximities(["psy", "psyinf"], method="mean")
    # with two proximities the median is the mean
    np.testing.assert_allclose(med.dismat, mean.dismat)


def test_average_needs_two(col):
    assert col.average_problem(["psy"])
    with pytest.raises(ValueError):
        col.average_proximities(["psy"])


def test_merge(col):
    merged = col.merge_networks(["psy_pf", "bio_pf"])
    assert merged.name == "~~mg2_psy_pf|bio_pf"
    assert merged.type == "mg" and merged.nlinks == 37
    assert sorted(np.unique(merged.adjmat).tolist()) == [0, 1, 2, 3]
    assert merged.name in col.pfnets


def test_delete_prunes_terms_and_saved_layouts(col):
    col.save_layout(col.pfnets["psy_pf"])
    col.delete_proximities(["psy", "bio"])
    col.delete_pfnets(["psy_pf", "bio_pf"])
    assert not col.terms and not col.saved_coords


def test_saved_layout_shared_by_terms(col):
    psy, bio = col.pfnets["psy_pf"], col.pfnets["bio_pf"]
    col.save_layout(psy)
    assert col.has_saved_layout(bio.termsid)
    assert col.apply_saved_layout(bio)
    np.testing.assert_allclose(bio.coords, psy.coords)


def test_project_round_trip(col):
    col.add_pfnet(__import__("pypf.pfnet", fromlist=["PFnet"]).PFnet(col.proximities["bio"], q=2))
    psy = col.pfnets["psy_pf"]
    psy.coords = psy.coords * 0.5
    col.save_layout(psy)  # also switches psy_pf to "saved"
    col.pfnets["bio_pf_q2"].get_layout(method="spiral")

    state = col.get_project_state()
    data = json.loads(state)
    assert list(data)[0] == "saved_layouts"
    assert set(data["saved_layouts"]) == {psy.termsid}

    restored = Collection()
    restored.load_project_state(state)
    assert set(restored.pfnets) == set(col.pfnets)
    assert restored.pfnets["bio_pf_q2"].nlinks == col.pfnets["bio_pf_q2"].nlinks
    assert restored.pfnets["bio_pf_q2"].layout_request == "spiral"
    assert restored.pfnets["psy_pf"].layout_request == "saved"
    np.testing.assert_allclose(restored.pfnets["psy_pf"].coords, psy.coords)


def test_merged_nets_not_saved(col):
    col.merge_networks(["psy_pf", "bio_pf"])
    names = [r["name"] for r in json.loads(col.get_project_state())["network_recipes"]]
    assert not any(n.startswith("~~mg") for n in names)


def test_sample_project_loads(sample_json):
    c = Collection()
    c.load_project_state(sample_json)
    assert "bio_pf" in c.pfnets and len(c.pfnets) == 9


def test_merge_and_average_problems(col):
    col.add_proximity(load("cities"))
    assert col.merge_problem(["psy_pf", "bio_pf"]) is None
    assert "same terms" in col.merge_problem(["psy_pf", "cities_pf"])
    assert "Not found" in col.merge_problem(["psy_pf", "nope"])
    assert col.average_problem(["psy", "bio"]) is None
    assert "same terms" in col.average_problem(["psy", "cities"])
    with pytest.raises(ValueError):
        col.merge_networks(["psy_pf", "cities_pf"])


def test_merged_net_has_its_own_eccentricity(col):
    merged = col.merge_networks(["psy_pf", "bio_pf"])
    assert merged.eccentricity == merged.get_eccentricity()
    assert merged.eccentricity != col.pfnets["psy_pf"].eccentricity


def test_deleting_proximity_keeps_its_networks(col):
    col.delete_proximities(["psy"])
    assert "psy" not in col.proximities and "psy_pf" in col.pfnets
    assert len(col.terms) == 1  # still used by the networks


def test_no_ui_state_in_collection():
    c = Collection()
    for attr in ("selected_prxs", "selected_nets", "focus_net"):
        assert not hasattr(c, attr)


def test_save_layout_switches_that_net_to_saved(col):
    psy, bio = col.pfnets["psy_pf"], col.pfnets["bio_pf"]
    psy.coords = psy.coords * 0.5
    col.save_layout(psy)
    assert psy.layout_request == "saved"
    np.testing.assert_allclose(psy.layout_coords, psy.coords)  # Reset returns here
    assert bio.layout_request != "saved"                         # same terms, not switched


def test_saved_layouts_are_rounded(col):
    psy = col.pfnets["psy_pf"]
    psy.coords = psy.coords / 3.0  # plenty of decimals
    col.save_layout(psy)
    saved = col.saved_coords[psy.termsid]
    np.testing.assert_array_equal(saved, np.round(saved, 4))
    for row in json.loads(col.get_project_state())["saved_layouts"][psy.termsid]:
        assert all(len(f"{v:.10g}".split(".")[-1]) <= 4 for v in row)
