# Display helpers in stutils (the Streamlit app's formatting of pypf results)
import numpy as np
import pytest

from pypf.collection import Collection
from pypf.pfnet import PFnet
from conftest import load

stutils = pytest.importorskip("stutils")


@pytest.mark.parametrize("x, out", [(67.00667, 67), (48.07333, 48.1), (0.92483, 0.925),
                                    (2070.86, 2071), (0, 0), (np.nan, np.nan)])
def test_sig_figs(x, out):
    got = stutils.sig_figs(x)
    assert (np.isnan(got) and np.isnan(out)) or got == out


def test_tables_look_as_before():
    col = Collection()
    col.add_proximity(load("bio"))
    col.add_pfnet(PFnet(col.proximities["bio"], q=2))
    col.add_pfnet(PFnet(col.proximities["bio"], type="nn"))
    prx = stutils.proximity_table(col)
    assert prx.loc[0, "mean"] == 67 and prx.loc[0, "sd"] == 23.3
    nets = stutils.network_table(col).set_index("name")
    assert nets.loc["bio_pf", ["q", "r"]].tolist() == ["inf", "inf"]
    assert nets.loc["bio_pf_q2", "q"] == "2" and nets.loc["bio_nn", "q"] == ""
    props = stutils.properties_table(col.pfnets["bio_pf"])
    centers = [c for c in props["center"] if c]
    assert centers and all(c in props["node"].tolist() for c in centers)


def test_sample_tables():
    assert list(stutils.get_sample_sources()["file"])[:3] == ["bank6", "bio", "psy"]
    ex = stutils.get_example_file_format()
    assert ex.loc["bank", "river"] == 14 and ex.shape == (6, 6)


def test_weight_labels():
    netview = pytest.importorskip("netview")
    col = Collection()
    for name in ["psy", "bio"]:
        col.add_proximity(load(name))
    bio = col.pfnets["bio_pf"]
    (i, j) = next(iter(bio.links))
    assert netview.weight_label(bio, bio.adjmat[i, j]) == f"{bio.dismat[i, j]:.4g}"
    assert netview.weight_label(bio, 2070.86) == "2071"

    merged = col.merge_networks(["psy_pf", "bio_pf"])
    labels = {netview.weight_label(merged, merged.adjmat[i, j]) for i, j in merged.links}
    assert labels <= {"01", "10", "11"} and len(labels) > 1
    assert set(merged.get_link_list()["nets"]) == labels  # Link List uses the same code


def test_merge_legend():
    netview = pytest.importorskip("netview")
    col = Collection()
    for name in ["psy", "bio"]:
        col.add_proximity(load(name))
    col.add_pfnet(PFnet(col.proximities["psy"], q=2))
    merged = col.merge_networks(["psy_pf", "bio_pf", "psy_pf_q2"])
    assert netview.merge_legend(merged) == [["001", "psy_pf"], ["010", "bio_pf"], ["100", "psy_pf_q2"]]
    assert netview.merge_legend(col.pfnets["psy_pf"]) == []
