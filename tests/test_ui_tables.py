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
