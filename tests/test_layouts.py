import numpy as np
import pytest

from pypf import layouts
from pypf.pfnet import PFnet
from conftest import load

METHODS = ["auto", "gravity", "kamada_kawai", "tree", "radial_tree", "mds_distance",
           "mds_graph", "spring", "spiral", "planar", "sugiyama", "circle"]


@pytest.fixture(scope="module")
def bio_net():
    return PFnet(load("bio"))


@pytest.mark.parametrize("method", METHODS)
def test_layout_methods(bio_net, method):
    coords, used = layouts.get_coords(bio_net, method=method)
    assert coords.shape == (bio_net.nnodes, 2)
    assert np.all(np.isfinite(coords))
    assert np.all(np.abs(coords) <= 1 + 1e-9)  # canonical range
    assert isinstance(used, str)


def test_layout_falls_back_to_auto_on_error(bio_net, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(layouts, "find_center_roots", broken)
    coords, used = layouts.get_coords(bio_net, method="tree")
    assert coords.shape == (bio_net.nnodes, 2)
    assert used == "kamada_kawai"  # what "auto" picks for a 25-node non-tree


def test_get_layout_records_request(bio_net):
    bio_net.get_layout(method="spiral")
    assert bio_net.layout_request == "spiral"
    np.testing.assert_array_equal(bio_net.coords, bio_net.layout_coords)
