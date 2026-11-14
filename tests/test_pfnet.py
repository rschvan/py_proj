import numpy as np
import pytest

from pypf.pfnet import PFnet
from conftest import load

INF = np.inf

# link counts for PFnet(q, r) on the sample data: (inf, inf), (2, inf), (inf, 1), (2, 2)
LINKS = {
    "bank6": [5, 6, 12, 9],
    "psy": [25, 32, 103, 47],
    "bio": [26, 35, 141, 62],
    "cities": [25, 26, 322, 36],
    "statecaps": [47, 55, 658, 88],
    "psyinf": [26, 37, 103, 51],
}
QR = [(INF, INF), (2, INF), (INF, 1), (2, 2)]


@pytest.mark.parametrize("name", LINKS)
def test_pathfinder_link_counts(name):
    prx = load(name)
    counts = [PFnet(prx, q=q, r=r).nlinks for q, r in QR]
    assert counts == LINKS[name]


@pytest.mark.parametrize("name, nlinks", [("bank6", 6), ("psy", 27), ("bio", 28), ("cities", 26)])
def test_nearest_neighbor(name, nlinks):
    net = PFnet(load(name), type="nn")
    assert net.nlinks == nlinks
    assert net.isdirected
    assert net.name == f"{name}_nn"


def test_minimum_network_properties():
    net = PFnet(load("psy"))
    assert net.name == "psy_pf"
    assert not net.isdirected and net.isconnected and net.is_planar
    assert net.eccentricity["radius"] == 5 and net.eccentricity["diameter"] == 9
    assert max(net.links.values()) == 13
    assert net.adjmat.shape == (25, 25)
    assert len(net.graph.edges) == net.nlinks


def test_names_show_parameters():
    prx = load("bio")
    assert PFnet(prx, q=2).name == "bio_pf_q2"
    assert PFnet(prx, q=2, r=2).name == "bio_pf_q2_r2"


def test_more_links_with_smaller_q_and_r():
    prx = load("bio")
    base = PFnet(prx)
    for q, r in [(2, INF), (INF, 1), (2, 2)]:
        net = PFnet(prx, q=q, r=r)
        assert net.nlinks >= base.nlinks
        # the minimum network's links are in every Pathfinder network
        assert np.all(net.adjmat[base.adjmat > 0] > 0)


@pytest.mark.parametrize("q, r, nlinks", [(INF, INF, 24), (2, INF, 27), (INF, 1, 42)])
def test_directed(q, r, nlinks):
    net = PFnet(load("assoc"), q=q, r=r)
    assert net.isdirected and net.nlinks == nlinks


def test_directed_nearest_neighbor():
    assert PFnet(load("assoc"), type="nn").nlinks == 10


def test_link_list_matches_links():
    net = PFnet(load("bio"))
    links = net.get_link_list()
    assert len(links["source"]) == net.nlinks


def test_disconnected_net():
    from pypf.proximity import Proximity
    dis = np.array([[0, 1, INF], [1, 0, INF], [INF, INF, 0]])
    net = PFnet(Proximity(terms=["a", "b", "c"], dismat=dis, name="abc"))
    assert net.nlinks == 1
    assert net.links == {(0, 1): 1}
    assert not net.isconnected


def test_invalid_arguments():
    with pytest.raises(TypeError):
        PFnet()
    with pytest.raises(ValueError):
        PFnet(load("bank6"), type="bogus")


def test_set_coords():
    net = PFnet(load("bank6"))
    xy = np.arange(12, dtype=float).reshape(6, 2)
    net.set_coords(xy)
    np.testing.assert_array_equal(net.coords, xy)
    assert net.layout_request == "saved"
    with pytest.raises(ValueError):
        net.set_coords(np.zeros((5, 2)))
